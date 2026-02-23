"""
  In this example, we illustrate how to get Active MOS
  for N2 in this case we introduce all possible orbitals
  other than the ncore which will be held frozen.
  CREATES: Molecular integrals in chemist notation NOT SPINORBITALS
ncas = {
    'Ag': 2,   # 2s sigma_g and 2pz sigma_g
    'B1u': 2,  # 2s sigma_u and 2pz sigma_u
    'B2u': 1,  # 2px pi_u
    'B3u': 1,  # 2py pi_u
    'B2g': 1,  # 2px pi_g*
    'B3g': 1   # 2py pi_g*
}

ncore = {
    'Ag': 1,   # 1s sigma_g
    'B1u': 1   # 1s sigma_u
}
"""

import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import ao2mo
import openfermion as of
import os

#- - - Tensor functions
def save_tensors(b, h0, h1, h2, folder="tensors"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b:2.1f}_tensors.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Tensors saved to {filename}")

def get_active_space_tensors(mc, localized=False, include_all_noncore=False):
    """
    Extract effective active-space Hamiltonian for either:
      - the CASSCF-defined active space (mc.ncas), or
      - a different selected non-core block (all non-core orbitals if include_all_noncore=True).

    Returns:
      h0: float (nuclear repulsion + frozen-core electronic energy)
      h1_active: ndarray (n_active, n_active)  -- CAS-space one-electron integrals
      h2_spatial: ndarray (n_active, n_active, n_active, n_active) -- two-electron integrals in physicist order (pq|rs)
    """
    mo_coeff = mc.mo_coeff.copy()   # full MO matrix (nao, nmo)
    ncore = int(mc.ncore)
    nmo = mo_coeff.shape[1]

    # determine target active count
    if include_all_noncore:
        target_ncas = nmo - ncore
    else:
        target_ncas = int(mc.ncas)

    if target_ncas <= 0:
        raise ValueError("target_ncas must be > 0")

    start = ncore
    stop = ncore + target_ncas

    # Optional: localize only the target block (the block we will treat as 'active' for extraction)
    if localized:
        from pyscf import lo
        active_block = mo_coeff[:, start:stop].copy()
        try:
            loc_block = lo.boys.BF(mc.mol, active_block).kernel()
        except Exception:
            # fallback
            loc_block = lo.PipekMezey(mc.mol, active_block).kernel()
        mo_coeff[:, start:stop] = loc_block

    # -----------------------
    # 1) One-electron (h1) and core energy: call h1e_for_cas via get_h1eff
    #    Pass explicit ncas so the function slices mo_coeff[:, ncore:ncore+ncas] correctly.
    # -----------------------
    # h1_active (in MO active basis) and core_energy (electronic contribution from frozen core)
    h1_active, core_energy = mc.get_h1eff(mo_coeff, ncas=target_ncas, ncore=ncore)

    # -----------------------
    # 2) Two-electron: compute using ao2mo.full with the exact MO block we want.
    #    This mirrors what CASSCF.get_h2eff does internally, but allows any target size.
    # -----------------------
    # mo block for h2 calculation: shape (nao, target_ncas)
    mo_for_h2 = mo_coeff[:, start:stop].copy()

    # Warning: building a full (n,n,n,n) tensor scales as n^4 memory.
    # For n=26, that's ~26^4 ~ 456,976 elements (float64 ~ 3.7 MB) — actually fine,
    # but for larger n this quickly explodes. Still, ao2mo.full may use intermediate memory.
    try:
        # If underlying SCF has precomputed eri available (faster), use it
        if hasattr(mc._scf, "_eri") and getattr(mc._scf, "_eri", None) is not None:
            eri = ao2mo.full(mc._scf._eri, mo_for_h2, max_memory=mc.max_memory)
        else:
            eri = ao2mo.full(mc.mol, mo_for_h2, verbose=mc.verbose, max_memory=mc.max_memory)
    except Exception as e:
        raise RuntimeError("Failed to build two-electron integrals with ao2mo.full: " + str(e)) from e

    # ao2mo.full returns chemists' (ij|kl) ordering as a 4-index array (n,n,n,n).
    # Convert to physicists' (pq|rs) ordering used by OpenFermion: (i,j,k,l) -> transpose(0,2,3,1)
    eri = np.asarray(eri)
    if eri.ndim != 4 or eri.shape != (target_ncas,)*4:
        # If ao2mo returned a packed/flattened format for some reason, try to handle it robustly:
        if eri.ndim == 1:
            n_guess = int(round(eri.size ** 0.25))
            if n_guess**4 == eri.size:
                eri = eri.reshape((n_guess,)*4)
            else:
                raise RuntimeError(f"Unexpected ao2mo.full output shape {eri.shape} (size {eri.size})")
        elif eri.ndim == 2:
            # some variants might return m x m where m=n*(n+1)/2  (packed). Unpack:
            m = eri.shape[0]
            n_guess = int(round((np.sqrt(1 + 8*m) - 1) / 2))
            if n_guess * (n_guess + 1) // 2 == m:
                eri = ao2mo.restore(1, eri, n_guess)
            else:
                raise RuntimeError(f"Unexpected 2D ao2mo.full output shape {eri.shape}")
        else:
            raise RuntimeError(f"Unexpected ao2mo.full output ndim={eri.ndim}, shape={eri.shape}")

    h2_spatial = eri #.transpose(0, 2, 3, 1).copy() #Use that transpose for the physicists notation

    # 3) Constant term: nuclear repulsion + frozen-core electronic energy
    h0 = float(core_energy)

    return h0, h1_active, h2_spatial


ehf = []
emc = []
def run_proj(b, dm, mo, ci=None, return_tensors=False):
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = './N2_tensors_ccpvdz/N2-%2.1f.out' % b
    mol.atom = [
        ['N', (0.0, 0.0, -b/2)],
        ['N', (0, 0, b/2)],
    ]
    mol.basis = 'ccpvdz'#'sto-3g'
    mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8

    # Use the previous density matrix to speed up SCF convergence
    ehf_val = mf.scf(dm)
    ehf.append(ehf_val)

    mc = mcscf.CASSCF(mf, 8, 10)
    mc.fcisolver.conv_tol = 1e-7
    mc.fcisolver.threads = 1
#    mc.fcisolver.nroots = 10

    if mo is None:
        # --- INITIAL POINT SETUP ---
        ncas = {
            'Ag': 2,  # 2s sigma_g and 2pz sigma_g
            'B1u': 2,  # 2s sigma_u and 2pz sigma_u
            'B2u': 1,  # 2px pi_u
            'B3u': 1,  # 2py pi_u
            'B2g': 1,  # 2px pi_g*
            'B3g': 1  # 2py pi_g*
        }

        ncore = {
            'Ag': 1,  # 1s sigma_g
            'B1u': 1  # 1s sigma_u
        }

        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    else:
        # --- PROJECTION STEP ---
        # Project MOs from the previous geometry onto the current basis
        mo = mcscf.project_init_guess(mc, mo)
 
    # Run CASSCF using projected MOs and previous CI vector
    # Kernel returns: Total Energy, E_cas, CI_vector, MO_coeffs, MO_energies
    e_tot, _, ci, mo, _ = mc.kernel(mo, ci)
    emc.append(e_tot)

    mc.analyze()

    # Return DM for next RHF, and MO/CI for next CASSCF and spinorbital tensors
    if return_tensors is True:
        # Simplified extraction
        h0_spatial, h1_spatial, h2_spatial = get_active_space_tensors(mc, True, True)

        # Convert to Spin-Orbital Tensors using OpenFermion
        #h1_spin, h2_spin = of.ops.representations.get_tensors_from_integrals(h1_spatial, h2_spatial)

        return mf.make_rdm1(), mo, ci, h0_spatial, h1_spatial, h2_spatial #h1_spin, h2_spin
    else:
        return mf.make_rdm1(), mo, ci



# --- Scan Logic ---
dm = mo = ci = None
nuc_separation = np.arange(1.0, 3.01, .1)
# Forward scan
for b in nuc_separation:
    dm, mo, ci = run_proj(b, dm, mo, ci, False)

# Backward scan re-uses last dm/mo/ci from the end of the forward scan
# Collect tensors for JWmapping
dir = "N2_tensors_ccpvdz"
for b in reversed(nuc_separation):
    dm, mo, ci, H0, H1, H2 = run_proj(b, dm, mo, ci, True)
    print(f"Tensors at {b} distance:")
    print("H0 =  nucrep + core_electronic_E")
    print("%2.4f   %2.6f" %(b, H0))
    #print("- - - -")
    #print(H1)
    #print("- - - -")
    #print(H2)
    #print("- - - -")
    print(f"Saving tensors to folder {dir}")
    save_tensors(b, H0, H1, H2, folder=dir)

# --- Data Processing and Plotting ---
x = nuc_separation
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf2.reverse()
emc2.reverse()

with open('../tests/N2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(8o,10e)      HF 3.0->1.5     CAS(12,12)\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,1.5->3.0')
plt.plot(x, ehf2, label='HF,3.0->1.5')
plt.plot(x, emc1, label='CAS(8,10),1.5->3.0')
plt.plot(x, emc2, label='CAS(8,10),3.0->1.5')
plt.legend()
plt.show()