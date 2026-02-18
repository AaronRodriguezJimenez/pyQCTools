"""
  In this example, we illustrate how to get Active MOS
  for N2 using a scan technique for getting the correct orbitals
  in the active space.
  CREATES: SPINORBITALS in physicist notation

# Nitrogen (Z=7), Dimer has 14 electrons.
# (10e, 8o) active space leaves 4 electrons (2 orbitals) in the core.
# These core orbitals are the 1s shells.

ncas = {
    'Ag': 2,   # 2s sigma_g and 2pz sigma_g
    'B1u': 2,  # 2s sigma_u and 2pz sigma_u
    'B2u': 1,  # 2px pi_u
    'B3u': 1,  # 2py pi_u
    'B2g': 1,  # 2px pi_g*
    'B3g': 1   # 2py pi_g*
    'Ag': 1,   # 1s sigma_g
    'B1u': 1   # 1s sigma_u
}
"""

import numpy as np
from pyscf import gto
from pyscf import scf, fci
from pyscf import mcscf
import openfermion as of
from pyscf import ao2mo
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

def get_active_space_tensors(mol, mf, localized=False):
    """
    Robustly extracts active space tensors from SymAdaptedCASSCF or CASSCF.
    """
    nuc = mol.energy_nuc()  # Nuclear repulsion.
    ao_kin = mol.intor('int1e_kin')  # Electronic kinetic energy.
    ao_nuc = mol.intor('int1e_nuc')  # Nuclear-electronic attraction.
    ao_obi = ao_kin + ao_nuc  # Single-electron hamiltonian.
    ao_eri = mol.intor('int2e')  # Electonic repulsion interaction.
    C = mf.mo_coeff

    # 1. Handle Localization if requested
    if localized:
        from pyscf import lo
        nocc = int((mf.mo_occ > 0).sum())
        occ_idx_act = np.arange(0, nocc)
        vir_idx_act = np.arange(nocc, len(C))
        # Split MOs into occupied / virtual subsets
        C_act_occ = C[:, occ_idx_act]
        C_act_vir = C[:, vir_idx_act]

        # Boys localization
        loc_occ = lo.Boys(mol, C_act_occ).kernel(verbose=4)
        loc_vir = lo.Boys(mol, C_act_vir).kernel(verbose=4)

        C_loc = np.column_stack((loc_occ, loc_vir))
        mo_coeff_final = C_loc

    else:
        mo_coeff_final = C

    U = mo_coeff_final

    h1_spatial = U.T @ ao_obi @ U  # Rotated single-body integrals.
    h2_spatial = ao2mo.incore.full(ao_eri, U)  # Rotated two-body integrals.
    h0 = nuc
    return h0, h1_spatial, h2_spatial

ehf = []
efci = []
def run_proj(b, get_ints):
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = './N2_tensors_rhf_stable/N2-%2.1f.out' % b
    mol.atom = [
        ['N', (0.0, 0.0, -b/2)],
        ['N', (0, 0, b/2)],
    ]
    mol.basis = 'sto-3g'
   # mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8

    #- - Check SCF Stability
    # Loop for optimizing orbitals until stable
    #
    from pyscf.lib import logger

    def stable_opt_internal(mf, stability_cycles):
        log = logger.new_logger(mf)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc = 0
        print("STABLE?", stable)
        while (not stable and cyc < stability_cycles):
            log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
            dm1 = mf.make_rdm1(mo1, mf.mo_occ)
            mf = mf.run(dm1)
            mo1, _, stable, _ = mf.stability(return_status=True)
            cyc += 1
        if not stable:
            log.note('Stability Opt failed after %d attempts' % cyc)
        return mf

    stability_cycles = 10
    mf.run()
    mf = stable_opt_internal(mf, stability_cycles)
    ehf_val = mf.kernel()
    ehf.append(ehf_val)
    print(f"Total HF energy: {ehf_val}")


    # 4. Analyze results (e.g., spin contamination)
    # <S^2> should be 3.75 for a quartet (S=3/2, S(S+1)=3.75)
    ss = mf.spin_square()
    print(f"Spin square <S^2>: {ss[0]}")

    # Get spatial integrals
    if get_ints:
        h0, h1, h2 = get_active_space_tensors(mol, mf, localized=True)
        dir = "N2_tensors_rhf_stable"
        save_tensors(b, h0, h1, h2, dir)


    # - - - FCI - - -
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    efci.append(e_fci)
    return


# --- Scan Logic ---
dm = mo = ci = None

# Forward scan
for b in np.arange(0.5, 3.01, .1):
    run_proj(b, get_ints=False)

# Backward scan re-uses last dm/mo/ci from the end of the forward scan
# Collect tensors for JWmapping

for b in reversed(np.arange(0.5, 3.01, .1)):
    run_proj(b, get_ints=True)

# --- Data Processing and Plotting ---
x = np.arange(0.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = efci[:len(x)]
emc2 = efci[len(x):]
ehf2.reverse()
emc2.reverse()

with open('N2-scan.txt', 'w') as fout:
    fout.write('     RHF 1.5->3.0     FCI      RHF 3.0->1.5     FCI\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))


import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='RHF,1.5->3.0')
plt.plot(x, ehf2, label='RHF,3.0->1.5')
plt.plot(x, emc1, label='FCI,1.5->3.0')
plt.plot(x, emc2, label='FCI,3.0->1.5')
plt.legend()
plt.show()