"""
  In this example, we illustrate how to get Active MOS
  for N2 using a scan technique for getting the correct orbitals
  in the active space.

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

def get_active_space_tensors(mc, localized=False):
    """
    Robustly extracts active space tensors from SymAdaptedCASSCF or CASSCF.
    """
    mo_coeff = mc.mo_coeff
    ncore = mc.ncore
    ncas = mc.ncas

    # 1. Handle Localization if requested
    if localized:
        from pyscf import lo
        # Localize only the active space part of the MOs
        active_mos = mo_coeff[:, ncore:ncore + ncas]
        # Pipek-Mezey is often more stable for d-orbitals than Boys
        loc_obj = lo.PipekMezey(mc.mol, active_mos).kernel()

        mo_coeff_final = mo_coeff.copy()
        mo_coeff_final[:, ncore:ncore + ncas] = loc_obj
    else:
        mo_coeff_final = mo_coeff

    # 2. Get H1 (Effective 1-electron Hamiltonian)
    # This includes kinetic + nuc-attraction + interaction with frozen core
    h1_active, core_energy = mc.get_h1eff(mo_coeff_final)

    # 3. Get H2 (2-electron Integrals in the active space)
    # Result is in chemists' notation (ij|kl)
    h2_active = mc.get_h2eff(mo_coeff_final)
    # Convert to 4-index tensor and transpose to OpenFermion (physicists') notation (pq|rs)
    h2_spatial = ao2mo.restore(1, h2_active, ncas).transpose(0, 2, 3, 1)

    # 4. Total Constant Energy (Nuclear repulsion + Frozen Core energy)
    h0 = mc.energy_nuc() + core_energy

    return h0, h1_active, h2_spatial

ehf = []
emc = []
def run_proj(b, dm, mo, ci=None, return_tensors=False):
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = 'N2-%2.1f.out' % b
    mol.atom = [
        ['N', (0.0, 0.0, -b/2)],
        ['N', (0, 0, b/2)],
    ]
    mol.basis = 'sto-3g'
    mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
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

    # Simplified extraction
    h0_spatial, h1_spatial, h2_spatial = get_active_space_tensors(mc, False)

    # Convert to Spin-Orbital Tensors using OpenFermion
    h1_spin, h2_spin = of.ops.representations.get_tensors_from_integrals(h1_spatial, h2_spatial)

    # Run CASSCF using projected MOs and previous CI vector
    # Kernel returns: Total Energy, E_cas, CI_vector, MO_coeffs, MO_energies
    e_tot, _, ci, mo, _ = mc.kernel(mo, ci)
    emc.append(e_tot)

    mc.analyze()

    # Return DM for next RHF, and MO/CI for next CASSCF and spinorbital tensors
    if return_tensors is True:
        return mf.make_rdm1(), mo, ci, h0_spatial, h1_spin, h2_spin
    else:
        return mf.make_rdm1(), mo, ci



# --- Scan Logic ---
dm = mo = ci = None

# Forward scan
for b in np.arange(0.5, 3.01, .1):
    dm, mo, ci = run_proj(b, dm, mo, ci, False)

# Backward scan re-uses last dm/mo/ci from the end of the forward scan
# Collect tensors for JWmapping
dir = "N2_tensors_sto3g"
for b in reversed(np.arange(0.5, 3.01, .1)):
    dm, mo, ci, H0, H1, H2 = run_proj(b, dm, mo, ci, True)
    print(f"Tensors at {b} distance:")
    print(H0)
    print("- - - -")
    print(H1)
    print("- - - -")
    print(H2)
    print("- - - -")
    print(f"Saving tensors to folder {dir}")
    save_tensors(b, H0, H1, H2, folder=dir)

# --- Data Processing and Plotting ---
x = np.arange(0.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf2.reverse()
emc2.reverse()

with open('N2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      HF 3.0->1.5     CAS(12,12)\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,1.5->3.0')
plt.plot(x, ehf2, label='HF,3.0->1.5')
plt.plot(x, emc1, label='CAS(12,12),1.5->3.0')
plt.plot(x, emc2, label='CAS(12,12),3.0->1.5')
plt.legend()
plt.show()