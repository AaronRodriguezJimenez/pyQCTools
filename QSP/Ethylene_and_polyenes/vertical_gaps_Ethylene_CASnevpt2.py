from pyscf import gto, scf, mcscf, mrpt, lib
import numpy as np

# --- user parameters ---
nstates = 4            # number of lowest singlet states to compute (S0..S(nstates-1))
basis = 'ccpvdz'       # cc-pVDZ
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

# --- molecular geometry (Angstrom) ---
# Simple ethylene geometry; replace with experimental/optimized geometry if desired.
mol = gto.Mole()
mol.atom = """
  C          -0.66239743908476     -0.00000011879825      0.00000025727475
  H          -1.23195628735572      0.92408580955377      0.00000003541117
  H          -1.23195584363698     -0.92408658869570     -0.00000029505709
  C           0.66239765285931     -0.00000003323471      0.00000026754285
  H           1.23195601732587      0.92408614541835     -0.00000029781984
  H           1.23195589989228     -0.92408621424346      0.00000003264815
"""
mol.basis = basis
mol.symmetry = True     # optional
mol.verbose = verbose
mol.build()

# --- mean-field RHF ---
mf = scf.RHF(mol)
mf.verbose = verbose
ehf = mf.kernel()

# --- state-averaged CASSCF(2,2) ---
ncas_orb = 2
ncas_elec = 2
weights = [1.0 / nstates] * nstates

mc_sa = mcscf.CASSCF(mf, ncas_orb, ncas_elec)
mc_sa = mcscf.state_average_(mc_sa, weights)   # state-averaged CASSCF
mc_sa.verbose = verbose
mc_sa.kernel()   # optimize SA orbitals

# Save optimized orbitals from SA-CASSCF
mo_opt = mc_sa.mo_coeff

# --- multi-root CASCI in SA orbitals (to get state-specific CASCI wavefunctions) ---
cas = mcscf.CASCI(mf, ncas_orb, ncas_elec)
cas.fcisolver.nroots = nstates
cas.verbose = verbose
# run CASCI using the SA-optimized orbitals
cas.kernel(mo_opt)

# Extract CASCI energies (state-specific) in Hartree
# cas.e_tot is a list/array of total CASCI energies for each root
casci_energies = np.asarray(cas.e_tot).copy()
order = np.argsort(casci_energies)  # typically already ordered, but be safe
casci_energies = casci_energies[order]

print("\nCASCI state energies (Hartree) and (eV):")
for i, idx in enumerate(order):
    e_cas = casci_energies[i]
    print(f"  S{i} (root {idx}): {e_cas:.12f} Ha   ({e_cas * hartree_to_ev:.6f} eV)")

# --- NEVPT2 corrections for each root ---
nevpt2_corr = []
nevpt2_total = []

print("\nRunning SC-NEVPT2 for each requested state (this may take time)...")
for i, root in enumerate(order):
    # instantiate NEVPT for the CASCI object, specifying the root
    nev = mrpt.NEVPT(cas, root=int(root))   # root must be the index of the CASCI root
    # You can set density_fit=False if you prefer not to use DF (default True for speed)
    # e_corr is the perturbative correction (Delta E from NEVPT2), in Hartree
    e_corr = nev.kernel()
    # total energy = CASCI energy for that root + NEVPT2 correction
    e_tot_nev = cas.e_tot[root] + e_corr
    nevpt2_corr.append(e_corr)
    nevpt2_total.append(e_tot_nev)
    print(f"  root {root} -> NEVPT2 corr = {e_corr:.12f} Ha   ({e_corr * hartree_to_ev:.6f} eV)   total = {e_tot_nev:.12f} Ha")

# reorder NEVPT2 totals to energy-sorted states (consistent with casci_energies ordering)
nevpt2_total = np.asarray(nevpt2_total)[np.argsort(order)]
nevpt2_corr = np.asarray(nevpt2_corr)[np.argsort(order)]

# Print final NEVPT2-corrected energies and vertical gaps relative to S0
print("\nNEVPT2-corrected state energies (Hartree) and (eV):")
for i, e in enumerate(nevpt2_total):
    print(f"  S{i}: {e:.12f} Ha   ({e * hartree_to_ev:.6f} eV)")

e0_nev = nevpt2_total[0]
print("\nVertical gaps (NEVPT2-corrected):")
for i in range(1, len(nevpt2_total)):
    gap_ha = nevpt2_total[i] - e0_nev
    gap_ev = gap_ha * hartree_to_ev
    print(f"  S{i} - S0: {gap_ha:.12f} Ha   ({gap_ev:.6f} eV)")