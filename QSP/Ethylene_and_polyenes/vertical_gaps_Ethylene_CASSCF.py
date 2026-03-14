# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf, lib
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
mol.spin = 0
mol.basis = basis
mol.symmetry = True     # optional: can help to reduce cost
mol.verbose = verbose
mol.build()

# --- mean-field ---
mf = scf.RHF(mol)
mf.kernel()

# --- state-averaged CASSCF(2,2) ---
ncas_orbitals = 2
ncas_electrons = 2     # CAS(2,2) for ethylene pi/pi* (2 electrons in 2 orbitals)
weights = [1.0 / nstates] * nstates  # equal weights for SA-CASSCF

mc = mcscf.CASSCF(mf, ncas_orbitals, (ncas_electrons//2, ncas_electrons//2))
mc = mcscf.state_average_(mc, weights)   # decorate to be state-averaged
mc.verbose = verbose
mc.kernel()   # optimize SA orbitals
mc.analyze()
# Orbitals after SA-CASSCF
mo_opt = mc.mo_coeff

# --- multi-root CASCI to get individual state energies in the SA orbitals ---
cas = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
cas.fcisolver.nroots = nstates
res = cas.casci(mo_opt)   # returns energies (and CI vectors)
# res may be (energies, ci_vecs, ...) or simply energies depending on version
if isinstance(res, tuple) or isinstance(res, list):
    energies = np.asarray(res[0])  # energies in Hartree
else:
    energies = np.asarray(res)

# sort (CASCI may not return ordered by energy in all setups)
order = np.argsort(energies)
energies = energies[order]

# print energies and vertical gaps
print("\nState energies (Hartree) and vertical gaps relative to S0:")
for i, e in enumerate(energies):
    print(f"  S{i}: {e:.12f} Ha   ({e * hartree_to_ev:.6f} eV)")

e0 = energies[0]
print("\nVertical gaps:")
for i in range(1, len(energies)):
    gap_ha = energies[i] - e0
    gap_ev = gap_ha * hartree_to_ev
    print(f"  S{i} - S0: {gap_ha:.12f} Ha   ({gap_ev:.6f} eV)")