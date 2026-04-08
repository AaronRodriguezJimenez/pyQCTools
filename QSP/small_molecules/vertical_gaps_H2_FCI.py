# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, fci, mcscf
from pyscf.lib import logger
from pyqctools.geometries import H2
import numpy as np

# --- user parameters ---
nstates = 4            # number of lowest singlet states to compute (S0..S(nstates-1))
hartree_to_ev = 27.211386245988  # conversion factor

mol = gto.Mole()
mol.atom = H2()
mol.basis = "sto-3g"
#mol.symmetry = True
mol.verbose = 4
mol.output = "./H2.out"
mol.build()

# --- mean-field ---
mf = scf.RHF(mol)
mf.kernel()

#Before anything check stability:
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
mf = stable_opt_internal(mf, stability_cycles)

# --- FCI
# - - - FCI - - -
myhf = mol.RHF().run()
# create an FCI solver based on the SCF object
cisolver = fci.FCI(myhf)
cisolver.spin = 0
cisolver.nroots = 4
energies = cisolver.kernel()[0]
print(energies)

# sort Energies
order = np.argsort(energies)
energies = energies[order]

# print energies and vertical gaps
print("\nState energies (Hartree) and vertical gaps relative to S0:")
for i, e in enumerate(energies):
    print(f"  S{i}: {e:.12f} Ha   ({e * hartree_to_ev:.6f} eV)")

e0 = energies[0]
print("\nVertical gaps: (a.u.)")
for i in range(1, len(energies)):
    gap_ha = energies[i] - e0
    gap_ev = gap_ha * hartree_to_ev
    print(f"  S{i} - S0: {gap_ha:.12f}")# {gap_ev:.6f}")