# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 10            # number of lowest singlet states to compute (S0..S(nstates-1))
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

p2 ="""
  C          -0.61315490669049      0.40023050451111      0.00002076641733
  C           0.61315475271315     -0.40006851258307      0.00002367277570
  C          -1.83714210045915     -0.12164733566620      0.00001829709420
  C           1.83714254733887      0.12180653296845      0.00001919613192
  H          -0.48624341694090      1.48095607361609      0.00002411914816
  H           0.48624148957999     -1.48079361526528      0.00002540979972
  H          -2.72205260425108      0.50449176034317      0.00001983727721
  H          -1.99141667810657     -1.19678576348874      0.00001399071790
  H           2.72205150022975     -0.50433410163189      0.00001731310711
  H           1.99141941658643      1.19694445719635      0.00001539753074
"""

mol = gto.M(atom=p2,
            basis='cc-pvdz',
)

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

#- - Check SCF Stability
    # Loop for optimizing orbitals until stable
    #
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
print(f"Total HF energy: {ehf_val}")

#2) active MO coefficients used by FCI
# pi orbitals
pi_orbital_space = [14,15,16,20]
ncas_orbitals = 4
ncas_electrons = 4

# --- state-averaged CASSCF(2,2) ---
weights = [1.0 / nstates] * nstates  # equal weights for SA-CASSCF
mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.sort_mo(pi_orbital_space)
#mc = mcscf.CASSCF(mf, ncas_orbitals, (ncas_electrons//2, ncas_electrons//2))
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
    print(f"  S{i}: {e:.12f}")#   ({e * hartree_to_ev:.6f} eV)")

e0 = energies[0]
print("\nVertical gaps:")
for i in range(1, len(energies)):
    gap_ha = energies[i] - e0
    gap_ev = gap_ha * hartree_to_ev
    print(f"  S{i} - S0: {gap_ha:.12f}")# Ha   ({gap_ev:.6f} eV)")