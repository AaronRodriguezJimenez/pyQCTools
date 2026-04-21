# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import pyqctools as pq
import numpy as np

# 1. Define Molecule
benzene = pq.benzene()
mol = gto.M(atom=benzene,
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
pi_orbital_space = [17,20,21,22,23,30]
ncas_orbitals = 6
ncas_electrons = 6

# --- state-averaged CASSCF(6,6) ---
mf.verbose = 4
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
mc.sort_mo(pi_orbital_space)
mc.verbose = 4
mc.fcisolver.nroots = 8
mc.kernel()   # optimize SA orbitals
mc.analyze()