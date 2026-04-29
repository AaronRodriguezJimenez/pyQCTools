# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 4            # number of lowest singlet states to compute (S0..S(nstates-1))
basis = 'ccpvdz'       # cc-pVDZ
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

#1) --- PySCF input molecular geometry (Angstrom) ---
ethylene ="""
  C          -0.66239743908476     -0.00000011879825      0.00000025727475
  H          -1.23195628735572      0.92408580955377      0.00000003541117
  H          -1.23195584363698     -0.92408658869570     -0.00000029505709
  C           0.66239765285931     -0.00000003323471      0.00000026754285
  H           1.23195601732587      0.92408614541835     -0.00000029781984
  H           1.23195589989228     -0.92408621424346      0.00000003264815
"""

mol = gto.M(atom=ethylene,
            basis='cc-pvdz',
            spin=0
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
pi_orbital_space = [8,9] #pi and pi*
ncas_orbitals = 2
ncas_electrons = 2

# ---CALCULATION STEP
mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object
mc.mo_coeff = mc.sort_mo(pi_orbital_space)
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.verbose = verbose
mc.fcisolver.nroots = 20
mc.kernel()
mc.analyze()
