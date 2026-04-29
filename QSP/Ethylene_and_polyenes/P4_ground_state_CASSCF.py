# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 10            # number of lowest singlet states to compute (S0..S(nstates-1))
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

p4 ="""
  C          -0.64094939320315      0.34244078671707     -0.00019690181473
  C           0.64094530960938     -0.34225201617716     -0.00018775842871
  C          -1.82374003730610     -0.28493236580319     -0.00021150421370
  C           1.82374180055777      0.28510952611721     -0.00021813413914
  C          -3.10744532271736      0.40626693597750     -0.00024618759651
  C           3.10743612759383     -0.40610977428093     -0.00023633759102
  C          -4.28678148801571     -0.21351675996947     -0.00029277593834
  C           4.28678634301912      0.21364501007730     -0.00029086084636
  H          -0.61717351924008      1.43097566619531     -0.00020045782985
  H           0.61716315270417     -1.43078598145873     -0.00016760342236
  H          -1.85106008868652     -1.37332689958226     -0.00021167496664
  H           1.85107501907228      1.37350420413576     -0.00023855362093
  H          -3.07202291405005      1.49388384231799     -0.00023783966652
  H           3.07199687505203     -1.49372664798236     -0.00020785034460
  H          -5.21881564106056      0.33973572025573     -0.00032246992444
  H          -4.35426321364532     -1.29749398049876     -0.00030149248323
  H           5.21880363244361     -0.33963395206832     -0.00030760836150
  H           4.35430335787266      1.29761968602730     -0.00032198881141
"""

mol = gto.M(atom=p4,
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
pi_orbital_space = [25,26,27,28,29,30,36,39]
ncas_orbitals = 8
ncas_electrons = 8

# ---CALCULATION STEP
mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object
mc.mo_coeff = mc.sort_mo(pi_orbital_space)
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.verbose = verbose
mc.fcisolver.nroots = 4
mc.kernel()
mc.analyze()