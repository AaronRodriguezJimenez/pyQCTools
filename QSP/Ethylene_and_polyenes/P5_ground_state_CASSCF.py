# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 10            # number of lowest singlet states to compute (S0..S(nstates-1))
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

p5 ="""
  C           0.58961490073832     -0.31823572596657     -0.00013239287219
  C          -0.58786578527106      0.32268863856419     -0.00006916580666
  C           1.87694387322878      0.35323905450744     -0.00003393091609
  C          -1.87518590396563     -0.34880310603299     -0.00017173208932
  C           3.05455025688859     -0.28487532887759     -0.00012850415895
  C          -3.05281190580485      0.28927698591199     -0.00010524227972
  C           4.34389129881139      0.39503714168072     -0.00006220734500
  C          -4.34212766975526     -0.39068725335027     -0.00019602150452
  C           5.51820727333720     -0.23450655615062     -0.00019045772009
  C          -5.51647788514319      0.23879679824294     -0.00011186268253
  H           0.60171697164652     -1.40688834687593     -0.00026577293013
  H          -0.59997683106280      1.41134155582818      0.00007555679414
  H           1.86375032811172      1.44201016357332      0.00011403813551
  H          -1.86197185062676     -1.43757375901188     -0.00030031613749
  H           3.07208452847743     -1.37345986546234     -0.00027374288486
  H          -3.07038109383510      1.37786171302223      0.00003189344542
  H           4.31773951967633      1.48293762333993      0.00009763593399
  H          -4.31592467820843     -1.47858708077716     -0.00033490109885
  H           6.45471677648009      0.31111706905689     -0.00013701048606
  H           5.57680607316528     -1.31899636753312     -0.00035549260403
  H          -6.45295902106873     -0.30687550158092     -0.00018503956251
  H          -5.57513617581984      1.32328414789154      0.00003266876993
"""

mol = gto.M(atom=p5,
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
pi_orbital_space = [32,33,34,35,36,37,38,43,47,48]
ncas_orbitals = 10
ncas_electrons = 10

mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.sort_mo(pi_orbital_space)
mc.verbose = verbose
mc.kernel()   # optimize SA orbitals
mc.analyze()