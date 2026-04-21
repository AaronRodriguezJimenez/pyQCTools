# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 10            # number of lowest singlet states to compute (S0..S(nstates-1))
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

p6 ="""
  C           0.64343308076370      0.33052831870835     -0.00611671240108
  C          -0.64526833862642     -0.33501643307384     -0.00249576108294
  C           1.81929347868988     -0.31446856464087     -0.00269387914829
  C          -1.82113529325407      0.30996913158699     -0.00591188621427
  C           3.10820555220410      0.35292102153994     -0.00634221955089
  C          -3.11003049057436     -0.35745308428207     -0.00227669762442
  C           4.28423377858042     -0.28844767791269     -0.00297176751693
  C          -4.28608569180871      0.28386581498976     -0.00565622853339
  C           5.57521804773979      0.38799822077652     -0.00666573500171
  C          -5.57703024033174     -0.39265784865553     -0.00196733412831
  C           6.74799473285936     -0.24449656375129     -0.00334321862602
  C          -6.74985022692894      0.23975921716944     -0.00529598745373
  H           0.63574625876063      1.41925912804067     -0.01196566396111
  H          -0.63757600758710     -1.42374721628794      0.00335137962884
  H           1.82776056027689     -1.40312688111194      0.00315871880127
  H          -1.82962116991243      1.39862715301456     -0.01175174824473
  H           3.09826252616200      1.44172812576266     -0.01217949822426
  H          -3.10004957352039     -1.44626009833457      0.00355795852826
  H           4.29880888598569     -1.37705351181525      0.00286463094804
  H          -4.30071769785359      1.37247064635448     -0.01148644768907
  H           5.55191317722983      1.47595375399565     -0.01249347201344
  H          -5.55365750033573     -1.48061195640129      0.00386058930108
  H           7.68585751811085      0.29878080346402     -0.00630483325702
  H           6.80385935688023     -1.32911157715159      0.00246166169651
  H          -7.68767647315088     -0.30358086387369     -0.00233603804087
  H          -6.80578725035899      1.32437094188952     -0.01109881019151
"""

mol = gto.M(atom=p6,
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
pi_orbital_space = [38,39,40,41,42,43,44,45,47,54,57,58]
ncas_orbitals = 12
ncas_electrons = 12

mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.sort_mo(pi_orbital_space)
mc.verbose = verbose
mc.kernel()   # optimize SA orbitals
mc.analyze()