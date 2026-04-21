# save as ethylene_casscf_vertical_gaps.py
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
import numpy as np

# --- user parameters ---
nstates = 10            # number of lowest singlet states to compute (S0..S(nstates-1))
verbose = 4
hartree_to_ev = 27.211386245988  # conversion factor

p3 ="""
  C          -0.59691673020855      0.30127414661253     -0.00001136360562
  C           0.59691563766006     -0.30129125807882      0.00000053581842
  C          -1.86804608686186     -0.41509590916692     -0.00000963704951
  C           1.86804514506670      0.41508092209292     -0.00001764031881
  C          -3.05829004310842      0.18263535992774     -0.00004608231215
  C           3.05829019206652     -0.18264778731851     -0.00002074609123
  H          -0.64612966472928      1.38897315062128     -0.00003311715897
  H           0.64613066685960     -1.38899103866914      0.00002234449039
  H          -1.81182015020897     -1.50178133143489      0.00001944549488
  H           1.81181784661532      1.50176608248966     -0.00002958338871
  H          -3.98002471737652     -0.38766336395150     -0.00004961142061
  H          -3.14576530696508      1.26519154024145     -0.00007824430025
  H           3.98002384171624      0.38765260286158     -0.00003743160775
  H           3.14576936947424     -1.26520411622738     -0.00000786855006
"""

mol = gto.M(atom=p3,
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
pi_orbital_space = [20,21,22,23,25,30]
ncas_orbitals = 6
ncas_electrons = 6

mc = mcscf.CASCI(mf, ncas_orbitals, ncas_electrons)
#mc = mcscf.CASSCF(mf, ncas_orbitals, ncas_electrons)
mc.sort_mo(pi_orbital_space)
mc.verbose = verbose
mc.fcisolver.nroots = 20
mc.kernel()   # optimize SA orbitals
mc.analyze()