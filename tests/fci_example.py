# Example: FCI calculation
from pyscf import gto, scf, cc, fci
import numpy as np
import pyqctools as pq
from pyqctools import ethylene, H2

# 1. Define Molecule
Ethyl = pq.ethylene()
water = pq.H2O()
n2 = pq.N2()
h2 = H2()

mol = gto.M(atom=[
        ['N', (0.0, 0.0, -1.5)],
        ['N', (0, 0, 1.5)]],
            basis='sto-3g',
            spin=0,
            charge=0,
            cart=True,
            verbose=4)

# Redo with CASSCF and analyze result (for H2)

# 2. Run Restricted Hartree-Fock (RHF)
#- - - Normal SCF - - -
mf = scf.UHF(mol)
mf.max_cycle = 100
mf.conv_tol = 1e-8
#hf_energy = mf.kernel()
#print('RHF energy from PySCF is: ', hf_energy)

# - - - FCI - - -
from pyscf.lib import logger

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

# create an FCI solver based on the SCF object
cisolver = fci.FCI(mf)
cisolver.spin = 0
cisolver.nroots = 4
e_fci = cisolver.kernel()[0]

from src.pyqctools.int_fcns import get_unrestricted_ints_full, save_unrestricted_integrals
h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab = get_unrestricted_ints_full(mol, mf,  localized=True, debug=False)

save_unrestricted_integrals("N2_3.0_sto3g_UHF", h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab, folder="Test_N2")

print(f"Total UHF energy: {ehf_val}")
# Lowest 4 FCI roots
print(f"FCI roots: {e_fci}")