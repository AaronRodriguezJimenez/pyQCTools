# Example: FCI calculation
from pyscf import gto, scf, cc, fci
import numpy as np
import pyqctools as pq
from pyqctools import ethylene

# 1. Define Molecule
Ethyl = pq.ethylene()
water = pq.H2O()
n2 = pq.N2()

mol = gto.M(atom=n2,#Ethyl,
            basis='sto-3g',
            spin=0,
            cart=True,
            verbose=0)

# 2. Run Restricted Hartree-Fock (RHF)
#- - - Normal SCF - - -
mf = scf.RHF(mol)
hf_energy = mf.kernel()
print('RHF energy from PySCF is: ', hf_energy)

# - - - FCI - - -
myhf = mol.RHF().run()
# create an FCI solver based on the SCF object
cisolver = fci.FCI(myhf)
e_fci = cisolver.kernel()[0]
print('E(FCI) = %.12f' % e_fci)
