# Example: FCI calculation
from pyscf import gto, scf, cc, fci, mcscf
import numpy as np
import pyqctools as pq
from pyqctools import ethylene, H2

# 1. Define Molecule
Ethyl = pq.ethylene()
water = pq.H2O()
n2 = pq.N2()
h2 = H2()

mol = gto.M(atom=h2, #n2,#Ethyl,
            basis='sto-3g',
            spin=0,
            cart=True,
            verbose=4,
            output= 'h2_singlets.out')


# 2. Run Restricted Hartree-Fock (RHF)
#- - - Normal SCF - - -
mf = scf.RHF(mol)
hf_energy = mf.kernel()
print('RHF energy from PySCF is: ', hf_energy)

mc = mcscf.CASCI(mf, 2, 2)
mc.fcisolver.conv_tol = 1e-7
mc.fcisolver.threads = 1
mc.fcisolver.nroots = 10  # Let's compute various roots

e_tot, _, ci, mo, _ = mc.kernel()
mc.analyze()  #with verbose=4 prints details about the calculation

exit()
# - - - FCI - - -
myhf = mol.RHF().run()
# create an FCI solver based on the SCF object
cisolver = fci.FCI(myhf)
cisolver.spin = 0
cisolver.nroots = 4
e_fci = cisolver.kernel()[0]
# Lowest 4 FCI roots
print(e_fci)

