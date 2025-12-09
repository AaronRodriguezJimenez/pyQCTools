# Example: CCSD calculation
from pyscf import gto, scf, cc
import numpy as np
import pyqctools as pq
from pyqctools import ethylene

# 1. Define Molecule
Ethyl = pq.ethylene()
#print(Ethyl)
mol = gto.M(atom=Ethyl, basis='sto-3g')
#mol = gto.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, 0.74))], basis='sto-3g')

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

# 3. Initialize and run CCSD object with the HF solution
mycc = cc.CCSD(mf).run()

# 4. (Optional) Set parameters, e.g., freeze 2 core orbitals
# mycc.set(frozen=2) # Use if you have more complex molecules

# 5. Run the CCSD calculation (kernel() method)
# The total energy is mf.e_tot + mycc.e_corr
e_ccsd = mycc.e_tot
et = mycc.ccsd_t()

print(f"HF Energy: {mf.e_tot}")
print(f"CCSD Correlation Energy: {mycc.e_corr}")
print(f"Total CCSD Energy: {e_ccsd}")
print('CCSD(T) total energy', mycc.e_tot + et)