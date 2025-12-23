from pyscf import gto, scf, cc, fci
import numpy as np
import pyqctools as pq
import openfermion as of
from openfermion.ops import InteractionOperator
from openfermion.linalg import get_sparse_operator

# 1. Define Molecule
Ethyl = pq.ethylene()
water = pq.H2O()
#print(Ethyl)
#mol = gto.M(atom=Ethyl, basis='sto-3g')
mol = gto.M(atom=water,
            basis='sto-3g',
            spin=0,
            cart=True,
            verbose=0)

#- - - Normal SCF - - -
mf = scf.RHF(mol)
hf_energy = mf.kernel()
print('RHF energy from PySCF is: ', hf_energy)

H0, H1, H2 = pq.ham_fcns.get_tensors(mol, mf, localized=True)

print("H1 shape:", H1.shape)        # should be (n_spin_orb, n_spin_orb)
print("H2 shape:", H2.shape)        # should be (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)

# Hermiticity of one-body:
print("max|H1 - H1.T*|:", np.max(np.abs(H1 - H1.T.conj())))

# two-body 4-point symmetry quick checks (chemist symmetry subset)
print("max|H2[p,q,r,s] - H2[r,s,p,q]|:", np.max(np.abs(H2 - H2.transpose(2,3,0,1))))
