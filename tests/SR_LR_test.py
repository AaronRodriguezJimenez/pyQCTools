"""

 For larger molecular systems, the DF approximation can become a bottleneck due
 to computational costs and the storage of the intermediate three-index tensor.
 At the mean field level theory, a long-range density fitting (LRDF) scheme was
 developed in PySCF to address these challenges.
 The full-range Coulomb potential can always be written as the summation of the
 long-range (LR) potential and the short-range (SR) potential trhoug an error
 function:

 1/r_12 = erf(wr)/r_12 + erfc(wr)/r_12

 The LRDF scheme employs the density fitting technique to approximate the LR Coulomb
 integrals, while the SR Coulomb integrals are computed exactly.

 The computation of the SR integrals scales linearly with system size.
 Compared to the full-range Coulomb potential, the rank of the LR integral tensor
 is substantially lower.
 One can use a small auxiliary basis set for the LRDF tensor to achieve sufficient accuracy.
 The size of the auxiliary dimension needed is approximately only 1/10 of the size
 of the orbital basis funcitons. This leads to a reduction in computational resources and
 storage.

 In this test, we perform the separation of LR and SR integrals using PySCF.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import pyqctools as pyqc

def cholesky(V, eps):
    # see https://arxiv.org/pdf/1711.02242.pdf section B2
    # see https://arxiv.org/abs/1808.02625
    # see https://arxiv.org/abs/2104.08957
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng


#Compute full integrals
mol = gto.Mole()
mol.atom = [
    ['H', (0., 0., 0.)],
    ['H', (0., 0., 0.74)],
]

mol.basis = 'sto3g'
mol.spin = 0
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 100
mf.conv_tol = 1e-8
mf.kernel()
#MOs
mo_coeff = mf.mo_coeff

print("- - - - - - ")
print("Total ERIs")
eris = mol.intor('int2e')
print(eris)
print("- - - - - - ")
print("Complete Eris Cholesky decomposition")
Lop, ng = cholesky(eris, 1e-6)
print(Lop)
print(ng)
fro_norm = np.sqrt(np.sum(eris**2))
print("Full-Fro norm: ", fro_norm)
print("= = = = = = = = = = = = = = = = = = = ")

omega = 4.0 #1/a_0 units (1/0.4 = 2.5 is the optimal reported for WFT-srDFT)
lr_mol = mol.copy()
with lr_mol.with_long_range_coulomb(omega=omega):
    LR = lr_mol.intor('int2e')
    #print("LR")
    #print(LR)
    print("- - - - - - ")
    print("Eris-LR Cholesky decomposition")
    Lop, ng = cholesky(LR, 1e-6)
    print(Lop)
    print(ng)
    #- - Frobenius Norm
    fro_norm = np.sqrt(np.sum(LR**2))
    print("LR-Fro norm: ", fro_norm)


sr_mol = mol.copy()
with sr_mol.with_short_range_coulomb(omega=omega):
    SR = sr_mol.intor('int2e')
    #print("SR")
    #print(SR)
    print("- - - - - - ")
    print("Eris-SR Cholesky decomposition")
    Lop, ng = cholesky(SR, 1e-6)
    print(Lop)
    print(ng)
    fro_norm = np.sqrt(np.sum(SR**2))
    print("SR-Fro norm: ", fro_norm)
    print("= = = = = = = = = = = = = = = = = = = ")

new_totEris = LR + SR

print("- - - - - - - - - - - - - ")
print("LR + SR")
print(new_totEris)
print(eris - new_totEris)

