import numpy as np
from pyscf import gto, scf
from scipy.linalg import eigh   # generalized Hermitian eigensolver

# 1) Build system and run UHF
mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='cc-pVDZ', unit='Angstrom')
mf = scf.UHF(mol)
mf.verbose = 4
mf.kernel()

# 2) Get spin density matrices (AO basis)
# mf.make_rdm1() for UHF returns densities for alpha and beta channels (shape (2, nao, nao)).
dm = mf.make_rdm1()
# robust unpacking whether it's (2,nao,nao) or tuple/list
if isinstance(dm, (list, tuple)) or (isinstance(dm, np.ndarray) and dm.ndim == 3):
    dma, dmb = dm[0], dm[1]
else:
    # fallback: treat as closed-shell-like => split equally
    dma = dmb = 0.5 * dm

# total (spin-summed) 1RDM
D_tot = dma + dmb

# overlap matrix (AO overlap)
S = mf.get_ovlp()

# 3) Solve generalized eigenproblem D C = S C n
# eigh returns eigenvalues in ascending order by default
eigvals, eigvecs = eigh(D_tot, S)
# sort descending (largest occupations first)
idx = eigvals.argsort()[::-1]
nat_occupations = eigvals[idx]
nat_orbitals_ao = eigvecs[:, idx]   # columns are AO coefficients of natural orbitals

# print a few occupations
print("Natural occupations (highest):")
for i, occ in enumerate(nat_occupations[:10]):
    print(f"  {i:3d} : {occ:.6f}")

# 4) Optionally transform natural orbitals to the MO basis (example: alpha MOs)
mo_a = mf.mo_coeff[0]   # AO->MO (alpha) coefficients: shape (nao, nmo)
# overlap-weighted projection of nat orbitals into MO basis:
# C_nat_in_MO = (S @ nat_orbitals)^\dagger @ mo  -> but simpler: expressed in MO coef basis:
C_nat_in_MO_alpha = mo_a.T.conj() @ (S @ nat_orbitals_ao)

# Now nat_orbitals_ao[:,i] are AO coefficients; nat_occupations[i] are occupations.