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
#
# Calculation using UCCSD
print("# # # # # # UCCSD # # # # #")
from pyscf import cc

#ucc = cc.UCCSD(mf)
#eris_ucc = ucc.ao2mo()
#ecc, t1, t2 = ucc.kernel(eris=eris_ucc)
#print("E_cc : ", ecc)
#print("E_t1 : ", t1)
#print("E_t2 : ", t2)

# 3. Run CCSD(T) for High-Accuracy Reference
#    (Freezing core orbitals is standard, but you can set frozen=0 for full)
mycc = cc.UCCSD(mf)
#mycc.frozen = n_core_orb  # Set to 0 to correlate all electrons (FCI limit approximation)
mycc.kernel()
e_ccsd_t = mycc.ccsd_t()
print(f"UCCSD Energy:    {mycc.e_tot:.8f} Ha")
print(f"UCCSD(T) Energy: {mycc.e_tot + e_ccsd_t:.8f} Ha")
import numpy as np

# 1. Solve Lambda and get the FULL 1-RDM
#    PySCF returns this in the full MO basis (n_mo x n_mo),
#    with the frozen core block already set to occupancy=1.
mycc.solve_lambda()
rdm1 = mycc.make_rdm1()

# Containers for results
no_coeffs = []
no_occs = []

# 2. Process each spin channel
#    Since rdm1 is (28, 28), we use the full mf.mo_coeff (28, 28)
for spin_idx, (dm, mo_c_full) in enumerate(zip(rdm1, mf.mo_coeff)):
    spin_label = "Alpha" if spin_idx == 0 else "Beta"

    # Diagonalize the FULL 1-RDM
    # This naturally recovers:
    #   - Frozen Core orbitals (Occupancy = 1.0)
    #   - Active Natural Orbitals (Occupancy between 0 and 1)
    occ_vals, vecs = np.linalg.eigh(dm)

    # Sort High -> Low
    idx = occ_vals.argsort()[::-1]
    occ_vals = occ_vals[idx]
    vecs = vecs[:, idx]

    # Transform to AO basis
    # (28, 28) = (28, 28) . (28, 28)
    no_c = np.dot(mo_c_full, vecs)

    no_coeffs.append(no_c)
    no_occs.append(occ_vals)

    print(f"{spin_label} Natural Occupancies (Top 10):")
    print(f"  {occ_vals}")

# Final objects
no_coeff_uhf = np.array(no_coeffs)
no_occ_uhf = np.array(no_occs)

print("Natural Orbital construction complete.")
print("Shape :", no_coeff_uhf.shape)
print(no_coeff_uhf[0] - no_occ_uhf[1])