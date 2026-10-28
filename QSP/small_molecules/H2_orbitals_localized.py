from pyscf import gto, scf, lo
from pyscf.tools import cubegen
from pyqctools.geometries import H2
import numpy as np

bas = "cc-pvdz"
mol = gto.M(atom=H2(),
            basis=bas,
)

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

#- - Check SCF Stability
    # Loop for optimizing orbitals until stable
    #
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
print(f"Total HF energy: {ehf_val}")

# <S^2>
ss = mf.spin_square()
print(f"Spin square <S^2>: {ss[0]}")
print("RESTRICTED HF ORBITALS:")
print(f"{bas} - Orbital idx   Energy")
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    print(f"{i+1:3d}  {e:12.6f}  occ={occ}")

# --------------------------------------------------
# Canonical RHF orbitals
# --------------------------------------------------

C_rhf = mf.mo_coeff
S = mol.intor("int1e_ovlp")

# --------------------------------------------------
# Boys localization of selected orbital space
# --------------------------------------------------

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
C_sub = C_rhf[:, idx]

boys = lo.Boys(mol, C_sub)
C_boys = boys.kernel()

# --------------------------------------------------
# Check orthonormality
# --------------------------------------------------

print("\nBoys orbital overlap:")
print(C_boys.T @ S @ C_boys)

# Transformation between RHF and Boys orbitals
U = C_sub.T @ S @ C_boys

print("\nRHF -> Boys rotation:")
print(U)

print("\nCheck U^T U:")
print(U.T @ U)

# --------------------------------------------------
# Compare Fock matrices
# --------------------------------------------------

F = mf.get_fock()

print("\nFock matrix in canonical subspace:")
print(C_sub.T @ F @ C_sub)

print("\nFock matrix in Boys basis:")
print(C_boys.T @ F @ C_boys)

# --------------------------------------------------
# Orbital centers
# --------------------------------------------------

r_ao = mol.intor("int1e_r", comp=3)

def centers(C):
    return np.array([
        np.einsum("xuv,u,v->x", r_ao, C[:,i], C[:,i])
        for i in range(C.shape[1])
    ])

print("\nCanonical orbital centers:")
print(centers(C_sub))

print("\nBoys orbital centers:")
print(centers(C_boys))

# --------------------------------------------------
# Cube files for localized orbitals
# --------------------------------------------------

for i in range(C_sub.shape[1]):
    cubegen.orbital(
        mol,
        f"H2_Boys_{i+1}_{bas}.cube",
        C_boys[:, i],
        nx=80, ny=80, nz=80
    )