from pyscf import gto, scf, dft
import numpy as np
from scipy import linalg
# Define the molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g')

# Run a Hartree-Fock or DFT calculation
mf = scf.RHF(mol).run()

# === get AO 1-RDM from RHF
D_ao = mf.make_rdm1(ao_repr=True)
S = mf.get_ovlp()

C_ao_mo = mf.mo_coeff
mo_occ = mf.mo_occ

# ==== Diagonalize the DM in AO see eq.(1) in Keller et al. [DOI:10.1063/1.4922352] for details.
from functools import reduce
A = reduce(np.dot, (S, D_ao, S))

# ==== Diagonalize to get Natural orbitals
eigvals, eigvecs = linalg.eigh(A, b=S)

# Sort Descending by occupation (largest -> smallest)
order = np.argsort(eigvals)[::-1]
nat_occ = eigvals[order]
NOs = eigvecs[:,order]

# --- optionally, write a cube for the top natural orbital (NO index 0) ---
# convert NO coefficients to AO basis for visualization: C_ao_mo * nat_orb_mo
#nat_orb_ao = C_ao_mo.dot(NOs)
# cubegen.orbital(mol, 'natural_orbital_0.cube', nat_orb_ao[:, 0], nx=80, verbose=4)
# print("Wrote: natural_orbital_0.cube")

# --- 5) print results ---
np.set_printoptions(precision=6, suppress=True)
print("Natural occupations (descending):")
print(nat_occ)
print("\nFirst 6 elements of NO #0 (MO coeffs):")
print(NOs[:, 0][:6])   # show few AO coeffs for first NO
print("\nFirst 6 elements of MO #0 (MO coeffs):")
print(C_ao_mo[:, 0][:6])   # show few AO coeffs for first NO

# Natural orbitals can also be computed using addons
#from pyscf import mcscf
#noons, natorbs = mcscf.addons.make_natural_orbitals(mf)
#print("NOONs from addons: ")
#print(noons)


# Evaluate Density on a Grid (numerical integration)
# Setup a numerical grid
grid = dft.gen_grid.Grids(mol)
grid.level = 3  # Increase for higher precision
grid.build()

# Compute density values at each grid point: rho = D * (basis_at_grid_point**2)
dm = D_ao
phi = dft.numint.eval_ao(mol, grid.coords, deriv=1) #AO value and its grids
rho = dft.numint.eval_rho(mol, phi, dm, xctype="GGA")  #Density and density gradients

# Use Density as input for a Functional
# Choose a functional (e.g., B3LYP or PBE)
exc, vxc = dft.libxc.eval_xc('pbe,pbe', rho, spin=0, relativity=0, deriv=1)[:2]

# Calculate the XC energy contribution
# Energy = integral(rho * epsilon_xc)
n_grid = len(rho)
elemental_energy = rho * exc
total_exc = (elemental_energy * grid.weights).sum()

print(" ")
print(f"Exchange-Correlation Energy: {total_exc:.8f} Hartree")