from pyscf import gto, scf, mcscf
from pyscf.mcscf import PiOS
from pyscf.tools import cubegen
import numpy as np

# 1. Define the molecule (planar system)
mol = gto.M(atom='C 0 0 0; C 0 0 1.34; H 0 0.93 -0.54; H 0 -0.93 -0.54; H 0 0.93 1.88; H 0 -0.93 1.88',
            basis='sto-3g')

#collect info about atoms' positions
atom_info = PiOS.Atoms_w_Coords(mol)
print(atom_info)

# 2. Run a standard Hartree-Fock calculation
mf = scf.RHF(mol).run()

# PiOS uses 1-based atom indices
pi_atoms = [1, 2]

# PiOS returns: n_core, n_act, n_vir, n_elec, reordered_mo_coeff
n_core, n_act, n_vir, n_elec, mo_pios = PiOS.MakePiOS(mol, mf, pi_atoms)

print("n_core =", n_core)
print("n_act  =", n_act)
print("n_vir  =", n_vir)
print("n_elec =", n_elec)

# Active-space block: these are the orbitals you care about
mo_active = mo_pios[:, n_core:n_core + n_act]
print("active MO block shape:", mo_active.shape)

# For ethylene this should usually be 2 orbitals:
#   active orbital 0 -> pi
#   active orbital 1 -> pi*
# The lower-energy one is normally the bonding pi orbital.
for i in range(n_act):
    mo_idx = n_core + i
    print(f"\n=== Active orbital {i} (global MO index {mo_idx}) ===")
    print("MO energy:", mf.mo_energy[mo_idx])
    print("Largest AO coefficients:")
    ao_names = mol.ao_labels()
    coeff = mo_pios[:, mo_idx]
    for j in np.argsort(np.abs(coeff))[::-1][:12]:
        print(f"{ao_names[j]:20s}  {coeff[j]: .8f}")

# Write a Molden file with the reordered orbitals
#with open("ethylene_pios.molden", "w") as f:
#    molden.header(mol, f)
#    molden.orbital_coeff(mol, f, mo_pios, ene=mf.mo_energy, occ=mf.mo_occ)

# Write cube files for each active orbital
for i in range(n_act):
    mo_idx = n_core + i
    outfile = f"ethylene_active_{i+1}.cube"
    cubegen.orbital(mol, outfile, mo_pios[:, mo_idx], nx=80, ny=80, nz=80)
    print("wrote", outfile)

# Run CASSCF with the PiOS active space
mc = mcscf.CASSCF(mf, n_act, n_elec)
AS = range(n_core, n_core + n_act)
mo_sorted = mcscf.sort_mo(mc, mo_pios, AS, base=0)
mc.kernel(mo_sorted)