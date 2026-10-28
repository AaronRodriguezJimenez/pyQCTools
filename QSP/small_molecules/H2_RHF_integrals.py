"""
 Full-orbtial space RHF integrals for H2
"""
import numpy as np
import os
from pyscf import gto, scf
from pyscf.lib import logger
from pyqctools.geometries import H2
from src.pyqctools.int_fcns import get_restricted_ints_full
from src.pyqctools.int_fcns import save_restricted_integrals

# Create output directory, and check if it exists
output_dir = "./H2_integrals"
if not os.path.exists(output_dir):
    print(f"Creating output directory:, {output_dir}")
    os.makedirs(output_dir)
else:
    print(f"Output directory already exists, {output_dir}")

# Create mol object and outputfile name
mol = gto.Mole()
mol.atom = H2()
mol.basis = "cc-pvdz"
#mol.symmetry = True
mol.verbose = 4

output_file = os.path.join(output_dir, f"H2_{mol.basis}.out")
print(f"Output File: {output_file}")
mol.output = output_file

mol.build()

# 2. Run Restricted Hartree-Fock (RHF) calculation
mf = scf.RHF(mol)
e_hf = mf.kernel()
print(f"RHF Energy: {e_hf:.6f} Ha")

#- - Check SCF Stability
    # Loop for optimizing orbitals until stable
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

print("Orbital idx   Energy")
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    print(f"{i+1:3d}  {e:12.6f}  occ={occ}")

from pyscf import mcscf
orbital_space = list(range(10))
mc = mcscf.CASCI(mf, 10, 2) #Builder function just to select active orbitals
C_active = mc.sort_mo(orbital_space, base=0)
mf.mo_coeff = C_active  # Retrieve ordered orbitals to mf object

h0, h1, h2 = get_restricted_ints_full(mol, mf, localized=False)
filename = f"H2-RHF_{mol.basis}"
save_restricted_integrals(filename, h0, h1, h2, folder="H2_integrals")