"""
 Here we compute canonical and localized orbitals for Benzene and compare
 their plots

"""

from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen
from pyscf import symm
import pyqctools as pq


# 1. Define Molecule
benzene = pq.benzene()
mol = gto.M(atom=benzene,
            basis='cc-pvdz',
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

print("Orbital idx   Energy")
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    print(f"{i+1:3d}  {e:12.6f}  occ={occ}")

# The active space of the benzene (6e,6o) contains the following
# canonical orbitals
# pi_orbital_space = [17,20,21,22,23,30]
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo17.cube', mf.mo_coeff[:,16], nx=80, ny=80, nz=80) #pi1
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo20.cube', mf.mo_coeff[:,19], nx=80, ny=80, nz=80) #pi2
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo21.cube', mf.mo_coeff[:,20], nx=80, ny=80, nz=80) #pi3 HOMO
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo22.cube', mf.mo_coeff[:,21], nx=80, ny=80, nz=80) #pi4 LUMO
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo23.cube', mf.mo_coeff[:,22], nx=80, ny=80, nz=80) #pi5
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo30.cube', mf.mo_coeff[:,29], nx=80, ny=80, nz=80) #pi6

# However, for CASSCF and integral extraction routines,
# since they are not ordered, we need to reorder them
pi_orbital_space = [17,20,21,22,23,30]
mc = mcscf.CASSCF(mf, 6, 6)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object
mc.mo_coeff = mc.sort_mo(pi_orbital_space)

# --- AFTER sorting: active orbitals are now in a block ---
# In CASSCF ordering: [core | active | virtual]
ncore = mc.ncore
ncas = mc.ncas
print(mc.ncore)
print(mc.ncas)

#print("Generating cubes AFTER sorting...")
#for i in range(ncas):
#    idx = ncore + i
#    cubegen.orbital(mol, f"/Users/admin/after_active_{i}.cube", C_active[:, idx])

