"""
 Here we compute canonical and localized orbitals for Ethylene and compare
 their plots

"""

from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen
from pyscf import symm

# 1. Define Molecule
naphthalene = """
C 0.000000 0.716253 0.0000
C 0.000000 -0.716253 0.0000
C 1.241539 1.403577 0.0000
C -1.241539 -1.403577 0.0000
C -1.241539 1.403577 0.0000
C 1.241539 -1.403577 0.0000
C 2.432418 0.707325 0.0000
C -2.432418 -0.707325 0.0000
C -2.432418 0.707325 0.0000
C 2.432418 -0.707325 0.0000
H 1.240557 2.492735 0.0000
H -1.240557 -2.492735 0.0000
H -1.240557 2.492735 0.0000
H 1.240557 -2.492735 0.0000
H 3.377213 1.246082 0.0000
H -3.377213 -1.246082 0.0000
H -3.377213 1.246082 0.0000
H 3.377213 -1.246082 0.0000
"""
mol = gto.M(atom=naphthalene,
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

exit()

#print("Localized orbital coefficients:")
#print(C.mo_coeff[:,1])
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo16.cube', mf.mo_coeff[:,15], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo17.cube', mf.mo_coeff[:,16], nx=80, ny=80, nz=80) #pi1
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo18.cube', mf.mo_coeff[:,17], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo19.cube', mf.mo_coeff[:,18], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo20.cube', mf.mo_coeff[:,19], nx=80, ny=80, nz=80) #pi2
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo21.cube', mf.mo_coeff[:,20], nx=80, ny=80, nz=80) #pi3 HOMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo22.cube', mf.mo_coeff[:,21], nx=80, ny=80, nz=80) #pi4 LUMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo23.cube', mf.mo_coeff[:,22], nx=80, ny=80, nz=80) #pi5
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo24.cube', mf.mo_coeff[:,23], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo25.cube', mf.mo_coeff[:,24], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo26.cube', mf.mo_coeff[:,25], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo27.cube', mf.mo_coeff[:,26], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo28.cube', mf.mo_coeff[:,27], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo29.cube', mf.mo_coeff[:,28], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo30.cube', mf.mo_coeff[:,29], nx=80, ny=80, nz=80) #pi6
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes/Benz_mo31.cube', mf.mo_coeff[:,30], nx=80, ny=80, nz=80)
