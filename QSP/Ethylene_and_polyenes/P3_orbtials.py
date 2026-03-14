p3 = """
  C          -0.59691673020855      0.30127414661253     -0.00001136360562
  C           0.59691563766006     -0.30129125807882      0.00000053581842
  C          -1.86804608686186     -0.41509590916692     -0.00000963704951
  C           1.86804514506670      0.41508092209292     -0.00001764031881
  C          -3.05829004310842      0.18263535992774     -0.00004608231215
  C           3.05829019206652     -0.18264778731851     -0.00002074609123
  H          -0.64612966472928      1.38897315062128     -0.00003311715897
  H           0.64613066685960     -1.38899103866914      0.00002234449039
  H          -1.81182015020897     -1.50178133143489      0.00001944549488
  H           1.81181784661532      1.50176608248966     -0.00002958338871
  H          -3.98002471737652     -0.38766336395150     -0.00004961142061
  H          -3.14576530696508      1.26519154024145     -0.00007824430025
  H           3.98002384171624      0.38765260286158     -0.00003743160775
  H           3.14576936947424     -1.26520411622738     -0.00000786855006
"""

from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen

mol = gto.M(atom=p3,
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


#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo18.cube', mf.mo_coeff[:,17], nx=80, ny=80, nz=80) #
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo19.cube', mf.mo_coeff[:,18], nx=80, ny=80, nz=80) #
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo20.cube', mf.mo_coeff[:,19], nx=80, ny=80, nz=80) # pi1
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo21.cube', mf.mo_coeff[:,20], nx=80, ny=80, nz=80) # Pi2
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo22.cube', mf.mo_coeff[:,21], nx=80, ny=80, nz=80) # pi3 HOMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo23.cube', mf.mo_coeff[:,22], nx=80, ny=80, nz=80) #pi4* LUMO
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo24.cube', mf.mo_coeff[:,23], nx=80, ny=80, nz=80) #
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo25.cube', mf.mo_coeff[:,24], nx=80, ny=80, nz=80) #pi5*
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo26.cube', mf.mo_coeff[:,25], nx=80, ny=80, nz=80)
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo27.cube', mf.mo_coeff[:,26], nx=80, ny=80, nz=80)
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo28.cube', mf.mo_coeff[:,27], nx=80, ny=80, nz=80)
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo29.cube', mf.mo_coeff[:,28], nx=80, ny=80, nz=80)
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo30.cube', mf.mo_coeff[:,29], nx=80, ny=80, nz=80) #pi6*


pi_active_pace = [20,21,22,23,25,30] #only pi and pi*