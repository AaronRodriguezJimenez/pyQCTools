p2 = """
  C          -0.61315490669049      0.40023050451111      0.00002076641733
  C           0.61315475271315     -0.40006851258307      0.00002367277570
  C          -1.83714210045915     -0.12164733566620      0.00001829709420
  C           1.83714254733887      0.12180653296845      0.00001919613192
  H          -0.48624341694090      1.48095607361609      0.00002411914816
  H           0.48624148957999     -1.48079361526528      0.00002540979972
  H          -2.72205260425108      0.50449176034317      0.00001983727721
  H          -1.99141667810657     -1.19678576348874      0.00001399071790
  H           2.72205150022975     -0.50433410163189      0.00001731310711
  H           1.99141941658643      1.19694445719635      0.00001539753074
"""

from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen

mol = gto.M(atom=p2,
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


cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p2_mo14.cube', mf.mo_coeff[:,13], nx=80, ny=80, nz=80) #pi1
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p2_mo15.cube', mf.mo_coeff[:,14], nx=80, ny=80, nz=80) #pi2 HOMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p2_mo16.cube', mf.mo_coeff[:,15], nx=80, ny=80, nz=80) #pi3*LUMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p2_mo20.cube', mf.mo_coeff[:,19], nx=80, ny=80, nz=80) #pi4*

pi_active_pace = [14,15,16,20] #only pi and pi*