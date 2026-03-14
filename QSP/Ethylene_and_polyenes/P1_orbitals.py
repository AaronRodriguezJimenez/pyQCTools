from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen

ethylene ="""
  C          -0.66239743908476     -0.00000011879825      0.00000025727475
  H          -1.23195628735572      0.92408580955377      0.00000003541117
  H          -1.23195584363698     -0.92408658869570     -0.00000029505709
  C           0.66239765285931     -0.00000003323471      0.00000026754285
  H           1.23195601732587      0.92408614541835     -0.00000029781984
  H           1.23195589989228     -0.92408621424346      0.00000003264815
"""

mol = gto.M(atom=ethylene,
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

cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo6.cube', mf.mo_coeff[:,5], nx=80, ny=80, nz=80) #sigma1
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo7.cube', mf.mo_coeff[:,6], nx=80, ny=80, nz=80) #sigma1*
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo8.cube', mf.mo_coeff[:,7], nx=80, ny=80, nz=80) #pi1 HOMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo9.cube', mf.mo_coeff[:,8], nx=80, ny=80, nz=80) #pi2 LUMO
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo10.cube', mf.mo_coeff[:,9], nx=80, ny=80, nz=80) # sigma 2
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo11.cube', mf.mo_coeff[:,10], nx=80, ny=80, nz=80) # sigma 2
cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/c2h4_mo12.cube', mf.mo_coeff[:,11], nx=80, ny=80, nz=80) # sigma 2

pi_active_pace = [8,9] #only pi and pi*