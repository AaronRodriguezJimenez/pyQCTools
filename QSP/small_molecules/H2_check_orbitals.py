from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen
from pyqctools.geometries import H2
from pyscf.lib import logger
from pyscf import mcscf

"""
 This script computes H2-RHF case and compares orbitals used for 
 molecular integral drivers.  
"""
bas = "cc-pvdz"
mol = gto.M(atom=H2(),
            basis=bas,
)

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

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

print(f"{bas} - Orbital idx   Energy")
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    print(f"{i+1:3d}  {e:12.6f}  occ={occ}")

filename = f"H2_mo1_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,0], nx=80, ny=80, nz=80)
filename = f"H2_mo2_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,1], nx=80, ny=80, nz=80)
#Other virtual orbitals
filename = f"H2_mo3_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,2], nx=80, ny=80, nz=80)
filename = f"H2_mo4_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,3], nx=80, ny=80, nz=80)
filename = f"H2_mo5_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,4], nx=80, ny=80, nz=80)
filename = f"H2_mo6_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,5], nx=80, ny=80, nz=80)
filename = f"H2_mo7_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,6], nx=80, ny=80, nz=80)
filename = f"H2_mo8_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,7], nx=80, ny=80, nz=80)
filename = f"H2_mo9_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,8], nx=80, ny=80, nz=80)
filename = f"H2_mo10_{bas}.cube"
cubegen.orbital(mol, filename, mf.mo_coeff[:,9], nx=80, ny=80, nz=80)

# - - - Test active space ordering

orbital_space = list(range(3))
mc = mcscf.CASCI(mf, 3, 2) #Builder function just to select active orbitals
C_active = mc.sort_mo(orbital_space, base=0)
mf.mo_coeff = C_active  # Retrieve ordered orbitals to mf object
mc.mo_coeff = mc.sort_mo(orbital_space, base=0)

filename = f"H2_AS_mo1_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,0], nx=80, ny=80, nz=80) #sigma1
filename = f"H2_AS_mo2_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,1], nx=80, ny=80, nz=80) #sigma1*
#Other virtual orbitals
filename = f"H2_AS_mo3_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,2], nx=80, ny=80, nz=80)
exit()
filename = f"H2_AS_mo4_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,3], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo5_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,4], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo6_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,5], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo7_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,6], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo8_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,7], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo9_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,8], nx=80, ny=80, nz=80)
filename = f"H2_AS_mo10_{bas}.cube"
cubegen.orbital(mol, filename, mc.mo_coeff[:,9], nx=80, ny=80, nz=80)