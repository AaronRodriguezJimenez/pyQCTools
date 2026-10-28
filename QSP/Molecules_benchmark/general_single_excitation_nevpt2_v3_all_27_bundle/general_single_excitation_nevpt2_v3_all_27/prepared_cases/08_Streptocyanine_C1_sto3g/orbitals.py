from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen
from pyqctools.geometries import H2


#- - - Streptocyanine-C1 - - -
coords = [['C',  0.00000000,  0.00000000 , 0.80488833 ],
          ['N',  0.00000000,  2.19423463 , -0.33580561],
          ['N',  0.00000000,  -2.19423463, -0.33580561],
          ['H',  0.00000000,  0.00000000 , 2.84436959 ],
          ['H',  0.00000000,  2.36978315 , -2.23371976],
          ['H',  0.00000000,  -2.36978315, -2.23371976],
          ['H',  0.00000000,  3.79412648 , 0.69399206 ],
          ['H',  0.00000000,  -3.79412648, 0.69399206 ]]

mol = gto.Mole()
mol.atom = coords
mol.charge = 1
mol.spin = 0
mol.basis = "sto3g"
mol.unit = "Bohr"
mol.build()

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

cubegen.orbital(mol, '/home/aaron/SC_mo10.cube', mf.mo_coeff[:,9], nx=80, ny=80, nz=80) #orb 10
cubegen.orbital(mol, '/home/aaron/SC_mo11.cube', mf.mo_coeff[:,10], nx=80, ny=80, nz=80) #orb 11
cubegen.orbital(mol, '/home/aaron/SC_mo12.cube', mf.mo_coeff[:,11], nx=80, ny=80, nz=80) #orb 12 HOMO
cubegen.orbital(mol, '/home/aaron/SC_mo13.cube', mf.mo_coeff[:,12], nx=80, ny=80, nz=80) #orb 13 LUMO
cubegen.orbital(mol, '/home/aaron/SC_mo14.cube', mf.mo_coeff[:,13], nx=80, ny=80, nz=80) #orb 14
cubegen.orbital(mol, '/home/aaron/SC_mo15.cube', mf.mo_coeff[:,14], nx=80, ny=80, nz=80) #orb 15
