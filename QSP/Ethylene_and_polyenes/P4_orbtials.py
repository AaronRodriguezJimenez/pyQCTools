p4 = """
  C          -0.64094939320315      0.34244078671707     -0.00019690181473
  C           0.64094530960938     -0.34225201617716     -0.00018775842871
  C          -1.82374003730610     -0.28493236580319     -0.00021150421370
  C           1.82374180055777      0.28510952611721     -0.00021813413914
  C          -3.10744532271736      0.40626693597750     -0.00024618759651
  C           3.10743612759383     -0.40610977428093     -0.00023633759102
  C          -4.28678148801571     -0.21351675996947     -0.00029277593834
  C           4.28678634301912      0.21364501007730     -0.00029086084636
  H          -0.61717351924008      1.43097566619531     -0.00020045782985
  H           0.61716315270417     -1.43078598145873     -0.00016760342236
  H          -1.85106008868652     -1.37332689958226     -0.00021167496664
  H           1.85107501907228      1.37350420413576     -0.00023855362093
  H          -3.07202291405005      1.49388384231799     -0.00023783966652
  H           3.07199687505203     -1.49372664798236     -0.00020785034460
  H          -5.21881564106056      0.33973572025573     -0.00032246992444
  H          -4.35426321364532     -1.29749398049876     -0.00030149248323
  H           5.21880363244361     -0.33963395206832     -0.00030760836150
  H           4.35430335787266      1.29761968602730     -0.00032198881141
"""

from pyscf import gto, scf, lo, mcscf
from pyscf.tools import cubegen

mol = gto.M(atom=p4,
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

#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo28.cube', mf.mo_coeff[:,27], nx=80, ny=80, nz=80) # pi3 HOMO
#cubegen.orbital(mol, '/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p3_mo29.cube', mf.mo_coeff[:,28], nx=80, ny=80, nz=80) #pi4* LUMO
#orbital_range = [25,26,27,28,29,30,31,32,33,34,35,36,37,38]
orbital_range = [39,40,41]
for orb in orbital_range:
    cubegen.orbital(mol, f'/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/p4_mo{orb}.cube', mf.mo_coeff[:,orb-1], nx=80, ny=80, nz=80)

#26 pi1 25
#27 pi2 26
#28 pi3 27
#29 pi4 28
#30 pi5* 29
#31 pi6* 30
#37 pi7* 36
#40 pi8* 39


pi_active_pace = [25,26,27,28,29,30,36,39] #only pi and pi*