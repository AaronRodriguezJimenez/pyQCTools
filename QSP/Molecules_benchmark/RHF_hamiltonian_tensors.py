from pyscf import gto, scf
from pyscf.tools import cubegen

# - - - Diazomethane - - -
#coords = [['C', 0.00000000,  0.00000000 , -2.30830005],
#          ['N', 0.00000000,  0.00000000 , 0.14457890 ],
#          ['N', 0.00000000,  0.00000000 , 2.29923216 ],
#          ['H', 0.00000000,  1.79875201 , -3.24272317],
#          ['H', 0.00000000,  -1.79875201, -3.24272317]]

#- - - Ketene - - -
#coords = [['C',  0.00000000,  0.00000000 , -2.44810151],
#          ['C',  0.00000000,  0.00000000 , 0.03498545 ],
#          ['O',  0.00000000,  0.00000000 , 2.23663914 ],
#          ['H',  0.00000000,  1.77432079 , -3.43705988],
#          ['H',  0.00000000,  -1.77432079, -3.43705988]]

#- - - Streptocyanine-C1 - - -
#coords = [['C',  0.00000000,  0.00000000 , 0.80488833 ],
#          ['N',  0.00000000,  2.19423463 , -0.33580561],
#          ['N',  0.00000000,  -2.19423463, -0.33580561],
#          ['H',  0.00000000,  0.00000000 , 2.84436959 ],
#          ['H',  0.00000000,  2.36978315 , -2.23371976],
#          ['H',  0.00000000,  -2.36978315, -2.23371976],
#          ['H',  0.00000000,  3.79412648 , 0.69399206 ],
#          ['H',  0.00000000,  -3.79412648, 0.69399206 ]]

#- - - Cyanoacetylene
#coords = [['C',  0.00000000,  0.00000000,  -3.59120182],
#          ['C',  0.00000000,  0.00000000,  -1.30693904],
#          ['C',  0.00000000,  0.00000000,  1.28880240 ],
#          ['N',  0.00000000,  0.00000000,  3.48692211 ],
#          ['H',  0.00000000,  0.00000000,  -5.59619886]]

#- - - Cyanogen
#coords = [['C',  0.00000000,  0.00000000,  1.30401924 ],
#          ['C',  0.00000000,  0.00000000,  -1.30401924],
#          ['N',  0.00000000,  0.00000000,  3.49784121 ],
#          ['N',  0.00000000,  0.00000000,  -3.49784121]]

#- - - Diacetylene
#coords = [['C',  0.00000000,  0.00000000,  1.29447700 ],
#          ['C',  0.00000000,  0.00000000,  -1.29447700],
#          ['C',  0.00000000,  0.00000000,  3.58448429 ],
#          ['C',  0.00000000,  0.00000000,  -3.58448429],
#          ['H',  0.00000000,  0.00000000,  5.58943003 ],
#          ['H',  0.00000000,  0.00000000,  -5.58943003]]

#- - - Thiophene
coords = [['C',  0.00000000,  2.33342542 , -0.09858421],
          ['C',  0.00000000,  -2.33342542, -0.09858421],
          ['C',  0.00000000,  1.34371718 , -2.48297725],
          ['C',  0.00000000,  -1.34371718, -2.48297725],
          ['S',  0.00000000,  0.00000000 , 2.17250692 ],
          ['H',  0.00000000,  4.29028016 , 0.44577296 ],
          ['H',  0.00000000,  -4.29028016, 0.44577296 ],
          ['H',  0.00000000,  2.48760051 , -4.16768392],
          ['H',  0.00000000,  -2.48760051, -4.16768392]]

mol = gto.Mole()
mol.atom = coords
mol.charge = 0
mol.spin = 0
mol.basis = "sto3g"
mol.unit = "Bohr"
mol.build()

# 2. Run Restricted Hartree-Fock (RHF) calculation
mf = scf.RHF(mol)
e_hf = mf.kernel()
print(f"RHF Energy: {e_hf:.6f} Ha")

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

from src.pyqctools.int_fcns import get_restricted_ints_full
from src.pyqctools.int_fcns import save_restricted_integrals

h0, h1, h2 = get_restricted_ints_full(mol, mf, localized=False)
title = "Thiophene-STO3g"
save_restricted_integrals(title, h0, h1, h2, folder="Thiophene_tensors_sto3g")
