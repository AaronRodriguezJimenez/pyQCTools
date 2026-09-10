from pyscf import gto, scf
from pyscf.tools import cubegen

furan = [['C', 0.00000000,  2.06365826 , -0.60051250],
         ['C', 0.00000000,  -2.06365826, -0.60051250],
         ['C', 0.00000000,  1.35348578 , 1.86336416 ],
         ['C', 0.00000000,  -1.35348578, 1.86336416 ],
         ['O', 0.00000000,  0.00000000 , -2.13945332],
         ['H', 0.00000000,  3.86337287 , -1.53765695],
         ['H', 0.00000000,  -3.86337287, -1.53765695],
         ['H', 0.00000000,  2.59168789 , 3.47168051 ],
         ['H', 0.00000000,  -2.59168789, 3.47168051 ]]

mol = gto.Mole()
mol.atom = furan
mol.charge = 0
mol.spin = 0
mol.basis = "sto3g"
mol.unit = "Bohr"
mol.build()

# 2. Run Restricted Hartree-Fock (RHF) calculation
mf = scf.RHF(mol)
e_hf = mf.kernel()
print(f"RHF Energy for Furan: {e_hf:.6f} Ha")

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
title = "Furan-STO3g"
save_restricted_integrals(title, h0, h1, h2, folder="furan_tensors_sto3g")
