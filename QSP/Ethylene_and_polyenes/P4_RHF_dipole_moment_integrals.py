"""
 Dipole moment integrals for C2H4
"""
from pyscf import gto, scf, lo, mcscf
import pyqctools as pq
from pyscf.lib import logger
import numpy as np
import os
from src.pyqctools.int_fcns import get_restricted_active_space_integrals
from src.pyqctools.int_fcns import save_restricted_integrals

p4 ="""
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

mol = gto.M(atom=p4,
            basis='cc-pvdz',
)

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

#- - Check SCF Stability
    # Loop for optimizing orbitals until stable
    #
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

# active MO coefficients used by FCI
# pi orbitals
pi_orbital_space = [25,26,27,28,29,30,36,39]
mc = mcscf.CASSCF(mf, 8, 8)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object
mc.mo_coeff = mc.sort_mo(pi_orbital_space)

print(C_active.shape)
print("target_ncas =", int(mc.ncas))
print("ncore =", int(mc.ncore))

h0, h1, h2 = get_restricted_active_space_integrals(mol, mf, mc, localized=False, include_all_noncore=False)

dir = "/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes"
save_restricted_integrals("P4-RHF", h0, h1, h2, dir)

#= = = Dipole moment = = =
def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()

with mol.with_common_orig(_charge_center(mol)):
    dip_ao =  mol.intor_symmetric('int1e_r', comp=3)

#dip_mo = _contract_multipole(dip_ao, hermi=True, xy=None)
# contract to full MO basis
dip_mo_full = np.einsum('xij,ip,jq->xpq', dip_ao, C_active, C_active)

# slice the active-active block consistently with C_active
start = mc.ncore
stop  = mc.ncore + mc.ncas
dip_mo = dip_mo_full[:, start:stop, start:stop]    # shape (3, nmos, nmos)

#- - - Save integrals - - -

print("Dipole moment shapes:")
print(dip_mo.shape)
print(dip_mo[0].shape) #X
print(dip_mo[1].shape) #Y
print(dip_mo[2].shape) #
print("Hamiltonian shape:")
print(h1.shape)
print(h2.shape)

#- - - Save integrals - - -
def save_dipole_integrals(b, dip_mo, folder="tensors"):
    """ Save integrals from a restricted scf scheme.
    b - tittle
    dip_mo :: Dipole moment integrals in MO basis
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b}_dip_mo.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, dip_op=dip_mo)
    print(f"Integrals saved to {filename}")

save_dipole_integrals("P4-RHF", dip_mo, dir)