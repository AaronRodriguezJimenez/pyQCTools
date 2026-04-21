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

p2 ="""
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

mol = gto.M(atom=p2,
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
pi_orbital_space = [14,15,16,20]
mc = mcscf.CASSCF(mf, 4, 4)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object

print(C_active.shape)
print("target_ncas =", int(mc.ncas))
print("ncore =", int(mc.ncore))

h0, h1, h2 = get_restricted_active_space_integrals(mol, mf, mc, localized=False, include_all_noncore=False)

dir = "/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes"
save_restricted_integrals("P2-RHF", h0, h1, h2, dir)

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

save_dipole_integrals("P2-RHF", dip_mo, dir)