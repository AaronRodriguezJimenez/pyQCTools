from pyscf import gto, scf, lo, mcscf
import pyqctools as pq
from pyscf.lib import logger
import numpy as np
import os

# 1. Define Molecule
benzene = pq.benzene()
mol = gto.M(atom=benzene,
            basis='cc-pvdz',
)

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

#- - Check SCF Stability
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
pi_orbital_space = [17,20,21,22,23,30]
mc = mcscf.CASSCF(mf, 6, 6)
C_active = mc.sort_mo(pi_orbital_space)
mf.mo_coeff = C_active #Retrieve ordered orbitals to mf object

#print(mf.mo_coeff[:, 16])
print(C_active.shape)
print("target_ncas =", int(mc.ncas))
print("ncore =", int(mc.ncore))

from src.pyqctools.int_fcns import get_restricted_active_space_integrals
from src.pyqctools.int_fcns import save_restricted_integrals

h0, h1, h2 = get_restricted_active_space_integrals(mol, mf, mc, localized=False, include_all_noncore=False)

dir = "/Users/admin/PycharmProjects/pyQCTools/QSP/Benzene_and_acenes"
save_restricted_integrals("benz-RHF", h0, h1, h2, dir)

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

save_dipole_integrals("benz-RHF", dip_mo, dir)