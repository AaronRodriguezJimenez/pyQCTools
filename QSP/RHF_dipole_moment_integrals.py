"""
 Dipole moment integrals for C2H4
"""
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.lib import logger
from pyqctools.geometries import ethylene
import os

mol = gto.Mole()
mol.atom = ethylene()
mol.basis = "sto3g"
#mol.symmetry = True
mol.verbose = 4
mol.output = "./C2H4.out"
mol.build()

# Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

#Before anything check stability:
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
mf = stable_opt_internal(mf, stability_cycles)

# Froze core
ncore = 2
nroots = 10
ncas = mol.nao - ncore
nelecas = mol.nelectron - 2*ncore   # total active electrons
# split into alpha/beta (for singlet ground state, equally split)
nelec_a = nelecas // 2
nelec_b = nelecas // 2 # For s=0 #- nelec_a        # handles odd number of electrons if any

# active MO coefficients used by FCI
C = mf.mo_coeff
C_active = C[:, ncore:ncore+ncas]
nmos = C_active.shape[1]

#= = = Save Hamiltonian integrals = = =
def save_tensors(b, h0, h1, h2, folder="tensors"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b}_tensors.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Tensors saved to {filename}")

def get_active_space_tensors(mol, mf, localized=False):
    """
    Robustly extracts active space tensors from SymAdaptedCASSCF or CASSCF.
    """
    # Froze core
    ncore = 2
    ncas = mol.nao - ncore

    # active MO coefficients used by FCI
    C = mf.mo_coeff
    C_active = C[:, ncore:ncore + ncas]

    nuc = mol.energy_nuc()  # Nuclear repulsion.
    ao_kin = mol.intor('int1e_kin')  # Electronic kinetic energy.
    ao_nuc = mol.intor('int1e_nuc')  # Nuclear-electronic attraction.
    ao_obi = ao_kin + ao_nuc  # Single-electron hamiltonian.
    ao_eri = mol.intor('int2e')  # Electonic repulsion interaction.
    C = mf.mo_coeff

    # 1. Handle Localization if requested
    if localized:
        from pyscf import lo
        nocc = int((mf.mo_occ > 0).sum())
        occ_idx_act = np.arange(0, nocc)
        vir_idx_act = np.arange(nocc, len(C))
        # Split MOs into occupied / virtual subsets
        C_act_occ = C[:, occ_idx_act]
        C_act_vir = C[:, vir_idx_act]

        # Boys localization
        loc_occ = lo.Boys(mol, C_act_occ).kernel(verbose=4)
        loc_vir = lo.Boys(mol, C_act_vir).kernel(verbose=4)

        C_loc = np.column_stack((loc_occ, loc_vir))
        mo_coeff_final = C_loc[:, ncore:ncore + ncas]

    else:
        mo_coeff_final = C_active

    U = mo_coeff_final

    h1_spatial = U.T @ ao_obi @ U  # Rotated single-body integrals.
    h2_spatial = ao2mo.incore.full(ao_eri, U)  # Rotated two-body integrals.
    h0 = nuc
    return h0, h1_spatial, h2_spatial

h0, h1, h2 = get_active_space_tensors(mol, mf, localized=True)
dir = "/Users/admin/PycharmProjects/pyQCTools/QSP/dipole_moment"
save_tensors("c2h4-RHF", h0, h1, h2, dir)

#= = = Dipole moment = = =
def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()

with mol.with_common_orig(_charge_center(mol)):
    dip_ao =  mol.intor_symmetric('int1e_r', comp=3)

#dip_mo = _contract_multipole(dip_ao, hermi=True, xy=None)
# contract to full MO basis
dip_mo_full = np.einsum('xij,ip,jq->xpq', dip_ao, C, C)

# slice the active-active block consistently with C_active
start = ncore
stop  = ncore + ncas
dip_mo = dip_mo_full[:, start:stop, start:stop]    # shape (3, nmos, nmos)

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

print("Dipole moment shapes:")
print(dip_mo.shape)
print(dip_mo[0].shape) #X
print(dip_mo[1].shape) #Y
print(dip_mo[2].shape) #
print("Hamiltonian shape:")
print(h1.shape)
print(h2.shape)
save_dipole_integrals("c2h4-RHF", dip_mo, dir)