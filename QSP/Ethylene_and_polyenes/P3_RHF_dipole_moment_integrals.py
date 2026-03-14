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

p3 ="""
  C          -0.59691673020855      0.30127414661253     -0.00001136360562
  C           0.59691563766006     -0.30129125807882      0.00000053581842
  C          -1.86804608686186     -0.41509590916692     -0.00000963704951
  C           1.86804514506670      0.41508092209292     -0.00001764031881
  C          -3.05829004310842      0.18263535992774     -0.00004608231215
  C           3.05829019206652     -0.18264778731851     -0.00002074609123
  H          -0.64612966472928      1.38897315062128     -0.00003311715897
  H           0.64613066685960     -1.38899103866914      0.00002234449039
  H          -1.81182015020897     -1.50178133143489      0.00001944549488
  H           1.81181784661532      1.50176608248966     -0.00002958338871
  H          -3.98002471737652     -0.38766336395150     -0.00004961142061
  H          -3.14576530696508      1.26519154024145     -0.00007824430025
  H           3.98002384171624      0.38765260286158     -0.00003743160775
  H           3.14576936947424     -1.26520411622738     -0.00000786855006
"""

mol = gto.M(atom=p3,
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
pi_orbital_space = [20,21,22,23,25,30]
mc = mcscf.CASSCF(mf, 6, 6)
C_active = mc.sort_mo(pi_orbital_space)

print(C_active.shape)
print("target_ncas =", int(mc.ncas))
print("ncore =", int(mc.ncore))

h0, h1, h2 = get_restricted_active_space_integrals(mol, mf, mc, localized=False, include_all_noncore=False)

dir = "/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes"
save_restricted_integrals("P3-RHF", h0, h1, h2, dir)

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

save_dipole_integrals("P3-RHF", dip_mo, dir)