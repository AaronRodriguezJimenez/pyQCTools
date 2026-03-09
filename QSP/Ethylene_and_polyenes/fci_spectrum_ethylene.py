"""
 Here we explore the construction of the dipole moment operator for small molecules.
 and also we try to compute a UV spectrum using PySCF
 Pedagogical setup
 Molecule: Ethylene
 basis : sto3g (available for FCI calc)
"""
#
# Compute low-lying electronic excited states of ethylene
# - For ethylene, the important low-energy states are:
#  Type   |  Physical meaning
#  pi-pi*     (HOMO -> LUMO)  [Dominant UV transition]
# singlets    Spin-alllowed
# triplets    Spin-forbidden but important
import numpy as np
from pyscf import gto, scf, fci
from pyscf.lib import logger
from pyqctools.geometries import ethylene
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

# FCI calculation
# 1. Define active space: (number of orbitals, number of active electrons)
# --- determine active space properly ---
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

# Compute singlets
cisolver_s = fci.FCI(mol, C_active)
cisolver_s.spin = 0
e_singlet, ci_singlet = cisolver_s.kernel(nroots=nroots, nelec=(nelec_a, nelec_b))

# Compute triplets
cisolver_t = fci.FCI(mol, C_active)
cisolver_t.spin = 2
e_triplet, ci_triplet = cisolver_t.kernel(nroots=nroots, nelec=(nelec_a+2, nelec_b-2))

hartee_to_ev = 27.2114
exc_singlet = (np.array(e_singlet) - e_singlet[0]) * hartee_to_ev
exc_triplet = (np.array(e_triplet) - e_singlet[0]) * hartee_to_ev

print("Singlet Excitations (eV)")
print(exc_singlet)
print("Triplet Excitations (eV)")
print(exc_triplet)

# Identify states: Look at dominant CI configurations
#print("Singlet States CI coefficients")
#print("(    State    |    CI coeffs    )")
#for i, coeff in enumerate(ci_singlet):
#    print(f"{i+1}     {coeff}")
#
#print("Triplet States CI coefficients")
#print("(    State    |    CI coeffs    )")
#for i, coeff in enumerate(ci_triplet):
#    print(f"{i+1}     {coeff}")

# Compute oscillator strengths
# Spin-traced 1-particle transition density matrix
# <0| p^+ q |1>
print("Info for tdm1 :")
print(len(ci_singlet[0]))
print(len(ci_singlet[1]))
print(nmos)
print(nelec_a, nelec_b)

from math import comb
na_dets = comb(nmos, nelec_a)
nb_dets = comb(nmos, nelec_b)
print("type(ci0):", type(ci_singlet[0]))
print("ci0.shape:", getattr(ci_singlet[0], "shape", None))
print("ci0.size:", np.size(ci_singlet[0]))

print("norb_active, nelec_a, nelec_b:", nmos, nelec_a, nelec_b)
print("na_dets, nb_dets, product:", na_dets, nb_dets, na_dets * nb_dets)

tdm = cisolver_s.trans_rdm1(ci_singlet[0], ci_singlet[1], nmos, (nelec_a, nelec_b)) # transition density matrix for S0-S1

#
# = Next step involve contractions with dipole integrals
# dipole integrals are matrix elements in the AO basis:
# m_\alpha^{pq} = <phi_p|r_\alpha|phi_q>, where \alpha = x, y, z, phi_i are atomic orbitals
# dipole integrals describe how orbitals couple to "light"
#
# * * * Transition dipole moment * * *
# This tell us how strongly light couples two states
#dip_ao = mol.intor('int1e_r', comp=3) #dip_ao[0] = mu_x, ip_ao[1] = mu_y, ip_ao[2] = mu_z
#dip_mo = np.einsum('xij, ip, jq->xpq', dip_ao, C_active, C_active) #change because FCI uses MOs

def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()


def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    '''ints is the integral tensor of a spin-independent operator'''
    if xy is None: xy = tdobj.xy
    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    if not tdobj.singlet:
        return np.zeros((nstates,) + pol_shape)

    mask = tdobj.get_frozen_mask()
    mo_coeff = tdobj._scf.mo_coeff[:, mask]
    mo_occ = tdobj._scf.mo_occ[mask]
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo, orbv.conj())
    ints = np.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo, orbv.conj())
    pol = np.array([np.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], np.ndarray):
        if hermi:
            pol += [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol

with mol.with_common_orig(_charge_center(mol)):
    dip_ao =  mol.intor_symmetric('int1e_r', comp=3)


#dip_mo = _contract_multipole(dip_ao, hermi=True, xy=None)
# contract to full MO basis
dip_mo_full = np.einsum('xij,ip,jq->xpq', dip_ao, C, C)

# slice the active-active block consistently with C_active
start = ncore
stop  = ncore + ncas
dip_mo = dip_mo_full[:, start:stop, start:stop]    # shape (3, nmos, nmos)

#dip_mo = np.einsum('xij, ip, jq->xpq', dip_ao, C_active, C_active) #change because FCI uses MOs

tdip = np.einsum('pq, xpq->x', tdm, dip_mo)
print("Transition dipole (a.u.):", tdip)
print("Magnitude:", np.linalg.norm(tdip))

# OSCILLATOR STRENGTH
# This determines the intensity of the spectra
# f = 2/3 \DeltaE |d_{ge}|^2
delta_e = e_singlet[1] - e_singlet[0]
mu2 = np.dot(tdip, tdip)
f = (2/3) * delta_e * mu2
print("Oscillator strength:", f)

#= = = = = = = = = =
#
# = = = Build a "spectrum" = = =
# Each excitation contributes a peak"
# - Position: Excitation energy
# - Intensity: Oscillator strength
# - Shape: Gaussian/Lorentzian broadening
import matplotlib.pyplot as plt

energies = (np.array(e_singlet) - e_singlet[0]) * hartee_to_ev
osc = []
for i in range(1, nroots):
    tdm1 = cisolver_s.trans_rdm1(ci_singlet[0], ci_singlet[i], nmos, (nelec_a, nelec_b))
    tdip = np.einsum('pq, xpq->x', tdm1, dip_mo)
    #print(f"Transition dipole magnitude <0|tdip|{i}> :", np.linalg.norm(tdip))
    mu2 = np.dot(tdip, tdip)
    delta_e = e_singlet[i] - e_singlet[0]
    f = (2 / 3) * delta_e * mu2
    print(f"Oscillator strenght (0,{i}) :", np.linalg.norm(tdip))
    osc.append(f)

#Energy grid
E = np.linspace(0,35,1000)
spectrum = np.zeros_like(E)

#Gaussian broadening
sigma = 0.2 #eV

for e, f in zip(energies, osc):
    print(e, f)
    spectrum += f * np.exp(-(E-e)**2 / (2*sigma**2))


plt.plot(E, spectrum)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")
plt.title("Test Ethylene sto3g spectrum")
plt.show()