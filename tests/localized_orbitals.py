"""
 Here we compute Localized orbital coefficients (Boys)

 A molecular orbital is usually delocalized, i.e. it has non-negligible amplitude over the
 whole system rather than only around some atoms or bonds. However, one can choose a unitary
 rotation U: phi = psiU
 Such that the resulting phi orbitals are spatially localized as possible.

 The purpose of exploring orbital localization for quantum algorithms, is to see if we can
 gain some advantages/corrections when using localized orbitals in PP, QKSD, VQE, etc.

"""

import numpy
import openfermion
from pyscf import gto, scf, lo

x = .63
mol = gto.M(atom=[['C', (0, 0, 0)],
                  ['H', (x ,  x,  x)],
                  ['H', (-x, -x,  x)],
                  ['H', (-x,  x, -x)],
                  ['H', ( x, -x, -x)]],
            basis='sto3g')
mf = scf.RHF(mol).run()

print("Canonical orbital coefficients:")
print(mf.mo_coeff[:,1])

# C matrix stores the AO to localized orbital coefficients
#C = lo.orth_ao(mf, 'nao') #natural atomic orbital coefficients (in terms of the original atomic orbitals)
C = lo.boys.BF(mol, mf.mo_coeff)
C.kernel()

print("Localized orbital coefficients:")
print(C.mo_coeff[:,1])

def localize_boys(mol, mf):
    # Get h1 and h2 mo_coeffs
    Enuc = mol.energy_nuc()  # Nuclear repulsion
    ao_kin = mol.intor('int1e_kin')  # Kinetic energy
    ao_nuc = mol.intor('int1e_nuc')  # Nuclear-electron attraction
    ao_one = ao_kin + ao_nuc  # One-particle Hamiltonian
    ao_eri = mol.intor('int2e')  # electron-electron repulsion

    # Localize MOs
    CL = lo.boys.BF(mol, mf.mo_coeff)
    CL.kernel()

    # Transform to molecular orbital localized basis
    mo_one_loc = openfermion.general_basis_change(ao_one, CL.mo_coeff, (1, 0))
    mo_eri_loc = openfermion.general_basis_change(ao_eri, CL.mo_coeff, (1, 1, 0, 0))

    return mo_one_loc, mo_eri_loc

# Next we will construct the Fermionic Hamiltonian using these localized orbitals

def fermionop_from_molecule(mol, mo_coeff,
                            freeze_orbitals=None, active_orbitals=None,
                            localized_orbitals=False):
    """
    Constuct a molecular Hamiltonian from a pySCF molecule.
    :param mol: mol object from PySCF
    :param mo_coeff: array of the basis transformation from atomic to molecular orbitals
    :param freeze_orbitals: list of spatial orbital indices to "freeze"
    :param active_orbitals: list of active orbitals
    :return: fermion_operator in OpenFermion format.
    """
    if freeze_orbitals is None: freeze_orbitals = []
    if active_orbitals is None: active_orbitals = list(range(mol.nao))

    # Compute Atomic orbital Integrals
    Enuc = mol.energy_nuc()            #Nuclear repulsion
    ao_kin = mol.intor('int1e_kin')   #Kinetic energy
    ao_nuc = mol.intor('int1e_nuc')  #Nuclear-electron attraction
    ao_one = ao_kin + ao_nuc        #One-particle Hamiltonian
    ao_eri = mol.intor('int2e') #electron-electron repulsion

    # Transform to molecular orbital basis
    if localized_orbitals:
        mo_one, mo_eri = localize_boys(mol, mf)
    else:
        mo_one = openfermion.general_basis_change(ao_one, mo_coeff, (1,0))
        mo_eri = openfermion.general_basis_change(ao_eri, mo_coeff, (1,1,0,0))

    mo_eri = mo_eri.transpose(0,2,3,1)  #OpenFermion Tensor convention.

    # Trace out core orbitals and drop inactive virtuals
    core, one, eri = openfermion.ops.representations.get_active_space_integrals(mo_one,
                                             mo_eri, freeze_orbitals, active_orbitals)

    #
    # Now we can contruct the molecular hamiltonian from the spatial orbital integrals
    #
    h0 = Enuc + core
    print(h0, h0)

    h1, h2 = openfermion.ops.representations.get_tensors_from_integrals(one, eri)
    interop = openfermion.InteractionOperator(h0, h1, h2)
    fop = openfermion.get_fermion_operator(interop) #fop = fermion operator
    return fop

H_fermion = fermionop_from_molecule(mol, mf.mo_coeff, localized_orbitals=False)

print("Fermion Operator with localized orbitals is:")
print(H_fermion)

# JWT
hamiltonian_jw = openfermion.transforms.jordan_wigner(H_fermion)

print("Qubit operator ")
from pyqctools.ham_fcns import jw_to_sparse_pauli_op
H_op = jw_to_sparse_pauli_op(hamiltonian_jw)
print(H_op)
