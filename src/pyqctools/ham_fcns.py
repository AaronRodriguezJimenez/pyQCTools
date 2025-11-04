# Automatic functions that return a given molecular Hamiltonian from different sources:
# - qiskit-nature
# - openfermion

import openfermion as of
from tabulate import tabulate
from openfermion.utils import count_qubits
from qiskit.quantum_info import SparsePauliOp, Pauli
from openfermion import QubitOperator
import numpy as np
import scipy as sp
import itertools as it

def convert_openfermion_to_qiskit(openfermion_operator: QubitOperator, num_qubits: int) -> SparsePauliOp:
    terms = openfermion_operator.terms

    labels = []
    coefficients = []

    for term, constant in terms.items():
        # Default set to identity
        operator = list('I' * num_qubits)

        # Iterate through PauliSum and replace I with Pauli
        for index, pauli in term:
            operator[index] = pauli
        label = ''.join(operator)
        labels.append(label)
        coefficients.append(constant)

    return SparsePauliOp(labels, coefficients)

# Function to get the QISKIT Nature Hamiltonian for a given molecule
def get_QkiskitNature_hamiltonian(geometry, basis, spin, ch, interleaved=True):
    """
    Get the Hamiltonian for a given molecule using Qiskit Nature
    - This Hamiltonian is in the Jordan-Wigner form without the nuclear repulsion energy.
    :return: SparsePauliOp Hamiltonian and number of qubits
    """
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    driver = PySCFDriver(
        atom=geometry,
        basis=basis,
        spin=spin,
        charge=ch,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()
    #hamiltonian = problem.hamiltonian
    #print('Nuclear repulsion :', hamiltonian.nuclear_repulsion_energy)
    #coefficients = hamiltonian.electronic_integrals

    fermionic_op = problem.hamiltonian.second_q_op()

    # JORDAN-WIGNER MAPPING
    if interleaved:
        from qiskit_nature.second_q.mappers import InterleavedQubitMapper
        interleaved_mapper = InterleavedQubitMapper(JordanWignerMapper())
        qubit_jw_op = interleaved_mapper.map(fermionic_op)

    else:
        mapper = JordanWignerMapper()
        qubit_jw_op = mapper.map(fermionic_op)

    n_qubits = qubit_jw_op.num_qubits

    return qubit_jw_op, n_qubits

def get_hamiltonian(geometry, basis, mult, ch, print_table=False, NucRepterm=True):
    '''
    Computes the Hamiltonian for a given molecule/atom under the Jordan-Wigner transformation
    using  OpenFermion and OpenFermion-PySCF. The resulting Hamiltonian is in interleaved form.
    thus |0011> is the HF singlet state for H2 molecule (for example).
    It does not contain the nuclear repulsion energy.
    :geometry: geometry of the molecule/atom
    :basis: basis set as string
    :mult: multiplicity
    :ch: charge
    :NucRepterm: Add nuclear repulsion energy term if True
    :print_table: whether to print the table of the Hamiltonian
    :return: hamiltonian as a qubit operator
    '''
    # Perform electronic structure calc and get Hamiltonian as an InteractionOperator
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf
    molecule = MolecularData(geometry, basis, mult, ch)

    #run pyscf-HF
    molecule = run_pyscf(molecule,
                         run_scf=True)

    #hamiltonian_intop = ofpyscf.generate_molecular_hamiltonian(geometry, basis, mult, ch)
    hamiltonian_intop = molecule.get_molecular_hamiltonian()

    if NucRepterm == False:
        nuc_rep = molecule.nuclear_repulsion
        print("Nuclear Repulsion Term:", nuc_rep)
        hamiltonian_intop[''] = 0.0 # This is equivalent to ERASE the nuclear repulsion energy from the Hamiltonian
        hamiltonian_ferm_op = of.transforms.get_fermion_operator(hamiltonian_intop)
    else:
        # Convert COMPLETE InteractionOperator to FermionOperator
        hamiltonian_ferm_op = of.transforms.get_fermion_operator(hamiltonian_intop)

    if print_table:
        # Print the table of the Hamiltonian
        print('- - - Hamiltonian as FermionOperator - - -')
        print(f"Nuclear repulsion is: {molecule.nuclear_repulsion} a.u.")
        table_data =[]
        data_dict = hamiltonian_ferm_op.terms

        for key, value in data_dict.items():

            if key == ():

                key_str = 'Empty'
                nuclear_term = value

            else:

                key_str = str(key)
            table_data.append([key_str, value])

        headers = ["Key", "Value"]
        print(tabulate(table_data, headers, tablefmt="pretty"))

    # Convert to a QubitOperator using the JWT
    hamiltonian_jw = of.transforms.jordan_wigner(hamiltonian_ferm_op)
    ansatz_exponent = 1j * hamiltonian_jw

    #check consistency
    n_qubits = count_qubits(ansatz_exponent)

    return hamiltonian_jw, n_qubits

def empty_vector(n_qubits):
    # Returns a list of len n_qubits, where each element is 0
    vect = [0] * n_qubits
    return vect


# Create PauliOp form of the JW Hamiltonian
def jw_to_sparse_pauli_op(qubit_operator):
    """
    Converts an OpenFermion Jordan-Wigner QubitOperator to Qiskit's SparsePauliOp.

    Args:
        qubit_operator (QubitOperator): The QubitOperator from OpenFermion.

    Returns:
        SparsePauliOp: The equivalent operator in Qiskit's SparsePauliOp format.
    """
    pauli_strings = []
    coefficients = []
    num_qubits = count_qubits(qubit_operator)

    for term, coefficient in qubit_operator.terms.items():
        #print(term, coefficient)
        # Create an identity string of appropriate length
        pauli_string = ['I'] * num_qubits

        # Replace 'I' with the appropriate Pauli operator at the specified index
        for index, pauli in term:
            pauli_string[index] = pauli

        # Join the Pauli operators into a single string and append
        pauli_strings.append(''.join(pauli_string))
        coefficients.append(complex(coefficient))
        #print(pauli_strings)
    # Create SparsePauliOp

    return SparsePauliOp(pauli_strings, coefficients)


def heisenberg_single_particle_gs(H_op, n_qubits):
    """
    Find the ground state of the single particle(excitation) sector
    """

    H_x = []
    for p, coeff in H_op.to_list():
        H_x.append(set([i for i,v in enumerate(Pauli(p).x) if v]))

    H_z = []
    for p, coeff in H_op.to_list():
        H_z.append(set([i for i,v in enumerate(Pauli(p).z) if v]))

    H_c = H_op.coeffs

    print('n_sys_qubits', n_qubits)

    n_exc = 1
    sub_dimn = int(sp.special.comb(n_qubits+1,n_exc))
    print('n_exc', n_exc, ', subspace dimension', sub_dimn)

    few_particle_H = np.zeros((sub_dimn,sub_dimn), dtype=complex)

    sparse_vecs = [set(vec) for vec in it.combinations(range(n_qubits+1),r=n_exc)] # list all of the possible sets of n_exc indices of 1s in n_exc-particle states

    m = 0
    for i, i_set in enumerate(sparse_vecs):
        for j, j_set in enumerate(sparse_vecs):
            m += 1

            if len(i_set.symmetric_difference(j_set)) <= 2:

                for p_x, p_z, coeff in zip(H_x, H_z, H_c):

                    if i_set.symmetric_difference(j_set) == p_x:
                        sgn = ((-1j)**len(p_x.intersection(p_z)))*((-1)**len(i_set.intersection(p_z)))
                    else:
                        sgn = 0

                    few_particle_H[i,j] += sgn*coeff

    gs_en = min(np.linalg.eigvalsh(few_particle_H))
    print('single particle ground state energy: ', gs_en)
    return gs_en

if __name__ == "__main__":
    # Example usage
    geometry= [
        ('H', (0.0000,  0.0000,  0.0000)),
        ('H', (0.0000,  0.0000,  0.7400)),
        ('H', (0.0000,  0.0000,  1.4800)),
        ('H', (0.0000,  0.0000,  2.2200))]

    basis = 'sto-3g'
    mult = 1
    ch = 0
    hamiltonian, n_qubits = get_hamiltonian(geometry, basis, mult, ch, print_table=True)
    print(f"Number of qubits: {n_qubits}")
    empty_vector = empty_vector(n_qubits)
    print(empty_vector)
    print('The openfermion Qubit Hamiltonian is: ', hamiltonian)
    print('The Sparse pauli form is:')
    H_sparse_pauli = jw_to_sparse_pauli_op(hamiltonian)
    print(H_sparse_pauli)
    print('The converted to Qiskit SparsePauliOp is: ')
    qiskit_operator = convert_openfermion_to_qiskit(hamiltonian, n_qubits)
    print(qiskit_operator)

    # Use Qiskit Nature to get the Hamiltonian
    print('- - - QISKIT NATURE HAMILTONIAN - - -')
    spin = 0
    geometry = "H 0 0 0; H 0 0 0.74"
    qubit_jw_op, n_qubits = get_QkiskitNature_hamiltonian(geometry, basis, spin, ch)
    print(f"Number of qubits: {n_qubits}")
    print('The Qiskit Nature Hamiltonian is: ', qubit_jw_op)