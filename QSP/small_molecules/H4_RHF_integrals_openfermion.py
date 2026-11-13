"""
 Full-orbtial space RHF integrals for H2
 Integrals are stored into an npz file, and this file is read by openfermion
 to construct the Hamiltonian.
"""
import numpy as np
import os
from pyscf import gto, scf
from pyscf.lib import logger
from pyqctools.geometries import HChain
from src.pyqctools.int_fcns import get_restricted_ints_full
from src.pyqctools.int_fcns import save_restricted_integrals

# Create output directory, and check if it exists
output_dir = "./H4_integrals_OF"
if not os.path.exists(output_dir):
    print(f"Creating output directory:, {output_dir}")
    os.makedirs(output_dir)
else:
    print(f"Output directory already exists, {output_dir}")

# Create mol object and outputfile name
mol = gto.Mole()
mol.atom = HChain(4, 0.7414)
mol.basis = "sto-3g"
#mol.symmetry = True
mol.verbose = 4

output_file = os.path.join(output_dir, f"H4_{mol.basis}.out")
print(f"Output File: {output_file}")
mol.output = output_file

mol.build()

# 2. Run Restricted Hartree-Fock (RHF) calculation
mf = scf.RHF(mol)
e_hf = mf.kernel()
print(f"RHF Energy: {e_hf:.6f} Ha")

#- - Check SCF Stability
    # Loop for optimizing orbitals until stable
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

h0, h1, h2 = get_restricted_ints_full(mol, mf, localized=False)
filename = f"H4-RHF_{mol.basis}"
save_restricted_integrals(filename, h0, h1, h2, folder="H4_integrals_OF")

# * * * * * *
# READ INTEGRALS AND CONSTRUCT HAMILTONIAN with OPENFERMION
# * * * * * *
from pyscf import ao2mo, fci
from openfermion import InteractionOperator, jordan_wigner, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import jw_get_ground_state_at_particle_number
from qiskit.quantum_info import SparsePauliOp

# ----------------------------------------------------------------------------
# 2. Read .npz
# ----------------------------------------------------------------------------
def load_ints(fname):
    data = np.load(fname)
    h0 = float(data["hc"])
    h1 = np.asarray(data["h1e"])
    h2 = np.asarray(data["h2e"])
    norb = h1.shape[0]
    # ao2mo returns a compact (npair, npair) array -> expand to (n, n, n, n)
    if h2.ndim != 4:
        h2 = ao2mo.restore(1, h2, norb)
    return h0, h1, h2

# ----------------------------------------------------------------------------
# 3. OpenFermion fermionic + qubit Hamiltonian
# ----------------------------------------------------------------------------
def build_openfermion_hamiltonian(h0, h1, h2_chem):
    #OpenFermion stores the coefficient of $a^\dagger_p a^\dagger_q a_r a_s$ (with the ½ in front),
    # which is the physicists’ integral ⟨pq|sr⟩ = (ps|qr).
    # That’s here weneed to transpose(0, 2, 3, 1) before building the InteractionOperator.
    h2_phys = np.asarray(h2_chem.transpose(0, 2, 3, 1), order="C")
    one_body, two_body = spinorb_from_spatial(h1, h2_phys)
    fermion_op = InteractionOperator(h0, one_body, 0.5 * two_body)
    qubit_op = jordan_wigner(fermion_op)
    qubit_op.compress()                       # drop ~0 terms
    return fermion_op, qubit_op

# ----------------------------------------------------------------------------
# 4. OpenFermion QubitOperator -> Qiskit SparsePauliOp
# ----------------------------------------------------------------------------
def openfermion_to_qiskit(qubit_op, n_qubits):
    """
    from_sparse_list takes explicit qubit indices, so Qiskit's little-endian
    label ordering is handled automatically (OpenFermion qubit i -> Qiskit qubit i).
    """
    sparse_list = []
    for term, coeff in qubit_op.terms.items():
        paulis = "".join(p for _, p in term)
        indices = [i for i, _ in term]
        sparse_list.append((paulis, indices, coeff))   # term == () -> identity
    op = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits)
    op = op.simplify()
    if np.allclose(op.coeffs.imag, 0.0):
        op = SparsePauliOp(op.paulis, op.coeffs.real)
    return op

h0, h1, h2 = load_ints(os.path.join(output_dir, f"H4-RHF_{mol.basis}_integrals.npz"))
nelec = mol.nelectron
norb = h1.shape[0]
n_qubits = 2 * norb

print("Number of qubits:", n_qubits)
fermion_op, qubit_op = build_openfermion_hamiltonian(h0, h1, h2)
print("Number of terms in QubitOperator:", len(qubit_op.terms))
op_qiskit = openfermion_to_qiskit(qubit_op, n_qubits)
print("Number of terms in Qiskit SparsePauliOp:", len(op_qiskit.paulis))
print(op_qiskit)


# --- Checks ---------------------------------------------------------------
# (a) ground state in the correct particle-number sector (not the global min!)
e_of, _ = jw_get_ground_state_at_particle_number(
    get_sparse_operator(qubit_op, n_qubits), nelec)

# (b) same matrix from Qiskit
M_qk = op_qiskit.to_matrix(sparse=True)
M_of = get_sparse_operator(qubit_op, n_qubits)
print("Qiskit == OpenFermion matrix:", abs(M_qk - M_of).max() < 1e-10)

# (c) PySCF FCI reference with the same integrals
e_fci, _ = fci.direct_spin1.kernel(h1, h2, norb, nelec, ecore=h0)

print(f"RHF  energy : {mf.e_tot:.10f}")
print(f"FCI  (PySCF): {e_fci:.10f}")
print(f"JW   (OF)   : {e_of:.10f}")