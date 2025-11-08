# General utilities

#- PySCF
def set_basis_and_build(mol, basis):
    mol.basis = basis
    mol.build()
    return mol
def compute_scf_energy(mf):
    return mf.kernel()

# The following have been taken and in some cases resemble modified versions of
# PySCFWorkspace created by Kyle Sherbert

""" """
import openfermion
import numpy

def totalparitycheck(n, η):
    """ The binary code for a Z2 symmetry corresponding to total particle number.

    Parameters
    ----------
    n (int): the number of qubits
    η (int): the number of particles

    After applying this code, all basis states correspond to configurations
        with a number of particles having the same parity as `η`.

    """
    return openfermion.checksum_code(n, η & 1)

def spinparitycheck(n, ηα, ηβ):
    """ The binary code for two Z2 symmetries,
        corresponding to total particle number in each of the α and β registers.

    Parameters
    ----------
    n (int): the number of qubits
    ηα (int): the number of particles in the α register (even indices)
    ηβ (int): the number of particles in the β register (odd indices)

    After applying this code, all basis states correspond to configurations
        with a number of α particles having the same parity as `ηα`,
        and a number of β particles having the same parity as `ηβ`.

    """
    L = n >> 1
    α_code = openfermion.checksum_code(L, ηα & 1)
    β_code = openfermion.checksum_code(L, ηβ & 1)
    stagger = openfermion.interleaved_code(n)
    return stagger * (α_code + β_code)

def taperedcode(n, ηα, ηβ, codefn=openfermion.jordan_wigner_code, taper=2):
    """ Construct a binary code corresponding to the given qubit mapping and tapering.

    Parameters
    ----------
    n (int): the number of qubits
    ηα (int): the number of particles in the α register (even indices)
    ηβ (int): the number of particles in the β register (odd indices)
    codefn: a function taking a number of qubits and outputting a binary code,
        typically `openfermion.jordan_wigner_code` or similar.
    taper: the number of qubits to taper. Must be 0, 1, or 2.

    Assumptions
    -----------
    taper=0 is always safe.
    taper=1 assumes your Hamiltonian preserves particle number.
    taper=2 assumes your Hamiltonian preserves spin,
        and that your fermionic indices are interleaved like αβαβ,
        as is standard in openfermion (but not qiskit!).

    """
    tapercode = (
        1 if taper == 0 else
        totalparitycheck(n, ηα+ηβ) if taper == 1 else
        spinparitycheck(n, ηα, ηβ) if taper == 2 else
        ValueError(f"Invalid `taper={taper}` - must be one of {0,1,2}.")
    )
    return tapercode * codefn(n-taper)

def referencevector(n, ηα, ηβ):
    """ Construct the bitvector representation of the Hartree-Fock state.

    Parameters
    ----------
    n (int): the number of qubits
    ηα (int): the number of α particles
    ηβ (int): the number of β particles

    Assumptions
    -----------
    Fermionic indices are sorted from lowest energy to highest,
        and are interleaved like αβαβ,
        as is standard in openfermion (but not qiskit!).

    """
    bitvector = numpy.zeros(n, dtype=int)
    for i in range(ηα): bitvector[2*i] = 1
    for i in range(ηβ): bitvector[2*i+1] = 1
    return bitvector

def encode_vector(bitvector, code=None):
    """ Encode a fermionic occupation bitvector with the given binary code.

    The default `code` is Jordan-Wigner, whose encoding function is identity,
        so you'll get the same bitvector out.

    """
    if code is None:    return bitvector
    return (code.encoder @ bitvector) % 2

def encode_operator(fermiop, code=None):
    """ Encode a fermionic operator into a qubit operator with the given binary code.

    The default `code` is Jordan-Wigner.

    """
    if code is None:    return openfermion.jordan_wigner(fermiop)
    return openfermion.binary_code_transform(fermiop, code)


def quantum_numbers(molecule, freeze_orbitals=None, active_orbitals=None):
    """ Programmatically extract a suite of integers from a pyscf molecule.

    Parameters
    ----------
    molecule: a pre-built pyscf molecule object
    freeze_orbitals: a list of spatial orbital indices to trace out as "core orbitals".
        Defaults to an empty list.
    active_orbitals: a list of spatial orbital indices to retain
        Defaults to a list of all indices.

    Any indices not present in either list will be omitted.
    Defaults are not magic. If you decide to provide one list, you must provide both.

    Returns
    -------
    nspatial: number of spatial orbitals in active space
    nspin: number of spin orbitals in active space
    η: number of electrons in active space
    ηα: number of α electrons in active space
    ηβ: number of β electrons in active space

    """
    if freeze_orbitals is None:  freeze_orbitals = []
    if active_orbitals is None:  active_orbitals = list(range(molecule.nao))
    nspatial = len(active_orbitals)     # Number of active spatial orbitals.
    nspin = 2*nspatial                  # Number of active spin orbitals.
    ηα, ηβ = molecule.nelec             # Total α and β electron counts.
    ηα -= len(freeze_orbitals)          # Account for filled core orbitals.
    ηβ -= len(freeze_orbitals)          #       "                   "
    η = ηα + ηβ                         # Number of active electrons.
    return nspatial, nspin, η, ηα, ηβ

def fermiop_from_integrals(nuc, obi, eri):
    """ Construct a molecular Hamiltonian from the spatial orbital integrals.

    Parameters
    ----------
    nuc: scalar constant shift in energy (usually that due to NUClear repulsion)
    obi: N⨯N ndarray of One-Body Integrals
    eri: N⨯N⨯N⨯N ndarray of two-body integrals, aka Electronic Repulsion Interaction

    N refers to the number of spatial orbitals in the active space.

    """
    h1, h2 = openfermion.ops.representations.get_tensors_from_integrals(obi, eri)
    interop = openfermion.InteractionOperator(nuc, h1, h2)
    return openfermion.get_fermion_operator(interop)

def fermiop_from_molecule(
    molecule, mo_coeff,
    freeze_orbitals=None,
    active_orbitals=None,
):
    """ Construct a molecular Hamiltonian from a pyscf molecule.

    Parameters
    ----------
    molecule: a pre-built pyscf molecule object
    mo_coeff: the basis transformation from atomic to molecular orbitals
    freeze_orbitals: a list of spatial orbital indices to trace out as "core orbitals".
        Defaults to an empty list.
    active_orbitals: a list of spatial orbital indices to retain
        Defaults to a list of all indices.

    Any indices not present in either list will be omitted.
    Defaults are not magic. If you decide to provide one list, you must provide both.

    """
    if freeze_orbitals is None:  freeze_orbitals = []
    if active_orbitals is None:  active_orbitals = list(range(molecule.nao))

    # COMPUTE ATOMIC ORBITAL INTEGRALS
    nuc = molecule.energy_nuc()             # Nuclear repulsion.
    ao_kin = molecule.intor('int1e_kin')    # Electronic kinetic energy.
    ao_nuc = molecule.intor('int1e_nuc')    # Nuclear-electronic attraction.
    ao_obi = ao_kin + ao_nuc                # Single-electron hamiltonian.
    ao_eri = molecule.intor('int2e')        # Electonic repulsion interaction.

    # TRANSFORM TO MOLECULAR ORBITAL BASIS
    full_obi = openfermion.general_basis_change(ao_obi, mo_coeff, (1,0))
    full_eri = openfermion.general_basis_change(ao_eri, mo_coeff, (1,1,0,0))
    full_eri = full_eri.transpose(0,2,3,1)  # Switch to openfermion tensor convention.

    # TRACE OUT CORE ORBITALS AND DROP INACTIVE VIRTUALS
    core, obi, eri = openfermion.ops.representations.get_active_space_integrals(
        full_obi, full_eri,
        freeze_orbitals, active_orbitals,
    )

    return fermiop_from_integrals(nuc+core, obi, eri)

def save_system(file, qubitop, ket):
    """ Save a qubit operator and reference state into a compact `.npz` file.

    The reference `ket` is saved directly.
    The `qubitop` is broken down into a list of coefficients `C`
        and two integer lists `X` and `Z` identifying each Pauli word.
    The integers' binary representations identify bits in the word with X and Z character.
    By definition, a bit with the Y Pauli operator has both X *and* Z character,
        as is standard in the symplectic representation of Paulis.

    """
    n = len(ket)
    X = []; Z = []; C = []
    for actions, c in qubitop.terms.items():
        x = 0; z = 0
        for (q, σ) in actions:
            mask = 1<<(n-q-1)
            if σ == "X" or σ == "Y": x ^= mask
            if σ == "Z" or σ == "Y": z ^= mask
        X.append(x); Z.append(z); C.append(c)
    numpy.savez(file, X=X, Z=Z, C=C, ket=ket)

if __name__ == "__main__":
    n = 8
    na = 4
    nb = 0
    bitvect = referencevector(n, na, nb)
    print(bitvect)