import numpy as np
from functools import reduce

from pyscf import gto, scf, ao2mo
from pyscf.lo.iao import iao
from pyscf.lo import orth
from MEAO.meao import MEAO


def build_meao_hamiltonian(
    atom,
    basis="ccpvdz",
    minao_basis="minao",
    charge=0,
    spin=0,
    unit="A",
    n_active_orbitals=None,
    active_indices=None,
):
    """
    Build one- and two-electron molecular Hamiltonian integrals
    in the MEAO basis.

    Returns
    -------
    result : dict
        {
            "mol": PySCF molecule,
            "mf": RHF object,
            "C_meao": AO x MEAO coefficient matrix,
            "h1e": one-electron integrals in MEAO basis,
            "eri": two-electron integrals in MEAO basis, chemist notation,
            "ecore": nuclear repulsion energy,
            "norb": number of orbitals,
            "nelec": total electron count,
            "bonds": MEAO bond-pair list
        }
    """

    # Main molecule
    mol = gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        symmetry=False,
        verbose=0,
    )

    # Minimal reference molecule, used to determine atomic orbital partitions
    pmol = gto.M(
        atom=atom,
        basis=minao_basis,
        charge=charge,
        spin=spin,
        unit=unit,
        symmetry=False,
        verbose=0,
    )

    aoslices = pmol.aoslice_by_atom()
    norbs_in_atoms = [
        aoslices[ia][3] - aoslices[ia][2]
        for ia in range(pmol.natm)
    ]

    # RHF reference
    mf = scf.RHF(mol)
    mf.kernel()

    # Build IAOs, matching the MEAO example
    orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
    c_iao = iao(mol, orbocc)

    s = mol.intor("int1e_ovlp")
    mo_iao = c_iao @ orth.lowdin(c_iao.T @ s @ c_iao)

    # Run MEAO optimization
    my_meao = MEAO(mol, mf, mo_iao, norbs_in_atoms)
    my_meao.meao()

    C_meao_full = my_meao.mo_meao
    bonds = my_meao.get_bonds()

    # Choose full MEAO basis or active subset
    if active_indices is not None:
        C_meao = C_meao_full[:, active_indices]
    elif n_active_orbitals is not None:
        C_meao = C_meao_full[:, :n_active_orbitals]
    else:
        C_meao = C_meao_full

    norb = C_meao.shape[1]

    # One-electron integrals
    h_ao = mf.get_hcore()
    h1e = C_meao.T @ h_ao @ C_meao

    # Two-electron integrals in MEAO basis
    eri = ao2mo.kernel(mol, C_meao, compact=False)
    eri = eri.reshape(norb, norb, norb, norb)

    result = {
        "mol": mol,
        "mf": mf,
        "meao": my_meao,
        "C_meao": C_meao,
        "C_meao_full": C_meao_full,
        "h1e": h1e,
        "eri": eri,
        "ecore": mol.energy_nuc(),
        "norb": norb,
        "nelec": mol.nelectron,
        "bonds": bonds,
    }

    return result

def meao_integrals(
            mol,
            mf,
            n_active_orbitals=None,
            active_indices=None,
    ):
        """
        Build one- and two-electron molecular Hamiltonian integrals
        in the MEAO basis. This assumes that a RHF mf stable object
        is assed as an argument

        Returns
        -------
        result : dict
            {
                "h1e": one-electron integrals in MEAO basis,
                "eri": two-electron integrals in MEAO basis, chemist notation,
                "ecore": nuclear repulsion energy,
                "norb": number of orbitals,
                "nelec": total electron count,
                "bonds": MEAO bond-pair list
            }
        """
        # minimal ref molecule
        pmol = mol.copy()

        aoslices = pmol.aoslice_by_atom()
        norbs_in_atoms = [
            aoslices[ia][3] - aoslices[ia][2]
            for ia in range(pmol.natm)
        ]

        # Build IAOs, matching the MEAO example
        orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
        c_iao = iao(mol, orbocc)

        s = mol.intor("int1e_ovlp")
        mo_iao = c_iao @ orth.lowdin(c_iao.T @ s @ c_iao)

        # Run MEAO optimization
        my_meao = MEAO(mol, mf, mo_iao, norbs_in_atoms)
        my_meao.meao()

        C_meao_full = my_meao.mo_meao
        #bonds = my_meao.get_bonds()

        # Choose full MEAO basis or active subset
        if active_indices is not None:
            C_meao = C_meao_full[:, active_indices]
        elif n_active_orbitals is not None:
            C_meao = C_meao_full[:, :n_active_orbitals]
        else:
            C_meao = C_meao_full

        norb = C_meao.shape[1]

        # One-electron integrals
        h_ao = mf.get_hcore()
        h1e = C_meao.T @ h_ao @ C_meao

        # Two-electron integrals in MEAO basis
        eri = ao2mo.kernel(mol, C_meao, compact=False)
        eri = eri.reshape(norb, norb, norb, norb)

        result = {
            "meao": my_meao, #class object
            "h1e": h1e,
            "eri": eri,
            "ecore": mol.energy_nuc()
        }

        return result

if __name__ == "__main__":
    ham = build_meao_hamiltonian(
        atom="N 0 0 0; N 0 0 1.098",
        basis="ccpvdz",
        spin=0,
    )
    print("Number of MEAOs:", ham["norb"])
    print("Number of electrons:", ham["nelec"])
    print("MEAO bonds:", ham["bonds"])
    print("h1e shape:", ham["h1e"].shape)
    print("eri shape:", ham["eri"].shape)