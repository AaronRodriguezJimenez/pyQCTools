# small helpers that operate on PySCF objects
def set_basis_and_build(mol, basis):
    """
    Replace molecule basis and rebuild in-place.
    Expects a PySCF gto.Mole object.
    """
    mol.basis = basis
    mol.build()
    return mol

def compute_scf_energy(mf):
    """
    Run SCF and return energy. Accepts PySCF SCF object (e.g. scf.RHF).
    """
    return mf.kernel()
