def set_basis_and_build(mol, basis):
    mol.basis = basis
    mol.build()
    return mol
def compute_scf_energy(mf):
    return mf.kernel()
