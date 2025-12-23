import numpy as np

def build_hubbard_2d(Nx, Ny, t, U, periodic_y=False):
    n_sites = Nx * Ny
    h1 = np.zeros((n_sites, n_sites))
    h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
    # Standard Row-Major Indexing: idx = i * Ny + j
    # get_idx = lambda i, j: i * Ny + j
    def get_idx(i, j):
        if j % 2 == 0:  # even row
            return j * Nx + i
        else:           # odd row
            return j * Nx + (Nx - i - 1)
    for i in range(Nx):
        for j in range(Ny):
            curr = get_idx(i, j)
            # 1. Horizontal Hopping (X-direction, Open Boundaries)
            if i + 1 < Nx:
                right = get_idx(i + 1, j)
                h1[curr, right] = h1[right, curr] = -t
            # 2. Vertical Hopping (Y-direction)
            if j + 1 < Ny:
                up = get_idx(i, j + 1)
                h1[curr, up] = h1[up, curr] = -t
            elif periodic_y and Ny > 2:
                # Periodic wrap-around in Y
                wrap = get_idx(i, 0)
                h1[curr, wrap] = h1[wrap, curr] = -t
    # 3. On-site interaction (U)
    for i in range(n_sites):
        h2[i, i, i, i] = U

    print("Hubbard Hamiltonian")
    #Nx = 4
    #Ny = 4
    #t = 1.0
    #U = 1.0
    #h1, h2 = build_hubbard_2d(Nx, Ny, t, U, periodic_y=False)

    from pyscf import gto, scf, ao2mo

    # Make HF
    mol = gto.M()
    n = 16
    mol.nelectron = 12

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(1, h2, n)

    mf.kernel()

    print("HF energy:", mf.e_tot)

    import openfermion as of
    import pyqctools as pq

    mfEris = mf._eri
    mfhcore = mf.get_hcore()
    MOs = mf.mo_coeff

    H1, H2 = of.ops.representations.get_tensors_from_integrals(mfhcore, mfEris)
    return H1, H2



