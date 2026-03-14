"""
  Implementation based on the paper:: Phys. Chem. Lett. 2015, 6, 1982−1988
  "Computational Quantum Chemistry for Multiple-Site Heisenberg Spin Couplings Made Simple:
  Only one spin-flip required"

  Steps involved:
  (1) Spin-Flip + Projection
  (2) Block Diagonalization
  (3) Effective Hamiltonian
"""
# heis_nsite_sfcis.py
# Translation of Octave heis_nsite_sfcis.m -> Python (NumPy/SciPy)
# Assumes numpy, scipy installed.
import numpy as np
import scipy.linalg as la
import scipy.io as sio
import os
import sys

def try_load_var(path_without_ext):
    """
    Try to load variables from either a .mat file (preferred) or a text file.
    The Octave script uses `load(filename)` where filename may be e.g. './heis_map_vecs.m'.
    In MATLAB/Octave, these can be .mat files or ascii files. Adjust as needed.
    Returns the dictionary from scipy.io.loadmat or a numpy array if a plain data file.
    """
    # prefer .mat
    matfile = None
    for ext in ('.mat', '.npy', '.txt', '.dat'):
        candidate = path_without_ext + ext
        if os.path.exists(candidate):
            if ext == '.mat':
                return sio.loadmat(candidate)
            elif ext == '.npy':
                return np.load(candidate, allow_pickle=True)
            else:
                try:
                    return np.loadtxt(candidate)
                except Exception:
                    # try whitespace-delimited fallback
                    return np.genfromtxt(candidate)
    # Try the literal name (Octave sometimes has file named .m that sets variables)
    mfile = path_without_ext + '.m'
    if os.path.exists(mfile):
        # If it's a tiny ascii file with variable assignment, read and exec safely:
        # WARNING: executing unknown code is dangerous. Here we try a minimal parse.
        # Better: convert .m files to .mat with Octave or provide .mat.
        raise RuntimeError(f"Found {mfile}. Please convert .m variable-files to .mat or .npy and re-run.")
    raise FileNotFoundError(f"No file found for {path_without_ext} (checked .mat, .npy, .txt, .dat, .m)")

def extract_variable(loaded, varname):
    """
    Given what scipy.io.loadmat returned or a numpy array, extract the variable.
    If loaded is a numpy array (ndarray) just return it.
    """
    if isinstance(loaded, np.ndarray):
        return loaded
    if isinstance(loaded, dict):
        # scipy.io.loadmat returns { '__header__', '__version__', '__globals__', 'varname': value }
        if varname in loaded:
            return loaded[varname]
        # sometimes MATLAB files have variable names slightly different; attempt heuristics:
        # If there's only one non-meta key, return it.
        keys = [k for k in loaded.keys() if not k.startswith("__")]
        if len(keys) == 1:
            return loaded[keys[0]]
        # else raise
        raise KeyError(f"Variable '{varname}' not found in loaded file. Keys: {keys}")
    raise TypeError("Unsupported loaded type: " + str(type(loaded)))

def main(dir_name='./'):
    # -- Load inputs (attempt to follow original Octave script naming) --
    def load_var(name):
        base = os.path.join(dir_name, name.replace('.m',''))
        loaded = try_load_var(base)
        return extract_variable(loaded, name.replace('.m',''))

    # Files expected (original Octave script used load on these file names)
    # We'll attempt variants: heis_map_vecs, heis_map_energies, heis_map_sort_ind, heis_map_proj_weights
    # The extracted variable names are assumed to match these base names.
    base_names = ['heis_map_vecs', 'heis_map_energies', 'heis_map_sort_ind', 'heis_map_proj_weights']
    data = {}
    for bn in base_names:
        try:
            val = load_var(bn)
        except Exception as e:
            print(f"Error loading '{bn}': {e}")
            print("Please ensure data are in .mat (preferred) or .npy/.txt formats and variable names match.")
            raise

        data[bn] = np.array(val, dtype=float, copy=False)

    vectors = data['heis_map_vecs']      # shape (Ndet, Nstates_total)
    energies = data['heis_map_energies'].squeeze()  # shape (Nstates_total,)
    sort_ind = data['heis_map_sort_ind'].squeeze().astype(int)  # 0-based or 1-based? handle below
    proj_weights = data['heis_map_proj_weights'].squeeze()

    # If sort_ind appears 1-based (contains values >=1 and min >=1), convert to 0-based.
    if sort_ind.min() >= 1:
        sort_ind = sort_ind - 1

    # Units & constants
    au2ev = 27.21165
    au2cm = 219474.63
    units = "cm-1"
    convert = au2cm

    # Count size of blocks based on sort_ind (how many determinants per site index)
    tmp1 = int(np.max(sort_ind)) + 1
    count_vec = np.zeros(tmp1, dtype=int)
    for i in range(len(sort_ind)):
        ii = sort_ind[i]
        count_vec[ii] += 1

    blocks = [int(c) for c in count_vec if c > 0]
    print("blocks =", blocks)

    # Should each orbital be counted as a separate site? (original flag)
    sing_orb_site = False
    if sing_orb_site:
        blocks = [1] * len(sort_ind)

    proj_weights = proj_weights
    dim_heis = int(np.sum(blocks))
    nstates = int(len(blocks))   # number of sites

    # optional sorting by projection (turned off in original)
    sort_by_proj = False
    if sort_by_proj:
        permutation = np.argsort(proj_weights)  # ascending
        permutation = permutation[::-1]         # flip
        energies = energies[permutation]
        vectors = vectors[:, permutation]
        for s in range(nstates):
            print(f" state: {permutation[s]:5d}  site: {sort_ind[s]:5d}   proj: {proj_weights[s]:.6f}")
        print("Sort vectors from high to low projections")
        print(permutation[:nstates])
        print(proj_weights[:nstates])

    # Use original ordering
    energies = energies
    vectors = vectors

    print("\nState energies")
    # shift energies for convenience (match Octave logic)
    e = energies[:nstates].copy()
    e = e - np.min(e)
    e = e - np.max(e) - 1.0
    for i in range(nstates):
        print(f"      State {i+1:4d}: {e[i]:12.8f}")

    # Projected eigenvectors: select first nstates vectors
    print("\nProjected eigenvectors")
    c = vectors[:, :nstates].copy()

    # sort determinant indices: Octave did [~, permutation] = sort(sort_ind)
    permutation = np.argsort(sort_ind)
    c = c[permutation, :]

    print("\nNorms of projected eigenvectors (Projection onto neutral determinant basis)")
    for s in range(nstates):
        print(f"      State {s+1:4d}: {np.linalg.norm(c[:, s]):12.8f}")

    print("\nOverlap of projected eigenvectors")
    S = c.T @ c    # shape (nstates, nstates)
    print(S)

    # eigen-decompose S
    # Use eigh since S is symmetric (real)
    w, U = np.linalg.eigh(S)
    # clip small negative eigenvalues due to numerical noise
    w[w < 0] = np.maximum(w[w < 0], 0.0)
    # build X = U * l^(-1/2) * U'
    inv_sqrt_l = np.diag(1.0 / np.sqrt(np.where(w > 0, w, np.finfo(float).eps)))
    X = U @ inv_sqrt_l @ U.T
    sqrt_l = np.diag(np.sqrt(np.where(w > 0, w, np.finfo(float).eps)))
    Xinv = U @ sqrt_l @ U.T

    C = c @ X

    print("\nOrthogonalized eigenvectors")
    print(C)

    print("\nEffective Hamiltonian")
    Heff = C @ np.diag(e) @ C.T
    print(Heff)

    # ensure symmetry numerically
    Heff = 0.5 * (Heff + Heff.T)
    # diagonalize Heff
    # Heff should be symmetric; use eigh
    e_eff_vals, Ceff = np.linalg.eigh(Heff)
    # sort ascending
    order = np.argsort(e_eff_vals)
    e_eff_vals = e_eff_vals[order]
    Ceff = Ceff[:, order]

    print("\nState Energies:")
    print(f"   {'State':>10s}: {'Ab Initio':>20s} {'Effective Ham':>20s} (units)")
    for i in range(nstates):
        print(f"   {i+1:10d}: {e[i]:20.8f} {e_eff_vals[i]:20.8f} (au)")
    print("")

    # Diagonalize on-site blocks of effective Hamiltonian
    Uloc = np.eye(dim_heis)
    first = 0
    # iterate through blocks
    for b in blocks:
        last = first + b  # python slice end
        i = first
        j = last - 1
        ht = Heff[i:last, i:last]
        # compute eigen-decomposition of ht
        et_vals, ct = np.linalg.eigh(ht)
        # sort ascending
        order = np.argsort(et_vals)
        et_vals = et_vals[order]
        ct = ct[:, order]

        # local high-spin vector (uniform)
        hs = (1.0 / np.sqrt(b)) * np.ones((b, 1))
        hs = ct @ hs
        # determine sign such that max(hs) > -min(hs) => sign +1, else -1
        mh = float(np.max(hs))
        mn = float(np.min(hs))
        if mh > -mn:
            sign_hs = 1.0
        elif mh < -mn:
            sign_hs = -1.0
        else:
            raise RuntimeError("Error finding sign_hs for block starting at index {}".format(first))

        Uloc[i:last, i:last] = sign_hs * ct
        first = last

    print("Transformation matrix which rotates the Heff into the local eigenstate basis")
    print(Uloc)

    Heff = Uloc.T @ Heff @ Uloc

    print("\nHeff in block diagonal (local eigenstate) basis")
    print(Heff)

    print("\nFor this second projection to work, each block should have only 1 nonzero diagonal element.")
    print("Diagonals of Heff:")
    for i in range(dim_heis):
        print(f"  H({i+1},{i+1}) = {Heff[i,i]:20.8f}")

    # Symmetrize and diagonalize again
    Heff = 0.5 * (Heff + Heff.T)
    ee_vals, Ceff2 = np.linalg.eigh(Heff)
    order = np.argsort(ee_vals)
    ee_vals = ee_vals[order]
    Ceff2 = Ceff2[:, order]

    e_eff = ee_vals

    print("\nState Energies:")
    print(f"   {'State':>10s}: {'Ab Initio':>20s} {'Effective Ham':>20s} {'Relative':>20s}")
    for i in range(nstates):
        relative = (e[i] - e[0]) * au2ev
        print(f"   {i+1:10d}: {e[i]:20.8f} {e_eff[i]:20.8f} {relative:20.4f}")

    # Project to local groundstate basis
    P = np.zeros((dim_heis, nstates))
    Q = np.zeros((dim_heis, nstates))
    I = np.eye(dim_heis)
    q = 0
    for i in range(nstates):
        Q[:, i] = I[:, i]
        P[:, i] = I[:, q]
        q += blocks[i]

    # c = P'*Ceff*Q  (note: Octave uses Ceff from most recent diag; replicate with Ceff2 & ee_vals)
    c_proj = P.T @ (Ceff2 @ Q)
    e_proj = Q.T @ np.diag(ee_vals) @ Q

    print("\nProjected H Eigenvectors")
    print(c_proj)

    print("\nOverlap of projected eigenvectors")
    S2 = c_proj.T @ c_proj
    print(S2)

    # orthogonalize projected vectors
    w2, U2 = np.linalg.eigh(S2)
    w2[w2 < 0] = np.maximum(w2[w2 < 0], 0.0)
    inv_sqrt_l2 = np.diag(1.0 / np.sqrt(np.where(w2 > 0, w2, np.finfo(float).eps)))
    X2 = U2 @ inv_sqrt_l2 @ U2.T
    X2inv = U2 @ np.diag(np.sqrt(np.where(w2 > 0, w2, np.finfo(float).eps))) @ U2.T
    c_orth = c_proj @ X2

    print("\nOrthogonalized Projected H Eigenvectors")
    print(c_orth)

    Heff_final = c_orth @ e_proj @ c_orth.T
    print("\nFinal Effective Hamiltonian in local ground state basis:")
    print(Heff_final)

    print("\nExchange coupling constants:")
    print("   H = J * Sa * Sb")
    print("     or")
    print("   H = -2J' * Sa * Sb")
    for i in range(nstates):
        for j in range(i+1, nstates):
            Si = blocks[i] / 2.0
            Sj = blocks[j] / 2.0
            J = Heff_final[i, j] / np.sqrt(Si * Sj)
            print(f" J({i+1:2d},{j+1:2d}) = {J*convert:20.6f} : J'({i+1:2d},{j+1:2d}) = {J*(-0.5)*convert:20.6f} ({units})")

    # Norms: un-orthogonalize and print norms
    c_un = c_orth @ X2inv
    print("\nNorms")
    for i in range(nstates):
        print(f"  Vector {i+1:4d}: {np.linalg.norm(c_un[:, i]):8.4f}")

if __name__ == '__main__':
    # allow passing directory as argv
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        d = './ni_cubane_data_sfcas'
    main(d)
