"""
make_cas_integrals.py

Read geometry from input.xyz, run a cheap SCF, and produce one- and two-electron
integrals in an *arbitrary* MO basis corresponding to an active space *without*
performing CASSCF.

Outputs:
 - one-electron integrals h_pq (AO core-H transformed to chosen MOs) saved as .npy
 - two-electron integrals (pq|rs) in MO basis saved as .npy (or as FCIDUMP)

Usage example:
python make_cas_integrals.py --xyz input.xyz --basis cc-pVDZ \
  --nact 10 --ncas 10 --hcore-out h_cas.npy --eri-out eri_cas.npy --fcidump FCIDUMP
Requires: pyscf, numpy
"""

import argparse
import numpy as np
from pyscf import gto, scf, ao2mo, lib
from pyscf.tools import fcidump
from pyscf.lo import boys
import os, sys

def read_xyz(xyzfile, charge=0, spin=0):
    # PySCF can read xyz lines: each line is "atom x y z" after header
    with open(xyzfile) as f:
        lines = f.read().strip().splitlines()
    # if first line is number of atoms, skip it
    if len(lines) >= 2 and lines[0].strip().isdigit():
        atom_lines = lines[2:]  # skip natom + comment
    else:
        atom_lines = lines
    atoms = []
    for L in atom_lines:
        tok = L.split()
        if len(tok) < 4: continue
        atoms.append((tok[0], (float(tok[1]), float(tok[2]), float(tok[3]))))
    return atoms

def run_scf(mol, method='rhf', conv=1e-8, max_cycle=200, df=False):
    if method.lower() == 'rhf':
        mf = scf.RHF(mol)
    elif method.lower() == 'uhf':
        mf = scf.UHF(mol)
    elif method.lower() == 'rohf':
        mf = scf.ROHF(mol)
    else:
        raise ValueError("Unsupported method: " + method)
    mf.max_cycle = max_cycle
    mf.conv_tol = conv
    if df:
        # density fitting to accelerate integrals if desired
        mf = mf.density_fit()
    mf.verbose = 1
    mf.kernel()
    if not mf.converged:
        print("Warning: SCF did not converge. Results may be poor.", file=sys.stderr)
    return mf

def mo_localize(mol, mo_coeff, method='boys'):
    if method == 'boys':
        return boys.orth_ao_loc(mol, mo_coeff)  # returns localized MO coeffs
    else:
        raise NotImplementedError("Only 'boys' localization implemented in this helper.")

def transform_one_electron(hcore_ao, mo_coeff_active):
    # h_pq = C^T hcore C  (but we want *only* the active-active block)
    return mo_coeff_active.T.dot(hcore_ao.dot(mo_coeff_active))

def transform_two_electron_by_ao2mo(mol, mo_coeff_active, compact=False):
    """
    Use pyscf.ao2mo to transform AO integrals to MO basis for the given mo_coeff_active.
    Returns full 4-index eri array with shape (nact,nact,nact,nact) in physicist (pq|rs) notation.
    Note: memory heavy for large nact.
    """
    # ao2mo.kernel returns packed integrals suitable for ao2mo.restore
    eri_packed = ao2mo.kernel(mol, mo_coeff_active)
    eri_full = ao2mo.restore(1, eri_packed, mo_coeff_active.shape[1])
    # eri_full is shape (nact*nact, nact*nact) if restore returns 2-index packed? But restore(1, ) returns full 4-index
    # Ensure we have shape (nact,nact,nact,nact)
    n = mo_coeff_active.shape[1]
    eri_full = eri_full.reshape(n, n, n, n)
    return eri_full

def transform_two_electron_blockwise(mol, mo_coeff_active, blocksize=32):
    """
    Memory-efficient AO -> MO block transform. This avoids allocating the full (nact^4) intermediate.
    Returns full 4-index eri in shape (nact,nact,nact,nact).
    This is slower but uses less peak memory.
    """
    nact = mo_coeff_active.shape[1]
    # Precompute AO integrals on-the-fly using intor('int2e'), but that returns full AO 4-index -> too big.
    # Instead use ao2mo.partial or transform in two steps: (ij|kl) = sum_abcd C_ia C_jb C_kc C_ld (ab|cd)
    # We'll build intermediate Q = sum_ab C_kc C_ld (ab|cd) in blocks using ao2mo.general
    # Simpler approach: use ao2mo.incore.full but in blocks by picking AO->MO transform subsets.
    # We'll use ao2mo.general to transform AO integrals into MO basis in stages.
    # Implementation below transforms integrals with outer products in blocks of two MO indices.
    import math
    eri_mo = np.zeros((nact, nact, nact, nact))
    nao = mol.nao_nr()
    # transform AO->MO in two-step: (pq|rs) = \sum_{ab} C_pa C_qb (ab|rs)_AO_mapped
    # Use ao2mo.general which accepts mo_coeff list; we can pass shapes to transform only needed columns.
    # But pyscf has no direct builtin very-low memory path here; we emulate via ao2mo.incore.general with blocks
    # We'll call ao2mo.general for blocks of active orbitals to create intermediates.
    for i0 in range(0, nact, blocksize):
        i1 = min(nact, i0+blocksize)
        Ci = mo_coeff_active[:, i0:i1]   # shape (nao, bix)
        for j0 in range(0, nact, blocksize):
            j1 = min(nact, j0+blocksize)
            Cj = mo_coeff_active[:, j0:j1]
            # transform first two indices (p,q). ao2mo.general supports (mo_coeff_list, (nocc, nvir)) but we can pass a tuple:
            # We'll transform AO integrals to (pq|rs) packed for these pq blocks, then restore and finish transforming r,s as well
            # The simplest technique: use ao2mo.general with mo_coeff = [Ci, Cj, mo_coeff_active, mo_coeff_active]
            eri_ij_rs_packed = ao2mo.general(mol._eri, (Ci, Cj, mo_coeff_active, mo_coeff_active), compact=False)
            # Now eri_ij_rs_packed has shape (bix, bjx, nact, nact) if compact=False
            eri_mo[i0:i1, j0:j1, :, :] = eri_ij_rs_packed
    return eri_mo

def save_integrals(h1e, eri, h_out='h_cas.npy', eri_out='eri_cas.npy', fcidump_file=None, norb=None, nelec=None):
    np.save(h_out, h1e)
    # Be careful with size — warn user
    print(f"Saved one-electron integrals to {h_out} (shape {h1e.shape})")
    # Save eri as numpy. If very big, consider writing in chunks or using memory-mapped arrays
    np.save(eri_out, eri)
    print(f"Saved two-electron integrals to {eri_out} (shape {eri.shape})")
    if fcidump_file:
        # Write FCIDUMP: fcidump.from_integrals expects 1e integrals as 2D (norb,norb) and eri as 4-index physicist (norb,norb,norb,norb)
        if norb is None or nelec is None:
            raise ValueError("To write FCIDUMP please specify norb and nelec")
        print(f"Writing FCIDUMP file to {fcidump_file} ...")
        fcidump.from_integrals(fcidump_file, h1e, eri, norb=norb, nelec=nelec)
        print("FCIDUMP written.")

def parse_act_range(s):
    if ":" in s:
        a,b = s.split(":")
        start = int(a)
        end = int(b)
        return list(range(start, end))
    else:
        # single int = nact
        return int(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz', required=True, help='input xyz file')
    parser.add_argument('--basis', default='cc-pVDZ', help='AO basis')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--spin', type=int, default=0, help='2S (multiplicity-1). e.g. spin=0 for closed-shell')
    parser.add_argument('--method', default='rhf', choices=['rhf','uhf','rohf'])
    parser.add_argument('--nact', type=int, default=None, help='Number of active orbitals (counts, selecting around HOMO/LUMO)')
    parser.add_argument('--act-range', type=str, default=None, help='Explicit active MO slice, e.g. 10:20 (0-based python indices)')
    parser.add_argument('--use-localize', action='store_true', help='Localize MOs (Boys) before selecting active space')
    parser.add_argument('--hcore-out', default='h_cas.npy')
    parser.add_argument('--eri-out', default='eri_cas.npy')
    parser.add_argument('--fcidump', default=None, help='Optional: write FCIDUMP file')
    parser.add_argument('--direct-block', action='store_true', help='Use blockwise AO->MO transform (slower but lower peak memory)')
    args = parser.parse_args()

    atoms = read_xyz(args.xyz, charge=args.charge, spin=args.spin)
    mol = gto.M(
        atom=atoms,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        verbose=4,
        output='pyscf_output.out'
    )
    mol.build()

    mf = run_scf(mol, method=args.method)

    # Core AO 1-electron h (kinetic + nuclear attraction)
    hcore_ao = mf.get_hcore()

    mo_coeff = mf.mo_coeff
    if args.method.lower() == 'uhf':
        # UHF mo_coeff is tuple (moA, moB)
        # For simplicity handle closed-shell or RHF; UHF handling would need per-spin transforms
        raise NotImplementedError("This script currently handles only RHF/ROHF canonical MO sets for single MO coeff matrix.")
    # Determine active orbital indices
    nmo = mo_coeff.shape[1]
    if args.act_range is not None:
        act_idx = list(range(*map(int, args.act_range.split(":"))))
    elif args.nact is not None:
        # pick centered on HOMO (assume closed-shell): HOMO = nocc-1
        nocc = mol.nelectron // 2
        homo = nocc - 1
        nact = args.nact
        start = max(0, homo - nact//2 + 1)
        end = min(nmo, start + nact)
        act_idx = list(range(start, end))
        if len(act_idx) != nact:
            # adjust if at edges
            start = max(0, end - nact)
            act_idx = list(range(start, end))
    else:
        raise ValueError("Specify either --nact or --act-range to select the active orbitals")

    print(f"Selected active MO indices (0-based): {act_idx}")

    # Optionally localize orbitals before slicing
    if args.use_localize:
        print("Localizing orbitals (Boys) ...")
        mo_coeff = mo_localize(mol, mo_coeff, method='boys')

    mo_active = mo_coeff[:, act_idx]    # shape (nao, nact)

    # one-electron integrals in active MO basis
    h1e_mo = transform_one_electron(hcore_ao, mo_active)

    # two-electron integrals
    nact = mo_active.shape[1]
    print(f"Transforming two-electron integrals to active MO basis (nact = {nact}) ...")
    # for moderate nact, direct ao2mo is simplest:
    if not args.direct_block:
        eri_mo = transform_two_electron_by_ao2mo(mol, mo_active)  # shape (nact,nact,nact,nact)
    else:
        eri_mo = transform_two_electron_blockwise(mol, mo_active, blocksize=16)

    # Save
    nelec = mol.nelectron
    save_integrals(h1e_mo, eri_mo, h_out=args.hcore_out, eri_out=args.eri_out,
                   fcidump_file=args.fcidump, norb=nact, nelec=nelec)

if __name__ == '__main__':
    main()