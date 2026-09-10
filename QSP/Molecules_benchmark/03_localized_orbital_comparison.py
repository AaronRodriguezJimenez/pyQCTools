#!/usr/bin/env python3
"""Compare canonical RHF orbitals with localized orbital representations.

Input
-----
The script consumes the ``scf.chk`` files written by
``02_rhf_orbital_audit.py``.  No new SCF optimization is required.

Default comparison
------------------
For each molecule it takes the *occupied valence space* (conventional core
orbitals omitted), and constructs

1. canonical occupied valence orbitals;
2. Pipek-Mezey (PM) localized orbitals using meta-Lowdin atomic populations;
3. Foster-Boys localized orbitals.

This is useful for active-space inspection:

* PM often gives chemically recognizable bonds/lone pairs and, importantly for
  the present benchmark, tends to preserve sigma/pi separation.
* Boys minimizes spatial spread and provides an independent localization view.
* The canonical orbitals retain the direct orbital-energy and symmetry picture.

The localized orbitals span exactly the same subspace as the input orbitals; only
unitary rotations within that subspace are performed.  Their printed
"fock_diag_Eh" values are diagonal Fock expectation values, *not canonical MO
energies*.

Virtual localization
--------------------
By default virtual orbitals are NOT localized.  In an aug-cc-pVTZ basis the low
virtual manifold can contain a mixture of valence antibonding and diffuse
Rydberg orbitals; indiscriminate localization can hide precisely the distinction
we want to inspect.  Use ``--nvirt-localize N`` deliberately to compare a chosen
low-virtual subspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyscf import scf
from pyscf.lo import boys, pipek
from pyscf.scf import chkfile as scf_chkfile
from pyscf.tools import molden


FROZEN_CORE_ORBITALS = {
    "H": 0,
    "C": 1,
    "N": 1,
    "O": 1,
    "S": 5,  # conventional [Ne] core for sulfur
}


def conventional_ncore(mol) -> int:
    return sum(FROZEN_CORE_ORBITALS.get(mol.atom_pure_symbol(i), 0) for i in range(mol.natm))


def parse_selection(text: str) -> set[int] | None:
    if text.strip().lower() == "all":
        return None
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            if b < a:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def checkpoint_dirs(audit_dir: Path, selected: set[int] | None) -> List[Path]:
    found = []
    for chk in sorted(audit_dir.glob("*/scf.chk")):
        m = re.match(r"^(\d+)_", chk.parent.name)
        if not m:
            continue
        idx = int(m.group(1))
        if selected is None or idx in selected:
            found.append(chk.parent)
    if not found:
        raise FileNotFoundError(f"No selected */scf.chk files found below {audit_dir}")
    return found


def reconstruct_rhf(chkfile: Path):
    mol, data = scf_chkfile.load_scf(str(chkfile))
    mf = scf.RHF(mol)
    mf.e_tot = float(data["e_tot"])
    mf.mo_energy = np.asarray(data["mo_energy"])
    mf.mo_coeff = np.asarray(data["mo_coeff"])
    mf.mo_occ = np.asarray(data["mo_occ"])
    mf.converged = True
    mf.chkfile = str(chkfile)
    return mol, mf


def spatial_moments(mol, coeff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = mol.intor_symmetric("int1e_r", comp=3)
    r2 = mol.intor_symmetric("int1e_r2")
    centres = np.einsum("pi,xpq,qi->ix", coeff.conj(), r, coeff).real
    mean_r2 = np.einsum("pi,pq,qi->i", coeff.conj(), r2, coeff).real
    spreads = mean_r2 - np.einsum("ix,ix->i", centres, centres)
    return centres, np.maximum(spreads, 0.0)


def atom_populations(mol, coeff: np.ndarray, method: str = "meta_lowdin") -> np.ndarray:
    # Shape [atom, orbital, orbital].  Diagonal elements are orbital populations.
    p = pipek.atomic_pops(mol, coeff, method=method)
    diag = np.stack([np.diag(p[a]).real for a in range(mol.natm)], axis=0)
    # Normalize small numerical population-scheme deviations for comparison metrics.
    denom = diag.sum(axis=0)
    denom[np.abs(denom) < 1e-12] = 1.0
    return diag / denom


def top_atoms(mol, pop: np.ndarray, i: int, n: int = 3) -> str:
    order = np.argsort(pop[:, i])[::-1][:n]
    return ";".join(
        f"{a + 1}:{mol.atom_symbol(a)}={pop[a, i]:.3f}" for a in order if pop[a, i] > 1e-4
    )


def effective_atoms(pop: np.ndarray, i: int) -> float:
    q = pop[:, i]
    denom = float(np.sum(q * q))
    return 1.0 / denom if denom > 1e-14 else float("inf")


def canonical_mixing_text(
    s: np.ndarray,
    canonical: np.ndarray,
    localized: np.ndarray,
    canonical_mo_numbers: Sequence[int],
    iloc: int,
    n: int = 4,
) -> str:
    overlaps = canonical.T.conj() @ s @ localized[:, iloc]
    weights = np.abs(overlaps) ** 2
    order = np.argsort(weights)[::-1][:n]
    return ";".join(
        f"MO{canonical_mo_numbers[j]}={weights[j]:.3f}" for j in order if weights[j] > 1e-4
    )


def localize_pm(mol, mf, coeff: np.ndarray, pop_method: str, max_cycle: int) -> np.ndarray:
    obj = pipek.PM(mol, coeff, mf=mf, pop_method=pop_method)
    obj.max_cycle = max_cycle
    obj.init_guess = None
    return np.asarray(obj.kernel(coeff))


def localize_boys(mol, coeff: np.ndarray, max_cycle: int) -> np.ndarray:
    obj = boys.Boys(mol, coeff)
    obj.max_cycle = max_cycle
    obj.init_guess = None
    return np.asarray(obj.kernel(coeff))


def make_rows(
    mol,
    method: str,
    coeff: np.ndarray,
    fock_ao: np.ndarray,
    s: np.ndarray,
    canonical_ref: np.ndarray,
    canonical_numbers: Sequence[int],
    pop_method: str,
) -> List[Dict[str, object]]:
    centres, spreads = spatial_moments(mol, coeff)
    pop = atom_populations(mol, coeff, method=pop_method)
    fsub = coeff.T.conj() @ fock_ao @ coeff

    rows = []
    for i in range(coeff.shape[1]):
        rows.append(
            {
                "method": method,
                "orbital_in_subspace_1based": i + 1,
                "fock_diag_Eh": float(fsub[i, i].real),
                "spread_bohr2": float(spreads[i]),
                "center_x_bohr": float(centres[i, 0]),
                "center_y_bohr": float(centres[i, 1]),
                "center_z_bohr": float(centres[i, 2]),
                "max_atom_population": float(np.max(pop[:, i])),
                "effective_number_of_atoms": float(effective_atoms(pop, i)),
                "top_atoms": top_atoms(mol, pop, i),
                "canonical_MO_mixing": canonical_mixing_text(
                    s, canonical_ref, coeff, canonical_numbers, i
                ),
            }
        )
    return rows


def subspace_summary(method: str, coeff: np.ndarray, mol, fock_ao: np.ndarray) -> Dict[str, object]:
    _, spreads = spatial_moments(mol, coeff)
    fsub = coeff.T.conj() @ fock_ao @ coeff
    off = fsub - np.diag(np.diag(fsub))
    return {
        "method": method,
        "norb": int(coeff.shape[1]),
        "sum_spread_bohr2": float(np.sum(spreads)),
        "mean_spread_bohr2": float(np.mean(spreads)),
        "fock_offdiag_frobenius_Eh": float(np.linalg.norm(off)),
    }


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def dump_partial_molden(mol, path: Path, coeff: np.ndarray, fock_diag: np.ndarray) -> None:
    # Occupancies are visualization metadata only for these partial orbital sets.
    occ = np.zeros(coeff.shape[1])
    molden.from_mo(mol, str(path), coeff, ene=fock_diag, occ=occ)


def process_space(
    mol,
    mf,
    outdir: Path,
    label: str,
    canonical: np.ndarray,
    canonical_numbers: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    s = mf.get_ovlp()
    dm = mf.make_rdm1()
    fock_ao = mf.get_fock(dm=dm)

    pm = localize_pm(mol, mf, canonical, args.pm_population, args.max_cycle)
    fb = localize_boys(mol, canonical, args.max_cycle)

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for method, coeff in [("canonical", canonical), ("pipek_mezey", pm), ("boys", fb)]:
        all_rows.extend(
            make_rows(
                mol,
                method,
                coeff,
                fock_ao,
                s,
                canonical,
                canonical_numbers,
                args.pm_population,
            )
        )
        summaries.append(subspace_summary(method, coeff, mol, fock_ao))

        fdiag = np.diag(coeff.T.conj() @ fock_ao @ coeff).real
        dump_partial_molden(mol, outdir / f"{label}_{method}.molden", coeff, fdiag)
        np.save(outdir / f"{label}_{method}_coeff.npy", coeff)

    return all_rows, summaries


def process_checkpoint(indir: Path, args: argparse.Namespace) -> Dict[str, object]:
    mol, mf = reconstruct_rhf(indir / "scf.chk")
    outdir = indir / "localized"
    outdir.mkdir(exist_ok=True)

    occ_idx = np.where(mf.mo_occ > 0)[0]
    vir_idx = np.where(mf.mo_occ == 0)[0]
    ncore = 0 if args.include_core else conventional_ncore(mol)
    if ncore >= len(occ_idx):
        raise ValueError(f"Frozen-core count {ncore} leaves no occupied valence orbitals")

    val_occ_idx = occ_idx[ncore:]
    c_occ = mf.mo_coeff[:, val_occ_idx]
    canonical_numbers_occ = [int(i + 1) for i in val_occ_idx]

    rows, summaries = process_space(
        mol,
        mf,
        outdir,
        "occupied_valence",
        c_occ,
        canonical_numbers_occ,
        args,
    )
    write_csv(rows, outdir / "occupied_valence_comparison.csv")

    result: Dict[str, object] = {
        "source": str(indir / "scf.chk"),
        "point_group": str(mol.groupname) if mol.symmetry else "C1/no symmetry",
        "ncore_omitted": int(ncore),
        "occupied_valence_MOs_1based": canonical_numbers_occ,
        "occupied_summaries": summaries,
    }

    if args.nvirt_localize > 0:
        n = min(args.nvirt_localize, len(vir_idx))
        chosen = vir_idx[:n]
        cvir = mf.mo_coeff[:, chosen]
        canonical_numbers_vir = [int(i + 1) for i in chosen]
        vrows, vsummaries = process_space(
            mol,
            mf,
            outdir,
            "low_virtual",
            cvir,
            canonical_numbers_vir,
            args,
        )
        write_csv(vrows, outdir / "low_virtual_comparison.csv")
        result["low_virtual_MOs_1based"] = canonical_numbers_vir
        result["low_virtual_summaries"] = vsummaries
        result["virtual_localization_warning"] = (
            "Low virtuals in an augmented basis may mix valence antibonding and Rydberg character; "
            "interpret localized virtuals as a subspace comparison, not as uniquely defined orbitals."
        )

    (outdir / "localization_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Canonical vs PM/Boys orbital comparison")
    p.add_argument("audit_dir", type=Path, help="Directory written by 02_rhf_orbital_audit.py")
    p.add_argument("--select", default="all", help="Molecule indices: all, 1, 1,3-6, ...")
    p.add_argument(
        "--include-core",
        action="store_true",
        help="Localize core occupied orbitals together with valence orbitals (default: omit conventional core)",
    )
    p.add_argument(
        "--pm-population",
        default="meta_lowdin",
        choices=["meta_lowdin", "mulliken", "lowdin", "iao", "becke"],
        help="Population definition used by Pipek-Mezey and comparison metrics",
    )
    p.add_argument(
        "--nvirt-localize",
        type=int,
        default=0,
        help="Also localize the N lowest canonical virtual orbitals (default: 0)",
    )
    p.add_argument("--max-cycle", type=int, default=200, help="Maximum localization macro cycles")
    args = p.parse_args()

    selected = parse_selection(args.select)
    dirs = checkpoint_dirs(args.audit_dir, selected)

    batch = []
    for d in dirs:
        print(f"\n=== {d.name} ===", flush=True)
        try:
            result = process_checkpoint(d, args)
            batch.append({"directory": d.name, "status": "OK", **result})
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            batch.append(
                {
                    "directory": d.name,
                    "status": f"ERROR:{type(exc).__name__}",
                    "error": str(exc),
                }
            )

    out = args.audit_dir / "localization_batch_summary.json"
    out.write_text(json.dumps(batch, indent=2) + "\n")
    print(f"\nWrote: {out.resolve()}")


if __name__ == "__main__":
    main()
