#!/usr/bin/env python3
"""RHF orbital audit for the 35-molecule vertical-excitation benchmark.

Purpose
-------
This is the *second* stage after the basis/electron-count inspection.  It runs a
closed-shell RHF calculation for selected molecules and records diagnostics that
are useful before attempting CASSCF/NEVPT2/ADC/EOM-CC calculations:

* SCF convergence and total energy
* AO-overlap conditioning (important with aug-cc-pVTZ)
* number of occupied/virtual orbitals
* HOMO/LUMO energies and gap
* orbital symmetry labels (when symmetry is enabled)
* concise meta-Lowdin AO/atomic character for every canonical MO
* orbital centres and spatial spreads <r^2>-|<r>|^2 (useful for spotting
  unusually diffuse/Rydberg-like virtuals)
* RHF internal stability and RHF -> UHF external stability, allowing point-group
  symmetry breaking in the stability search
* optional follow-up UHF calculation when the RHF solution is externally
  unstable

For each molecule, the script writes a PySCF checkpoint and a Molden file.  The
checkpoint is the intended input for ``03_localized_orbital_comparison.py``.

Coordinates are assumed to have the numbered-block format of the supplied
``Coordinates`` file and are in Bohr by default.

Notes on stability
------------------
``pyscf.scf.stability.rhf_external`` also evaluates a real->complex channel in
its log, but the boolean returned by the current PySCF API is the RHF->UHF
stability status.  This script therefore labels the returned flag explicitly as
``rhf_to_uhf_stable``.  Inspect ``scf.log`` if the real->complex diagnostic is
important for a particular system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pyscf import gto, lo, scf, symm
from pyscf.scf import stability as scf_stability
from pyscf.tools import molden


# -----------------------------------------------------------------------------
# Benchmark input helpers
# -----------------------------------------------------------------------------

SYSTEM_OVERRIDES = {
    "Streptocyanine-C1": {"charge": +1, "spin": 0},
}

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}

HEADER_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$"
)

Atom = Tuple[str, Tuple[float, float, float]]
MoleculeRecord = Tuple[int, str, List[Atom]]


def parse_coordinate_file(path: Path) -> List[MoleculeRecord]:
    records: List[MoleculeRecord] = []
    current_index = None
    current_name = None
    current_atoms: List[Atom] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        m = HEADER_RE.match(line)
        if m:
            if current_index is not None:
                if not current_atoms:
                    raise ValueError(f"Molecule {current_index} has no atoms")
                records.append((current_index, current_name, current_atoms))
            current_index = int(m.group(1))
            current_name = m.group(2).strip()
            current_atoms = []
            continue

        m = ATOM_RE.match(line)
        if m and current_index is not None:
            symbol = m.group(1)
            if symbol not in ATOMIC_NUMBER:
                raise ValueError(f"Unsupported element {symbol!r} at line {lineno}")
            xyz = tuple(float(m.group(i)) for i in (2, 3, 4))
            current_atoms.append((symbol, xyz))
            continue

        # Allow the descriptive preamble before the first numbered molecule.
        if current_index is None:
            continue
        raise ValueError(f"Could not parse line {lineno}: {raw!r}")

    if current_index is not None:
        if not current_atoms:
            raise ValueError(f"Molecule {current_index} has no atoms")
        records.append((current_index, current_name, current_atoms))

    if not records:
        raise ValueError(f"No molecule blocks found in {path}")
    return records


def parse_selection(text: str, records: Sequence[MoleculeRecord]) -> List[MoleculeRecord]:
    """Select molecule numbers, e.g. '1,3-5,12'.  'all' selects everything."""
    if text.strip().lower() == "all":
        return list(records)

    wanted = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            wanted.update(range(a, b + 1))
        else:
            wanted.add(int(part))

    by_index = {rec[0]: rec for rec in records}
    missing = sorted(wanted - set(by_index))
    if missing:
        raise ValueError(f"Unknown molecule indices: {missing}")
    return [by_index[i] for i in sorted(wanted)]


def slugify(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return out or "molecule"


def hill_formula(atoms: Sequence[Atom]) -> str:
    counts = Counter(sym for sym, _ in atoms)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(x for x in counts if x not in {"C", "H"})
    else:
        order = sorted(counts)
    return "".join(el if counts[el] == 1 else f"{el}{counts[el]}" for el in order)


def build_mol(
    atoms: Sequence[Atom],
    name: str,
    basis: str,
    unit: str,
    symmetry: bool,
    max_memory: int,
    log_file: Path,
    verbose: int,
) -> gto.Mole:
    meta = SYSTEM_OVERRIDES.get(name, {})
    charge = int(meta.get("charge", 0))
    spin = int(meta.get("spin", 0))

    nelec = sum(ATOMIC_NUMBER[sym] for sym, _ in atoms) - charge
    if spin == 0 and nelec % 2:
        raise ValueError(f"{name}: odd electron count {nelec} with spin=0")

    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = unit
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.cart = False
    mol.symmetry = symmetry
    mol.max_memory = max_memory
    mol.verbose = verbose
    mol.output = str(log_file)
    mol.build()
    return mol


# -----------------------------------------------------------------------------
# Orbital diagnostics
# -----------------------------------------------------------------------------


def orbital_symmetry_labels(mol: gto.Mole, coeff: np.ndarray) -> List[str]:
    if not mol.symmetry:
        return [""] * coeff.shape[1]
    try:
        labels = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, coeff, check=False
        )
        return [str(x) for x in labels]
    except Exception as exc:  # diagnostic only: do not abort an otherwise good SCF
        return [f"UNLABELED:{type(exc).__name__}"] * coeff.shape[1]


def meta_lowdin_weights(mol: gto.Mole, coeff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Return |MO coefficients|^2 in an orthonormal meta-Lowdin AO basis."""
    x = lo.orth.orth_ao(mol, method="meta_lowdin", s=s)
    c_orth = x.T.conj() @ s @ coeff
    weights = np.abs(c_orth) ** 2
    # Numerical normalization protects summaries from small metric noise.
    denom = weights.sum(axis=0)
    denom[denom == 0] = 1.0
    return weights / denom


def spatial_moments(mol: gto.Mole, coeff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return orbital centres (bohr) and spreads (bohr^2)."""
    r = mol.intor_symmetric("int1e_r", comp=3)
    r2 = mol.intor_symmetric("int1e_r2")
    centres = np.einsum("pi,xpq,qi->ix", coeff.conj(), r, coeff).real
    mean_r2 = np.einsum("pi,pq,qi->i", coeff.conj(), r2, coeff).real
    spreads = mean_r2 - np.einsum("ix,ix->i", centres, centres)
    spreads = np.maximum(spreads, 0.0)
    return centres, spreads


def ao_character_tables(
    mol: gto.Mole, weights: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    labels = mol.ao_labels(fmt=False)
    atom_ids = np.asarray([int(lbl[0]) for lbl in labels])
    nl = [str(lbl[2]) for lbl in labels]

    atom_pop = np.zeros((mol.natm, weights.shape[1]))
    for ia in range(mol.natm):
        atom_pop[ia] = weights[atom_ids == ia].sum(axis=0)

    angular: Dict[str, np.ndarray] = {}
    for letter in "spdfgh":
        mask = np.asarray([bool(re.search(letter + r"$", x)) for x in nl])
        if mask.any():
            angular[letter] = weights[mask].sum(axis=0)
        else:
            angular[letter] = np.zeros(weights.shape[1])
    return atom_pop, angular


def top_atom_text(mol: gto.Mole, atom_pop: np.ndarray, imo: int, n: int = 3) -> str:
    order = np.argsort(atom_pop[:, imo])[::-1][:n]
    parts = []
    for ia in order:
        p = atom_pop[ia, imo]
        if p < 1e-4:
            continue
        parts.append(f"{ia + 1}:{mol.atom_symbol(ia)}={p:.3f}")
    return ";".join(parts)


def relative_mo_label(imo: int, homo: int, lumo: int) -> str:
    if imo <= homo:
        d = homo - imo
        return "HOMO" if d == 0 else f"HOMO-{d}"
    d = imo - lumo
    return "LUMO" if d == 0 else f"LUMO+{d}"


def build_orbital_rows(mol: gto.Mole, mf: scf.hf.SCF) -> List[Dict[str, object]]:
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    ene = np.asarray(mf.mo_energy)
    s = mf.get_ovlp()

    occ_idx = np.where(occ > 0)[0]
    vir_idx = np.where(occ == 0)[0]
    homo = int(occ_idx[-1])
    lumo = int(vir_idx[0]) if len(vir_idx) else c.shape[1]

    irreps = orbital_symmetry_labels(mol, c)
    weights = meta_lowdin_weights(mol, c, s)
    atom_pop, angular = ao_character_tables(mol, weights)
    centres, spreads = spatial_moments(mol, c)

    rows = []
    for i in range(c.shape[1]):
        row: Dict[str, object] = {
            "mo_1based": i + 1,
            "frontier_label": relative_mo_label(i, homo, lumo),
            "occupation": float(occ[i]),
            "energy_Eh": float(ene[i]),
            "energy_eV": float(ene[i] * 27.211386245988),
            "irrep": irreps[i],
            "center_x_bohr": float(centres[i, 0]),
            "center_y_bohr": float(centres[i, 1]),
            "center_z_bohr": float(centres[i, 2]),
            "spread_bohr2": float(spreads[i]),
            "top_atoms_meta_lowdin": top_atom_text(mol, atom_pop, i),
        }
        for letter in "spdfgh":
            row[f"{letter}_character"] = float(angular[letter][i])
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    # Batch summaries can have molecule-specific fields (for example UHF
    # follow-up diagnostics only when an RHF instability is found).  Preserve
    # the order of first appearance while allowing such optional columns.
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frontier_subset(rows: Sequence[Dict[str, object]], n_each_side: int) -> List[Dict[str, object]]:
    occupied = [r for r in rows if float(r["occupation"]) > 0]
    virtual = [r for r in rows if float(r["occupation"]) == 0]
    return occupied[-n_each_side:] + virtual[:n_each_side]


# -----------------------------------------------------------------------------
# SCF and stability
# -----------------------------------------------------------------------------


def run_rhf(
    mol: gto.Mole,
    chkfile: Path,
    conv_tol: float,
    max_cycle: int,
    newton_fallback: bool,
) -> scf.hf.SCF:
    mf = scf.RHF(mol)
    mf.chkfile = str(chkfile)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.max_memory = mol.max_memory
    mf.kernel()

    if not mf.converged and newton_fallback:
        dm = mf.make_rdm1() if mf.mo_coeff is not None else None
        nmf = mf.newton()
        nmf.chkfile = str(chkfile)
        nmf.conv_tol = conv_tol
        nmf.max_cycle = max_cycle
        nmf.max_memory = mol.max_memory
        nmf.kernel(dm)
        mf = nmf
    return mf


def follow_external_instability(
    mol: gto.Mole,
    mo_pair: Tuple[np.ndarray, np.ndarray],
    outdir: Path,
    conv_tol: float,
    max_cycle: int,
) -> Dict[str, object]:
    nalpha, nbeta = mol.nelec
    ca, cb = mo_pair
    dm_a = ca[:, :nalpha] @ ca[:, :nalpha].T.conj()
    dm_b = cb[:, :nbeta] @ cb[:, :nbeta].T.conj()

    umf = scf.UHF(mol)
    umf.chkfile = str(outdir / "uhf_from_rhf_instability.chk")
    umf.conv_tol = conv_tol
    umf.max_cycle = max_cycle
    umf.max_memory = mol.max_memory
    e = umf.kernel((dm_a, dm_b))

    result: Dict[str, object] = {
        "uhf_followup_attempted": True,
        "uhf_converged": bool(umf.converged),
        "uhf_energy_Eh": float(e),
    }
    if umf.converged:
        ss, multiplicity = umf.spin_square()
        result["uhf_S2"] = float(ss)
        result["uhf_effective_multiplicity"] = float(multiplicity)
    return result


def audit_one(
    record: MoleculeRecord,
    args: argparse.Namespace,
) -> Dict[str, object]:
    index, name, atoms = record
    outdir = args.outdir / f"{index:02d}_{slugify(name)}"
    outdir.mkdir(parents=True, exist_ok=True)

    mol = build_mol(
        atoms,
        name,
        basis=args.basis,
        unit=args.unit,
        symmetry=args.symmetry,
        max_memory=args.max_memory,
        log_file=outdir / "scf.log",
        verbose=args.verbose,
    )

    # Linear-dependence diagnostic before SCF.
    s = mol.intor_symmetric("int1e_ovlp")
    seig = np.linalg.eigvalsh(s)
    smin = float(seig[0])
    smax = float(seig[-1])
    scond = float(smax / smin) if smin > 0 else math.inf

    mf = run_rhf(
        mol,
        chkfile=outdir / "scf.chk",
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        newton_fallback=args.newton_fallback,
    )

    summary: Dict[str, object] = {
        "index": index,
        "molecule": name,
        "formula": hill_formula(atoms),
        "charge": int(mol.charge),
        "spin_2S": int(mol.spin),
        "electrons": int(mol.nelectron),
        "basis": args.basis,
        "point_group": str(mol.groupname) if mol.symmetry else "C1/no symmetry",
        "nao_nmo": int(mol.nao_nr()),
        "nocc": int(np.count_nonzero(np.asarray(mf.mo_occ) > 0)) if mf.mo_occ is not None else None,
        "nvirt": int(np.count_nonzero(np.asarray(mf.mo_occ) == 0)) if mf.mo_occ is not None else None,
        "overlap_min_eigenvalue": smin,
        "overlap_condition_number": scond,
        "scf_converged": bool(mf.converged),
        "rhf_energy_Eh": float(mf.e_tot) if mf.e_tot is not None else None,
    }

    if not mf.converged:
        summary["status"] = "SCF_NOT_CONVERGED"
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    occ = np.asarray(mf.mo_occ)
    ene = np.asarray(mf.mo_energy)
    homo = int(np.where(occ > 0)[0][-1])
    lumo = int(np.where(occ == 0)[0][0])
    summary.update(
        {
            "homo_1based": homo + 1,
            "lumo_1based": lumo + 1,
            "homo_Eh": float(ene[homo]),
            "lumo_Eh": float(ene[lumo]),
            "gap_Eh": float(ene[lumo] - ene[homo]),
            "gap_eV": float((ene[lumo] - ene[homo]) * 27.211386245988),
        }
    )

    rows = build_orbital_rows(mol, mf)
    write_csv(rows, outdir / "canonical_orbitals.csv")
    write_csv(frontier_subset(rows, args.frontier), outdir / "frontier_orbitals.csv")
    molden.from_scf(mf, str(outdir / "canonical_orbitals.molden"))

    if args.stability:
        mo_i, stable_i = scf_stability.rhf_internal(
            mf,
            with_symmetry=False,
            return_status=True,
            nroots=args.stability_nroots,
            tol=args.stability_tol,
        )
        mo_e, stable_e = scf_stability.rhf_external(
            mf,
            with_symmetry=False,
            return_status=True,
            nroots=args.stability_nroots,
            tol=args.stability_tol,
        )
        summary["internal_stable_allow_symmetry_breaking"] = bool(stable_i)
        summary["rhf_to_uhf_stable_allow_symmetry_breaking"] = bool(stable_e)

        if not stable_i:
            np.save(outdir / "internal_instability_orbitals.npy", np.asarray(mo_i))

        if not stable_e and args.follow_external:
            uhf_info = follow_external_instability(
                mol,
                mo_e,
                outdir,
                conv_tol=args.conv_tol,
                max_cycle=args.max_cycle,
            )
            summary.update(uhf_info)
            if uhf_info.get("uhf_converged"):
                summary["uhf_minus_rhf_Eh"] = float(uhf_info["uhf_energy_Eh"] - mf.e_tot)
                summary["uhf_minus_rhf_eV"] = float(
                    (uhf_info["uhf_energy_Eh"] - mf.e_tot) * 27.211386245988
                )
    else:
        summary["internal_stable_allow_symmetry_breaking"] = None
        summary["rhf_to_uhf_stable_allow_symmetry_breaking"] = None

    summary["status"] = "OK"
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        ("index", "#"),
        ("molecule", "Molecule"),
        ("nao_nmo", "NMO"),
        ("scf_converged", "SCF"),
        ("gap_eV", "Gap/eV"),
        ("overlap_min_eigenvalue", "min(S)"),
        ("internal_stable_allow_symmetry_breaking", "Int.stab"),
        ("rhf_to_uhf_stable_allow_symmetry_breaking", "RHF->UHF"),
    ]
    print("  ".join(label for _, label in fields))
    for row in rows:
        vals = []
        for key, _ in fields:
            val = row.get(key, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            vals.append(str(val))
        print("  ".join(vals))


def main() -> None:
    p = argparse.ArgumentParser(description="RHF orbital/stability audit for benchmark molecules")
    p.add_argument("coordinates", type=Path)
    p.add_argument("--select", default="all", help="Molecule numbers: all, 1, 1,3-6, ...")
    p.add_argument("--basis", default="aug-cc-pvtz")
    p.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    p.add_argument("--outdir", type=Path, default=Path("rhf_orbital_audit"))
    p.add_argument("--frontier", type=int, default=8, help="Occupied/virtual MOs retained in frontier CSV")
    p.add_argument("--conv-tol", type=float, default=1e-10)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--max-memory", type=int, default=4000, help="PySCF memory limit in MB")
    p.add_argument("--verbose", type=int, default=4)
    p.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic molecular symmetry detection (default: yes)",
    )
    p.add_argument(
        "--newton-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If ordinary RHF does not converge, retry with second-order SCF",
    )
    p.add_argument(
        "--stability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RHF internal and RHF->UHF external stability tests",
    )
    p.add_argument("--stability-nroots", type=int, default=3)
    p.add_argument("--stability-tol", type=float, default=1e-4)
    p.add_argument(
        "--follow-external",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If RHF->UHF unstable, converge a UHF solution from the unstable mode",
    )
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    records = parse_selection(args.select, parse_coordinate_file(args.coordinates))

    summaries: List[Dict[str, object]] = []
    for rec in records:
        print(f"\n=== {rec[0]:02d} {rec[1]} ===", flush=True)
        try:
            summaries.append(audit_one(rec, args))
        except Exception as exc:
            err = {
                "index": rec[0],
                "molecule": rec[1],
                "status": f"ERROR:{type(exc).__name__}",
                "error": str(exc),
            }
            summaries.append(err)
            print(f"ERROR: {exc}", flush=True)

    write_csv(summaries, args.outdir / "audit_summary.csv")
    (args.outdir / "audit_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\n=== Batch summary ===")
    print_summary(summaries)
    print(f"\nWrote results below: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
