#!/usr/bin/env python3
"""Inspect molecular sizes from the benchmark coordinate file without SCF/DFT/CC.

The input format is the one used in the uploaded ``coordinates`` file, e.g.

    1 Acetaldehyde
    C  ... ... ...
    ...

    2 Carbon trimer
    C  ... ... ...

coordinates are assumed to be in bohr by default.

What this script does:
  * parses every molecule in the text file;
  * assigns charge/spin metadata (all neutral singlets except explicitly overridden);
  * builds only a PySCF ``gto.Mole`` object to load the requested basis;
  * reports electron counts and the size of the one-particle basis;
  * writes a CSV summary.

What this script does NOT do:
  * no SCF/HF/DFT calculation;
  * no molecular-orbital optimization;
  * no AO integral evaluation requested by the script;
  * no excited-state or correlated calculation.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from pyscf import gto


# These 35 systems are closed-shell singlets in the benchmark set.
# Default: neutral, spin = N_alpha - N_beta = 0.
# The cyanine system is the only charged species in the coordinate list.
SYSTEM_OVERRIDES = {
    "Streptocyanine-C1": {"charge": +1, "spin": 0},
}

# Atomic numbers needed by this data set.  Keeping this independent of PySCF
# makes the electron-count cross-check transparent.
ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
}

# Conventional frozen-core spatial orbitals for a later correlated calculation.
# This is only a bookkeeping diagnostic; this script does NOT freeze anything.
# H: none; C/N/O: 1s; S: [Ne] core = 1s,2s,2p (5 spatial orbitals).
FROZEN_CORE_ORBITALS = {
    "H": 0,
    "C": 1,
    "N": 1,
    "O": 1,
    "S": 5,
}

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
    """Parse numbered molecule blocks from the benchmark coordinate file."""
    records: List[MoleculeRecord] = []
    current_index = None
    current_name = None
    current_atoms: List[Atom] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            if current_index is not None:
                if not current_atoms:
                    raise ValueError(
                        f"Molecule {current_index} {current_name!r} has no atoms."
                    )
                records.append((current_index, current_name, current_atoms))

            current_index = int(header_match.group(1))
            current_name = header_match.group(2).strip()
            current_atoms = []
            continue

        atom_match = ATOM_RE.match(line)
        if atom_match and current_index is not None:
            symbol = atom_match.group(1)
            if symbol not in ATOMIC_NUMBER:
                raise ValueError(
                    f"Unsupported element {symbol!r} at line {lineno}. "
                    f"Extend ATOMIC_NUMBER if needed."
                )
            xyz = tuple(float(atom_match.group(i)) for i in (2, 3, 4))
            current_atoms.append((symbol, xyz))
            continue

        # Ignore preamble text before the first numbered molecule.
        if current_index is None:
            continue

        raise ValueError(f"Could not parse line {lineno}: {raw!r}")

    if current_index is not None:
        if not current_atoms:
            raise ValueError(f"Molecule {current_index} {current_name!r} has no atoms.")
        records.append((current_index, current_name, current_atoms))

    if not records:
        raise ValueError(f"No molecule blocks found in {path}")

    # Catch accidental missing/duplicated molecule numbers.
    indices = [idx for idx, _, _ in records]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate molecule indices found in coordinate file.")

    return records


def hill_formula(atoms: List[Atom]) -> str:
    """Return a Hill-system molecular formula from the parsed atom list."""
    counts = Counter(symbol for symbol, _ in atoms)

    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(el for el in counts if el not in {"C", "H"}))
    else:
        order = sorted(counts)

    def piece(el: str) -> str:
        n = counts[el]
        return el if n == 1 else f"{el}{n}"

    return "".join(piece(el) for el in order)


def nuclear_charge(atoms: List[Atom]) -> int:
    return sum(ATOMIC_NUMBER[symbol] for symbol, _ in atoms)


def conventional_frozen_core_orbitals(atoms: List[Atom]) -> int:
    return sum(FROZEN_CORE_ORBITALS[symbol] for symbol, _ in atoms)


def format_gib(nbytes: float) -> str:
    return f"{nbytes / (1024 ** 3):.3f}"


def inspect_system(
    index: int,
    name: str,
    atoms: List[Atom],
    basis: str,
    unit: str,
) -> Dict[str, object]:
    metadata = SYSTEM_OVERRIDES.get(name, {})
    charge = int(metadata.get("charge", 0))
    spin = int(metadata.get("spin", 0))

    z_total = nuclear_charge(atoms)
    expected_nelec = z_total - charge

    # For a closed-shell singlet the electron count must be even and spin = 0.
    if spin == 0 and expected_nelec % 2:
        raise ValueError(
            f"{name}: odd electron count ({expected_nelec}) with spin=0. "
            "Check charge/multiplicity metadata."
        )

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = unit
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.cart = False       # spherical GTOs, standard for correlation-consistent bases
    mol.verbose = 0
    mol.build()

    # Cross-check PySCF's electron bookkeeping against the transparent Z-charge count.
    if mol.nelectron != expected_nelec:
        raise RuntimeError(
            f"{name}: electron-count mismatch: expected {expected_nelec}, "
            f"PySCF reports {mol.nelectron}."
        )

    nalpha, nbeta = mol.nelec
    nao = mol.nao_nr()          # contracted spherical AO functions = AO basis dimension
    nmo = nao                   # nonrelativistic MO space has the same dimension after SCF
    nocc = nbeta                # closed-shell spatial occupied orbitals
    nvirt = nmo - nocc
    ncore = conventional_frozen_core_orbitals(atoms)
    nocc_fc = nocc - ncore

    # Purely algebraic storage diagnostics; no integrals or amplitudes are constructed.
    # dense_eri_gib: hypothetical dense (mu nu | lambda sigma) N_AO^4 tensor.
    # t2_gib: rough dense RHF-CCSD T2-size scale Nocc^2*Nvirt^2.
    dense_eri_bytes = 8.0 * (nao ** 4)
    t2_bytes = 8.0 * (nocc ** 2) * (nvirt ** 2)
    singles_dim = nocc * nvirt

    return {
        "index": index,
        "molecule": name,
        "formula": hill_formula(atoms),
        "charge": charge,
        "spin_2S": spin,
        "natoms": len(atoms),
        "nuclear_charge_Z": z_total,
        "electrons": mol.nelectron,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "basis": basis,
        "spherical": True,
        "basis_shells": mol.nbas,
        "primitive_AOs": mol.npgto_nr(),
        "contracted_AOs": nao,
        "spatial_MOs": nmo,
        "occupied_MOs": nocc,
        "virtual_MOs": nvirt,
        "conv_frozen_core_MOs": ncore,
        "valence_occupied_MOs_after_FC": nocc_fc,
        "singles_dimension_nocc_x_nvirt": singles_dim,
        "dense_AO_ERI_GiB": format_gib(dense_eri_bytes),
        "rough_CCSD_T2_GiB": format_gib(t2_bytes),
    }


def print_table(rows: List[Dict[str, object]]) -> None:
    columns = [
        ("index", "#"),
        ("molecule", "Molecule"),
        ("formula", "Formula"),
        ("charge", "Q"),
        ("electrons", "Ne"),
        ("contracted_AOs", "NAO/NMO"),
        ("occupied_MOs", "Nocc"),
        ("virtual_MOs", "Nvirt"),
        ("conv_frozen_core_MOs", "FC"),
        ("singles_dimension_nocc_x_nvirt", "Nocc*Nvirt"),
        ("dense_AO_ERI_GiB", "ERI GiB"),
    ]

    widths = {}
    for key, label in columns:
        widths[key] = max(len(label), max(len(str(row[key])) for row in rows))

    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    print(header)
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        parts = []
        for key, _ in columns:
            text = str(row[key])
            if key in {"molecule", "formula"}:
                parts.append(text.ljust(widths[key]))
            else:
                parts.append(text.rjust(widths[key]))
        print("  ".join(parts))


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PySCF basis/electron/orbital sizes without running SCF."
    )
    parser.add_argument("coordinates", type=Path, help="Benchmark coordinate text file")
    parser.add_argument(
        "--basis",
        default="aug-cc-pvtz",
        help="PySCF basis name (default: aug-cc-pvtz)",
    )
    parser.add_argument(
        "--unit",
        default="Bohr",
        choices=["Bohr", "Angstrom"],
        help="Coordinate unit (default: Bohr)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("system_size_summary.csv"),
        help="CSV output path (default: system_size_summary.csv)",
    )
    args = parser.parse_args()

    records = parse_coordinate_file(args.coordinates)
    rows = [
        inspect_system(index, name, atoms, basis=args.basis, unit=args.unit)
        for index, name, atoms in records
    ]

    print(f"Parsed {len(rows)} molecules from {args.coordinates}")
    print(f"Basis: {args.basis} | unit: {args.unit} | spherical GTOs: yes")
    print("No SCF/DFT/correlated calculation was run.\n")
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nWrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
