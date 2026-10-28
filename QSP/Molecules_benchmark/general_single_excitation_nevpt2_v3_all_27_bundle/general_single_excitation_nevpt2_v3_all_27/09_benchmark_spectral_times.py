#!/usr/bin/env python3
"""
09_benchmark_spectral_times.py

Display benchmark vertical excitation energies and the corresponding
characteristic oscillation period

    T = 2*pi*hbar / E = h/E

for the benchmark states stored in benchmark_single_excitation_cases.json.

The registry supplied with the benchmark workflow contains, for each state:
    - paper_tbe_augccpvtz_eV
    - paper_pc_nevpt2_augccpvtz_eV

The script reports both unless one is selected explicitly.

For E in eV:
    T(fs) = 4.135667696 / E(eV)

because h = 4.135667696 eV fs.

It also reports the equivalent frequency in THz.  An optional wavelength
column can be requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Exact enough for the present purpose; CODATA h in eV fs.
PLANCK_EV_FS = 4.135667696
HC_EV_NM = 1239.8419843320026

_HERE = Path(__file__).resolve().parent
_V3_REGISTRY = _HERE / "general_single_excitation_nevpt2_v3_all_27" / "benchmark_single_excitation_cases.json"
_DEFAULT_LOCAL = _HERE / "benchmark_single_excitation_cases.json"
DEFAULT_REGISTRY = _V3_REGISTRY if _V3_REGISTRY.exists() else _DEFAULT_LOCAL


def load_registry(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON registry: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Registry root must be a JSON object")
    return data


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "state"


def period_fs(energy_eV: float) -> float:
    if energy_eV <= 0.0:
        raise ValueError(f"Excitation energy must be positive, got {energy_eV}")
    return PLANCK_EV_FS / energy_eV


def frequency_thz(energy_eV: float) -> float:
    # E = h nu; 1 fs^-1 = 1000 THz.
    return energy_eV / PLANCK_EV_FS * 1000.0


def wavelength_nm(energy_eV: float) -> float:
    if energy_eV <= 0.0:
        raise ValueError(f"Excitation energy must be positive, got {energy_eV}")
    return HC_EV_NM / energy_eV


def parse_state_filter(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def iter_states(
    registry: Dict[str, Any],
    molecule: Optional[str] = None,
    state_keys: Optional[set[str]] = None,
    valence_only: bool = False,
    rydberg_only: bool = False,
) -> Iterable[Dict[str, Any]]:
    if valence_only and rydberg_only:
        raise ValueError("--valence-only and --rydberg-only are mutually exclusive")

    molecules = registry.get("molecules", registry)
    if not isinstance(molecules, dict):
        raise ValueError("Registry does not contain a molecule mapping")

    for name, entry in molecules.items():
        if name.startswith("_"):
            continue
        if molecule and name.lower() != molecule.lower():
            continue
        if not isinstance(entry, dict):
            continue
        states = entry.get("states", [])
        if not isinstance(states, list):
            continue
        for state in states:
            if not isinstance(state, dict):
                continue
            key = str(state.get("key", ""))
            if state_keys and key not in state_keys:
                continue
            stype = str(state.get("state_type", "")).upper()
            if valence_only and stype != "V":
                continue
            if rydberg_only and stype != "R":
                continue

            row = dict(state)
            row["molecule"] = name
            row["molecule_index"] = entry.get("index")
            row["charge"] = entry.get("charge", 0)
            yield row


def make_rows(states: Iterable[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for st in states:
        tbe = st.get("paper_tbe_augccpvtz_eV")
        nev = st.get("paper_pc_nevpt2_augccpvtz_eV")

        energies: List[tuple[str, float]] = []
        if source in {"tbe", "both"} and tbe is not None:
            energies.append(("TBE", float(tbe)))
        if source in {"pc-nevpt2", "both"} and nev is not None:
            energies.append(("PC-NEVPT2", float(nev)))

        for energy_label, energy in energies:
            rows.append(
                {
                    "molecule_index": st.get("molecule_index"),
                    "molecule": st["molecule"],
                    "state_key": st.get("key", ""),
                    "state": st.get("label", ""),
                    "multiplicity": st.get("multiplicity"),
                    "symmetry": st.get("paper_wfnsym", ""),
                    "state_type": st.get("state_type", ""),
                    "excitation": st.get("excitation", ""),
                    "energy_source": energy_label,
                    "E_vert_eV": energy,
                    "T_fs": period_fs(energy),
                    "frequency_THz": frequency_thz(energy),
                    "wavelength_nm": wavelength_nm(energy),
                    "TBE_eV": tbe,
                    "PC_NEVPT2_eV": nev,
                }
            )
    return rows


def fmt(value: Any, width: int, decimals: int = 3) -> str:
    if value is None or value == "":
        text = "-"
    elif isinstance(value, float):
        text = f"{value:.{decimals}f}"
    else:
        text = str(value)
    return text.rjust(width)


def print_table(rows: List[Dict[str, Any]], show_wavelength: bool) -> None:
    if not rows:
        print("No states matched the requested filters.")
        return

    headers = [
        ("#", 3),
        ("Molecule", 22),
        ("State", 18),
        ("Type", 4),
        ("Excitation", 16),
        ("Source", 10),
        ("E/eV", 9),
        ("T/fs", 10),
        ("nu/THz", 10),
    ]
    if show_wavelength:
        headers.append(("lambda/nm", 11))

    line = " ".join(title.ljust(width) for title, width in headers)
    print(line)
    print("-" * len(line))

    for i, row in enumerate(rows, start=1):
        values = [
            str(i),
            str(row["molecule"]),
            str(row["state"]),
            str(row["state_type"]),
            str(row["excitation"]),
            str(row["energy_source"]),
            f"{row['E_vert_eV']:.5f}",
            f"{row['T_fs']:.6f}",
            f"{row['frequency_THz']:.3f}",
        ]
        if show_wavelength:
            values.append(f"{row['wavelength_nm']:.3f}")

        widths = [width for _, width in headers]
        formatted = []
        for value, width in zip(values, widths):
            if width <= 4:
                formatted.append(value.rjust(width))
            elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                formatted.append(value.rjust(width))
            else:
                formatted.append(value.ljust(width))
        print(" ".join(formatted))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Display benchmark vertical excitation energies and T = 2*pi*hbar/E "
            "for the states in benchmark_single_excitation_cases.json."
        )
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Benchmark JSON registry (default: next to this script)",
    )
    p.add_argument("--molecule", help="Restrict output to one molecule")
    p.add_argument(
        "--states",
        help="Comma-separated state keys, e.g. s-b2-pi-to-pistar,t-b2-pi-to-pistar",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--valence-only", action="store_true")
    group.add_argument("--rydberg-only", action="store_true")
    p.add_argument(
        "--source",
        choices=["tbe", "pc-nevpt2", "both"],
        default="both",
        help="Benchmark energy source(s) to display",
    )
    p.add_argument("--wavelength", action="store_true", help="Also display lambda in nm")
    p.add_argument("--csv", type=Path, help="Write the table to CSV")
    p.add_argument(
        "--sort",
        choices=["registry", "energy", "period"],
        default="registry",
        help="Sort output rows",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        registry = load_registry(args.registry)
        states = iter_states(
            registry,
            molecule=args.molecule,
            state_keys=parse_state_filter(args.states),
            valence_only=args.valence_only,
            rydberg_only=args.rydberg_only,
        )
        rows = make_rows(states, args.source)

        if args.sort == "energy":
            rows.sort(key=lambda r: r["E_vert_eV"])
        elif args.sort == "period":
            rows.sort(key=lambda r: r["T_fs"], reverse=True)

        print(f"Registry: {args.registry.resolve()}")
        print(f"States displayed: {len(rows)}")
        print()
        print_table(rows, args.wavelength)

        if args.csv:
            write_csv(rows, args.csv)
            print(f"\nCSV written to {args.csv.resolve()}")

        print(
            "\nFormula used: T = 2*pi*hbar/E = h/E. "
            "For E in eV, T(fs) = 4.135667696/E(eV)."
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
