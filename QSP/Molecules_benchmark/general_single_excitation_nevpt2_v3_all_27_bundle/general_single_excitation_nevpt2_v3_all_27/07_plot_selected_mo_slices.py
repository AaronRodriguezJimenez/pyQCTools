#!/usr/bin/env python3
"""Plot selectable molecular-orbital slices from a PySCF checkpoint.

The plotting style matches the slice plots produced by the general
SA-CASSCF/CASCI/SC-NEVPT2 driver:

* best-fit molecular plane from the heavy atoms;
* optional plane offsets along the plane normal;
* signed red/blue MO amplitudes;
* projected atoms, bonds, and element labels;
* automatic orbital-symmetry labels when available.

The script can read either canonical SCF orbitals (``scf/mo_coeff``) or
CASSCF/CASCI orbitals (``mcscf/mo_coeff``) from a PySCF HDF5 checkpoint.
MO numbers supplied on the command line are one-based.

Examples
--------
Plot RHF MOs 11 and 13 on both sides of the molecular plane::

    python 07_plot_selected_mo_slices.py \
        prepared_cases/08_Streptocyanine-C1/results/rhf.chk \
        --source scf --mos 11,13 --offsets 0.70,-0.70

Plot the state-averaged CASSCF orbitals actually used for one target state::

    python 07_plot_selected_mo_slices.py \
        prepared_cases/08_Streptocyanine-C1/results/s-b2-pi-to-pistar/sa_casscf.chk \
        --source mcscf --mos 11,13 --offsets 0.70,-0.70

List all available MOs, occupations, energies, and irreps without plotting::

    python 07_plot_selected_mo_slices.py path/to/rhf.chk --source scf --list-only

Plot a range of orbitals::

    python 07_plot_selected_mo_slices.py path/to/rhf.chk \
        --mos 9-14 --offsets 0.0,0.70 --cube
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import lib, symm
from pyscf.tools import cubegen

ANGSTROM_PER_BOHR = 0.529177210903
COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
}


@dataclass
class OrbitalData:
    mol: object
    coefficients: np.ndarray
    occupations: np.ndarray | None
    energies: np.ndarray | None
    source_key: str
    spin_channel: str


def parse_integer_selection(text: str, maximum: int) -> List[int]:
    """Parse one-based selections such as ``11,13,16-20``."""
    text = text.strip().lower()
    if text == "all":
        return list(range(1, maximum + 1))

    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                start, stop = stop, start
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(token))

    bad = sorted(value for value in selected if value < 1 or value > maximum)
    if bad:
        raise ValueError(f"MO numbers outside 1..{maximum}: {bad}")
    if not selected:
        raise ValueError("No MO numbers were selected")
    return sorted(selected)


def parse_float_list(text: str) -> List[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one plane offset is required")
    return values


def parse_atom_triplet(text: str | None, natm: int) -> Tuple[int, int, int] | None:
    if text is None:
        return None
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError("--plane-atoms requires exactly three one-based atom indices")
    if len(set(values)) != 3 or any(value < 1 or value > natm for value in values):
        raise ValueError(f"Invalid --plane-atoms values {values}; molecule has {natm} atoms")
    return tuple(value - 1 for value in values)


def key_exists(checkpoint: Path, key: str) -> bool:
    with h5py.File(checkpoint, "r") as handle:
        return key in handle


def load_optional(checkpoint: Path, key: str):
    if not key_exists(checkpoint, key):
        return None
    return lib.chkfile.load(str(checkpoint), key)


def select_spin_channel(array, channel: str, name: str) -> np.ndarray | None:
    if array is None:
        return None
    value = np.asarray(array)
    if value.ndim >= 1 and value.shape[0] == 2 and name in {"mo_coeff", "mo_occ", "mo_energy"}:
        index = 0 if channel == "alpha" else 1
        value = np.asarray(value[index])
    return value


def resolve_source_key(checkpoint: Path, source: str, custom_key: str | None) -> str:
    if custom_key:
        if not key_exists(checkpoint, custom_key):
            raise KeyError(f"Checkpoint does not contain {custom_key!r}")
        return custom_key

    choices = {
        "scf": "scf/mo_coeff",
        "mcscf": "mcscf/mo_coeff",
    }
    if source in choices:
        key = choices[source]
        if not key_exists(checkpoint, key):
            raise KeyError(f"Checkpoint does not contain {key!r}")
        return key

    # In auto mode, prefer optimized MCSCF orbitals when they are present.
    for key in ("mcscf/mo_coeff", "scf/mo_coeff"):
        if key_exists(checkpoint, key):
            return key
    raise KeyError("No mcscf/mo_coeff or scf/mo_coeff dataset was found")


def load_orbitals(
    checkpoint: Path,
    source: str,
    custom_key: str | None,
    spin_channel: str,
) -> OrbitalData:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    mol = lib.chkfile.load_mol(str(checkpoint))
    key = resolve_source_key(checkpoint, source, custom_key)
    prefix = key.rsplit("/", 1)[0]

    coefficients = select_spin_channel(
        lib.chkfile.load(str(checkpoint), key), spin_channel, "mo_coeff"
    )
    if coefficients is None or coefficients.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional MO coefficient matrix at {key!r}; "
            f"received shape {None if coefficients is None else coefficients.shape}"
        )

    occupations = select_spin_channel(
        load_optional(checkpoint, f"{prefix}/mo_occ"), spin_channel, "mo_occ"
    )
    energies = select_spin_channel(
        load_optional(checkpoint, f"{prefix}/mo_energy"), spin_channel, "mo_energy"
    )

    if occupations is not None and occupations.ndim != 1:
        occupations = None
    if energies is not None and energies.ndim != 1:
        energies = None

    return OrbitalData(
        mol=mol,
        coefficients=np.asarray(coefficients),
        occupations=occupations,
        energies=energies,
        source_key=key,
        spin_channel=spin_channel,
    )


def orbital_irreps(mol, coefficients: np.ndarray) -> List[str]:
    if not getattr(mol, "symmetry", False):
        return [""] * coefficients.shape[1]
    try:
        labels = symm.label_orb_symm(
            mol,
            mol.irrep_name,
            mol.symm_orb,
            coefficients,
            check=False,
        )
        return [str(label) for label in labels]
    except Exception as exc:
        print(
            f"WARNING: could not label orbital symmetries ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return [""] * coefficients.shape[1]


def molecular_plane_axes(
    mol,
    plane_atoms: Tuple[int, int, int] | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(mol.atom_coords(), dtype=float)

    if plane_atoms is not None:
        i, j, k = plane_atoms
        selected = coords[[i, j, k]]
        center = selected.mean(axis=0)
        u_axis = selected[1] - selected[0]
        u_norm = np.linalg.norm(u_axis)
        if u_norm < 1e-12:
            raise ValueError("The first two --plane-atoms are coincident")
        u_axis /= u_norm
        normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            raise ValueError("The three --plane-atoms are collinear")
        normal /= normal_norm
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis)
    else:
        charges = np.asarray(mol.atom_charges())
        fit = coords[charges > 1]
        if len(fit) < 3:
            fit = coords
        center = fit.mean(axis=0)
        centered = fit - center
        _, singular, vh = np.linalg.svd(centered, full_matrices=True)

        if len(fit) == 2 or (len(singular) >= 2 and singular[1] < 1e-8):
            u_axis = centered[-1] - centered[0]
            u_axis /= np.linalg.norm(u_axis)
            trial = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(trial, u_axis))) > 0.85:
                trial = np.array([0.0, 1.0, 0.0])
            normal = np.cross(u_axis, trial)
            normal /= np.linalg.norm(normal)
            v_axis = np.cross(normal, u_axis)
            v_axis /= np.linalg.norm(v_axis)
        else:
            u_axis, v_axis, normal = vh[0].copy(), vh[1].copy(), vh[-1].copy()

    # Make the automatically chosen frame deterministic.
    for vector in (u_axis, v_axis, normal):
        largest = int(np.argmax(np.abs(vector)))
        if vector[largest] < 0:
            vector *= -1
    if np.dot(np.cross(u_axis, v_axis), normal) < 0:
        v_axis *= -1

    return center, u_axis, v_axis, normal


def inferred_bonds(mol) -> List[Tuple[int, int]]:
    coords = np.asarray(mol.atom_coords()) * ANGSTROM_PER_BOHR
    bonds: List[Tuple[int, int]] = []
    for i in range(mol.natm):
        ri = COVALENT_RADII_ANGSTROM.get(mol.atom_symbol(i), 0.75)
        for j in range(i + 1, mol.natm):
            rj = COVALENT_RADII_ANGSTROM.get(mol.atom_symbol(j), 0.75)
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= 1.25 * (ri + rj) + 0.10:
                bonds.append((i, j))
    return bonds


def component_values(values: np.ndarray, component: str) -> np.ndarray:
    if component == "real":
        imag_max = float(np.max(np.abs(values.imag))) if np.iscomplexobj(values) else 0.0
        real_max = float(np.max(np.abs(values.real)))
        if imag_max > max(1e-8, 1e-5 * max(real_max, 1e-16)):
            print(
                f"WARNING: selected orbital has a non-negligible imaginary part "
                f"(max |Im|={imag_max:.3e}); consider --component imag.",
                file=sys.stderr,
            )
        return np.asarray(values.real)
    if component == "imag":
        return np.asarray(values.imag)
    if component == "abs":
        return np.asarray(np.abs(values))
    raise ValueError(component)


def evaluate_slice(
    mol,
    coefficient: np.ndarray,
    offset_bohr: float,
    grid_size: int,
    margin_bohr: float,
    plane_atoms: Tuple[int, int, int] | None,
    component: str,
):
    center, u_axis, v_axis, normal = molecular_plane_axes(mol, plane_atoms)
    atom_coords = np.asarray(mol.atom_coords(), dtype=float)
    relative = atom_coords - center
    atom_u = relative @ u_axis
    atom_v = relative @ v_axis

    u_values = np.linspace(
        float(atom_u.min()) - margin_bohr,
        float(atom_u.max()) + margin_bohr,
        grid_size,
    )
    v_values = np.linspace(
        float(atom_v.min()) - margin_bohr,
        float(atom_v.max()) + margin_bohr,
        grid_size,
    )
    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    points = (
        center[None, :]
        + offset_bohr * normal[None, :]
        + uu.reshape(-1, 1) * u_axis[None, :]
        + vv.reshape(-1, 1) * v_axis[None, :]
    )

    evaluator = "GTOval_cart" if getattr(mol, "cart", False) else "GTOval_sph"
    ao_values = mol.eval_gto(evaluator, points)
    values = (ao_values @ coefficient).reshape(grid_size, grid_size)
    values = component_values(values, component)
    return uu, vv, values, atom_u, atom_v


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_") or "orbital"


def offset_label(offset: float) -> str:
    sign = "p" if offset >= 0 else "m"
    return f"{sign}{abs(offset):.2f}".replace(".", "p")


def plot_one_slice(
    data: OrbitalData,
    mo_1based: int,
    irrep: str,
    offset_bohr: float,
    output_path: Path,
    grid_size: int,
    margin_bohr: float,
    quantile: float,
    levels_count: int,
    dpi: int,
    plane_atoms: Tuple[int, int, int] | None,
    component: str,
    phase: float,
    title_prefix: str | None,
) -> float:
    coefficient = phase * np.asarray(data.coefficients[:, mo_1based - 1])
    uu, vv, values, atom_u, atom_v = evaluate_slice(
        data.mol,
        coefficient,
        offset_bohr,
        grid_size,
        margin_bohr,
        plane_atoms,
        component,
    )

    absolute = np.abs(values)
    vmax = float(np.quantile(absolute, quantile))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = float(absolute.max())
    if vmax < 1e-12:
        raise RuntimeError("Orbital is numerically zero on the requested slice")

    if component == "abs":
        levels = np.linspace(0.0, vmax, levels_count)
        cmap = "viridis"
    else:
        levels = np.linspace(-vmax, vmax, levels_count)
        cmap = "seismic"

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    contour = ax.contourf(uu, vv, values, levels=levels, cmap=cmap, extend="both")
    if component != "abs" and float(values.min()) < 0.0 < float(values.max()):
        ax.contour(uu, vv, values, levels=[0.0], colors="black", linewidths=0.6)

    for i, j in inferred_bonds(data.mol):
        ax.plot(
            [atom_u[i], atom_u[j]],
            [atom_v[i], atom_v[j]],
            color="black",
            linewidth=1.3,
        )
    for atom_index in range(data.mol.natm):
        symbol = data.mol.atom_symbol(atom_index)
        ax.scatter([atom_u[atom_index]], [atom_v[atom_index]], s=30, color="black", zorder=3)
        ax.text(
            atom_u[atom_index] + 0.08,
            atom_v[atom_index] + 0.08,
            symbol,
            fontsize=9,
        )

    irrep_text = f", {irrep}" if irrep else ""
    prefix = f"{title_prefix}\n" if title_prefix else ""
    ax.set_title(
        f"{prefix}Selected orbital: full MO {mo_1based}{irrep_text}\n"
        f"plane offset = {offset_bohr:+.2f} bohr"
    )
    ax.set_xlabel("u / bohr")
    ax.set_ylabel("v / bohr")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label=f"MO amplitude ({component})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return vmax


def print_orbital_table(data: OrbitalData, irreps: Sequence[str]) -> None:
    print(f"Checkpoint orbital key: {data.source_key}")
    print(f"Number of MOs: {data.coefficients.shape[1]}")
    print(f"{'MO':>5} {'Occ':>8} {'Energy/Eh':>14} {'Irrep':>10}")
    for index in range(data.coefficients.shape[1]):
        occupation = "" if data.occupations is None else f"{float(data.occupations[index]):.4f}"
        energy = "" if data.energies is None else f"{float(data.energies[index]):.8f}"
        print(f"{index + 1:5d} {occupation:>8} {energy:>14} {irreps[index]:>10}")


def write_rows(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot arbitrary PySCF molecular orbitals on molecular-plane slices"
    )
    parser.add_argument("checkpoint", type=Path, help="PySCF HDF5 checkpoint")
    parser.add_argument(
        "--source",
        choices=["auto", "scf", "mcscf"],
        default="auto",
        help="Orbital source; auto prefers mcscf/mo_coeff when present",
    )
    parser.add_argument(
        "--key",
        help="Custom HDF5 coefficient key, overriding --source, e.g. scf/mo_coeff",
    )
    parser.add_argument(
        "--spin-channel",
        choices=["alpha", "beta"],
        default="alpha",
        help="Channel to plot if an unrestricted coefficient pair is encountered",
    )
    parser.add_argument(
        "--mos",
        default="11",
        help="One-based MO selection: 11,13, 9-14, or all",
    )
    parser.add_argument(
        "--offsets",
        default="0.70,-0.70",
        help="Comma-separated offsets along the fitted plane normal, in bohr",
    )
    parser.add_argument(
        "--plane-atoms",
        help="Optional three one-based atom indices defining the plotting plane",
    )
    parser.add_argument("--outdir", type=Path, default=Path("selected_mo_slices"))
    parser.add_argument("--grid-size", type=int, default=301)
    parser.add_argument("--margin", type=float, default=3.0, help="Plot margin in bohr")
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--component",
        choices=["real", "imag", "abs"],
        default="real",
    )
    parser.add_argument(
        "--flip-mos",
        default="",
        help="One-based MO list whose overall sign should be reversed",
    )
    parser.add_argument("--title-prefix")
    parser.add_argument("--cube", action="store_true", help="Also write Gaussian cube files")
    parser.add_argument("--cube-resolution", type=float, default=0.18)
    parser.add_argument("--cube-margin", type=float, default=3.0)
    parser.add_argument("--list-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.quantile <= 1.0:
        raise ValueError("--quantile must be in (0,1]")
    if args.grid_size < 40:
        raise ValueError("--grid-size is too small")
    if args.levels < 3:
        raise ValueError("--levels must be at least 3")

    data = load_orbitals(
        args.checkpoint,
        args.source,
        args.key,
        args.spin_channel,
    )
    nmo = data.coefficients.shape[1]
    irreps = orbital_irreps(data.mol, data.coefficients)

    if args.list_only:
        print_orbital_table(data, irreps)
        return

    selected = parse_integer_selection(args.mos, nmo)
    flipped = set()
    if args.flip_mos.strip():
        flipped = set(parse_integer_selection(args.flip_mos, nmo))
    offsets = parse_float_list(args.offsets)
    plane_atoms = parse_atom_triplet(args.plane_atoms, data.mol.natm)

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {data.source_key} from {args.checkpoint}")
    print(f"Plotting MOs: {selected}")
    print(f"Offsets (bohr): {offsets}")

    rows = []
    for mo_1based in selected:
        irrep = irreps[mo_1based - 1]
        occupation = (
            None if data.occupations is None else float(data.occupations[mo_1based - 1])
        )
        energy = None if data.energies is None else float(data.energies[mo_1based - 1])
        phase = -1.0 if mo_1based in flipped else 1.0

        print(
            f"MO {mo_1based}: irrep={irrep or 'unlabeled'}, "
            f"occ={occupation}, energy_Eh={energy}, phase={phase:+.0f}"
        )

        coefficient = phase * np.asarray(data.coefficients[:, mo_1based - 1])
        if args.cube:
            cube_path = args.outdir / f"mo{mo_1based:03d}_{safe_label(irrep)}.cube"
            cubegen.orbital(
                data.mol,
                str(cube_path),
                coefficient,
                resolution=args.cube_resolution,
                margin=args.cube_margin,
            )

        for offset in offsets:
            filename = (
                f"mo{mo_1based:03d}_{safe_label(irrep)}_"
                f"offset_{offset_label(offset)}_{args.component}.png"
            )
            output = args.outdir / filename
            try:
                vmax = plot_one_slice(
                    data=data,
                    mo_1based=mo_1based,
                    irrep=irrep,
                    offset_bohr=offset,
                    output_path=output,
                    grid_size=args.grid_size,
                    margin_bohr=args.margin,
                    quantile=args.quantile,
                    levels_count=args.levels,
                    dpi=args.dpi,
                    plane_atoms=plane_atoms,
                    component=args.component,
                    phase=phase,
                    title_prefix=args.title_prefix,
                )
                print(f"  wrote {output}")
                status = "OK"
                error = ""
            except RuntimeError as exc:
                print(f"  WARNING at offset {offset:+.2f}: {exc}", file=sys.stderr)
                vmax = None
                status = "SKIPPED"
                error = str(exc)

            rows.append(
                {
                    "checkpoint": str(args.checkpoint),
                    "source_key": data.source_key,
                    "spin_channel": data.spin_channel,
                    "mo_1based": mo_1based,
                    "irrep": irrep,
                    "occupation": occupation,
                    "energy_Eh": energy,
                    "phase_factor": phase,
                    "component": args.component,
                    "offset_bohr": offset,
                    "vmax": vmax,
                    "status": status,
                    "error": error,
                    "output_png": str(output) if status == "OK" else "",
                }
            )

    write_rows(rows, args.outdir / "plot_manifest.csv")
    print(f"Manifest: {args.outdir / 'plot_manifest.csv'}")


if __name__ == "__main__":
    main()
