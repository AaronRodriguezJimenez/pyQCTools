#!/usr/bin/env python3
"""Acetaldehyde n -> pi* excitation audit with SA-CASSCF and SC-NEVPT2.

This script is the acetaldehyde analogue of ``04_furan_nevpt2_valence.py``.
It reads acetaldehyde from the supplied numbered ``Coordinates`` file and
characterizes the two lowest valence n -> pi* states reported in Sarkar et al.,
J. Chem. Theory Comput. 2022, 18, 2418-2436:

    1 A''  (V; n -> pi*)
    3 A''  (V; n -> pi*)

The default STO-3G calculation is deliberately a proof of concept.  The script
constructs a compact CAS(4e,3o) space containing

    n(O), pi(C=O), pi*(C=O).

Unlike the furan script, the active space is selected from the canonical RHF
orbitals rather than by a simple AVAS p-orbital list.  This is necessary because
the carbonyl n orbital is an in-plane oxygen hybrid, not a single Cartesian AO.
The automatic selector uses

  * C_s orbital symmetry,
  * oxygen-centered meta-Lowdin population,
  * projected carbonyl p character normal to the C-C-O plane,
  * occupation and orbital energy.

A manual 1-based MO list can always be supplied with ``--active-mos``.  The
manual order must be n,pi,pi*.

For each target state, the workflow is:

  1. Closed-shell RHF at the supplied ground-state geometry.
  2. Automatic or manual selection of a CAS(4,3) n/pi/pi* active space.
  3. Equal-weight state-averaged CASSCF over the 1 A' ground state and the
     requested 1 A'' or 3 A'' state.
  4. Separate symmetry- and spin-resolved CASCI calculations in the optimized
     state-averaged orbitals.  PySCF requires this because NEVPT2 does not act
     directly on the state-average solver wrapper.
  5. State-specific strongly contracted NEVPT2 corrections for ground and
     excited states and the resulting vertical excitation energy.
  6. CI configuration analysis, including the summed weight of configurations
     that differ from the dominant ground configuration by one electron
     promotion and the largest hole-particle contribution.
  7. Cube files and signed two-dimensional slices for the dominant hole and
     particle orbitals.  The n hole is plotted in the molecular symmetry plane;
     the pi* particle is plotted on a parallel plane displaced from the node.

Important scope limitations
---------------------------
* PySCF's built-in ``mrpt.NEVPT`` is strongly contracted NEVPT2 (SC-NEVPT2).
  The main NEVPT2 column in the paper is partially contracted NEVPT2.
* The default STO-3G calculation is a workflow test, not a numerical
  reproduction of the paper's aug-cc-pVTZ calculation.
* CAS(4,3) is a chemically motivated minimal active space chosen here.  The
  exact active space used by the paper is described in its Supporting
  Information and is not assumed by this script.
* The reported single-excitation configuration weights depend on the chosen
  active-orbital basis.  They are not equivalent to a coupled-cluster %T1
  diagnostic.

Examples
--------
Proof of concept for both states::

    python 05_acetaldehyde_nevpt2_npi.py Coordinates \
        --basis sto-3g \
        --outdir acetaldehyde_sc_nevpt2_sto3g

Only the triplet::

    python 05_acetaldehyde_nevpt2_npi.py Coordinates \
        --basis sto-3g \
        --states triplet-app \
        --outdir acetaldehyde_triplet_test

After inspecting ``orbital_selection_diagnostics.csv``, the automatic choice
can be overridden; the required order is n,pi,pi*::

    python 05_acetaldehyde_nevpt2_npi.py Coordinates \
        --basis sto-3g \
        --active-mos <n_mo>,<pi_mo>,<pi_star_mo>

Requirements
------------
Python 3.10+, NumPy, Matplotlib, and PySCF with MCSCF/NEVPT2 support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pyscf
    from pyscf import fci, gto, lo, mcscf, mrpt, scf, symm
    from pyscf.fci import cistring
    from pyscf.tools import cubegen, molden
except ImportError as exc:
    raise SystemExit(
        "PySCF is required. Install it in the environment used to run this "
        "script, for example with `python -m pip install pyscf matplotlib`."
    ) from exc


HARTREE_TO_EV = 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903

HEADER_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$"
)
P_COMPONENT_RE = re.compile(r"^(.*?p)([xyz])$", re.IGNORECASE)

Atom = Tuple[str, Tuple[float, float, float]]
MoleculeRecord = Tuple[int, str, List[Atom]]


@dataclass(frozen=True)
class TargetState:
    key: str
    label: str
    multiplicity: int
    spin_2s: int
    target_root: int
    paper_tbe_augccpvtz_eV: float
    paper_pc_nevpt2_augccpvtz_eV: float


TARGET_STATES: Mapping[str, TargetState] = {
    "singlet-app": TargetState(
        key="singlet-app",
        label="1A'' (V; n->pi*)",
        multiplicity=1,
        spin_2s=0,
        target_root=0,
        paper_tbe_augccpvtz_eV=4.31,
        paper_pc_nevpt2_augccpvtz_eV=4.39,
    ),
    "triplet-app": TargetState(
        key="triplet-app",
        label="3A'' (V; n->pi*)",
        multiplicity=3,
        spin_2s=2,
        target_root=0,
        paper_tbe_augccpvtz_eV=3.97,
        paper_pc_nevpt2_augccpvtz_eV=4.00,
    ),
}

DEFAULT_STATE_ORDER = ("triplet-app", "singlet-app")

COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
}


# -----------------------------------------------------------------------------
# Input and general utilities
# -----------------------------------------------------------------------------


def parse_coordinate_file(path: Path) -> List[MoleculeRecord]:
    records: List[MoleculeRecord] = []
    current_index: int | None = None
    current_name: str | None = None
    current_atoms: List[Atom] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            if current_index is not None:
                if not current_atoms:
                    raise ValueError(f"Molecule {current_index} has no atoms")
                assert current_name is not None
                records.append((current_index, current_name, current_atoms))
            current_index = int(header_match.group(1))
            current_name = header_match.group(2).strip()
            current_atoms = []
            continue

        atom_match = ATOM_RE.match(line)
        if atom_match and current_index is not None:
            symbol = atom_match.group(1)
            xyz = tuple(float(atom_match.group(i)) for i in (2, 3, 4))
            current_atoms.append((symbol, xyz))
            continue

        if current_index is None:
            continue
        raise ValueError(f"Could not parse line {lineno}: {raw!r}")

    if current_index is not None:
        if not current_atoms:
            raise ValueError(f"Molecule {current_index} has no atoms")
        assert current_name is not None
        records.append((current_index, current_name, current_atoms))

    if not records:
        raise ValueError(f"No molecule blocks found in {path}")
    return records


def find_named_molecule(records: Sequence[MoleculeRecord], name: str) -> MoleculeRecord:
    matches = [rec for rec in records if rec[1].casefold() == name.casefold()]
    if len(matches) != 1:
        available = ", ".join(rec[1] for rec in records)
        raise ValueError(
            f"Expected exactly one molecule named {name!r}; found {len(matches)}. "
            f"Available names: {available}"
        )
    return matches[0]


def hill_formula(atoms: Sequence[Atom]) -> str:
    counts = Counter(symbol for symbol, _ in atoms)
    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(el for el in counts if el not in {"C", "H"}))
    else:
        order = sorted(counts)
    return "".join(el if counts[el] == 1 else f"{el}{counts[el]}" for el in order)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "state"


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_root_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    return list(array)


def root_value(value: Any, root: int) -> float:
    values = as_root_list(value)
    if root >= len(values):
        raise IndexError(f"Requested root {root}, but only {len(values)} values exist")
    return float(values[root])


def root_ci(mc: Any, root: int) -> np.ndarray:
    ci = mc.ci
    if isinstance(ci, (list, tuple)):
        return np.asarray(ci[root])
    array = np.asarray(ci)
    if getattr(mc.fcisolver, "nroots", 1) > 1 and array.ndim >= 3:
        return np.asarray(array[root])
    if root != 0:
        raise IndexError("The CI object contains only one root")
    return array


def normalize_active_nelec(nelecas: Any, spin_2s: int) -> Tuple[int, int]:
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        total = int(nelecas[0]) + int(nelecas[1])
    else:
        total = int(nelecas)
    if (total + spin_2s) % 2:
        raise ValueError(f"Cannot combine {total} active electrons with 2S={spin_2s}")
    nalpha = (total + spin_2s) // 2
    nbeta = total - nalpha
    if nbeta < 0:
        raise ValueError(f"Invalid active electron split ({nalpha}, {nbeta})")
    return nalpha, nbeta


def parse_active_mos(text: str | None) -> List[int] | None:
    if text is None or not text.strip():
        return None
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--active-mos must be a comma-separated list of 1-based integers"
        ) from exc
    if len(values) != 3 or len(set(values)) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError(
            "--active-mos must contain exactly three distinct positive MO indices "
            "in the order n,pi,pi*"
        )
    return values


# -----------------------------------------------------------------------------
# Molecule, RHF, symmetry, and canonical-orbital diagnostics
# -----------------------------------------------------------------------------


def build_molecule(
    atoms: Sequence[Atom],
    basis: str,
    unit: str,
    max_memory: int,
    log_path: Path,
    verbose: int,
) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = unit
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.cart = False
    mol.symmetry = True
    mol.max_memory = max_memory
    mol.verbose = verbose
    mol.output = str(log_path)
    mol.build()
    return mol


def run_rhf(mol: gto.Mole, args: argparse.Namespace, outdir: Path) -> scf.hf.SCF:
    mf = scf.RHF(mol)
    mf.chkfile = str(outdir / "rhf.chk")
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle
    mf.max_memory = args.max_memory
    mf.kernel()

    if not mf.converged and args.newton_fallback:
        dm0 = mf.make_rdm1() if mf.mo_coeff is not None else None
        nmf = mf.newton()
        nmf.chkfile = str(outdir / "rhf.chk")
        nmf.conv_tol = args.scf_conv_tol
        nmf.max_cycle = args.scf_max_cycle
        nmf.max_memory = args.max_memory
        nmf.kernel(dm0)
        mf = nmf

    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def conventional_core_orbitals(mol: gto.Mole) -> int:
    count = 0
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        if symbol in {"C", "N", "O", "F"}:
            count += 1
        elif symbol in {"S", "P", "Cl"}:
            count += 5
    return count


def identify_carbonyl_atoms(mol: gto.Mole) -> Tuple[int, int, int]:
    oxygen_indices = [i for i in range(mol.natm) if mol.atom_symbol(i) == "O"]
    carbon_indices = [i for i in range(mol.natm) if mol.atom_symbol(i) == "C"]
    if len(oxygen_indices) != 1 or len(carbon_indices) != 2:
        raise RuntimeError(
            "The acetaldehyde selector expects exactly one O and two C atoms; "
            f"found O={oxygen_indices}, C={carbon_indices}"
        )

    coords = np.asarray(mol.atom_coords())
    oxygen = oxygen_indices[0]
    carbonyl_carbon = min(
        carbon_indices, key=lambda i: float(np.linalg.norm(coords[i] - coords[oxygen]))
    )
    methyl_carbon = next(i for i in carbon_indices if i != carbonyl_carbon)
    return oxygen, carbonyl_carbon, methyl_carbon


def carbonyl_frame(
    mol: gto.Mole, oxygen: int, carbonyl_carbon: int, methyl_carbon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return center, C=O axis, second in-plane axis, and plane normal."""
    coords = np.asarray(mol.atom_coords(), dtype=float)
    center = coords[carbonyl_carbon]
    co_axis = coords[oxygen] - center
    co_axis /= np.linalg.norm(co_axis)

    cc_vector = coords[methyl_carbon] - center
    in_plane = cc_vector - np.dot(cc_vector, co_axis) * co_axis
    norm = float(np.linalg.norm(in_plane))
    if norm < 1e-8:
        raise RuntimeError("C-C-O atoms are nearly collinear; cannot define C_s plane")
    in_plane /= norm

    normal = np.cross(co_axis, in_plane)
    normal /= np.linalg.norm(normal)

    # Deterministic signs make plots and projected populations reproducible.
    for vector in (co_axis, in_plane, normal):
        largest = int(np.argmax(np.abs(vector)))
        if vector[largest] < 0:
            vector *= -1
    if np.dot(np.cross(co_axis, in_plane), normal) < 0:
        in_plane *= -1
    return center, co_axis, in_plane, normal


def resolve_cs_irreps(mol: gto.Mole) -> Tuple[str, str]:
    names = [str(name) for name in mol.irrep_name]
    ids = [int(value) for value in mol.irrep_id]
    if str(mol.groupname).lower() != "cs" or len(names) != 2:
        raise RuntimeError(
            f"Expected C_s symmetry for acetaldehyde; PySCF detected "
            f"{mol.groupname} with irreps {names}"
        )

    if 0 in ids:
        ground = names[ids.index(0)]
    else:
        # Fallback for unusual builds; the first irrep is normally totally symmetric.
        ground = names[0]
    target = next(name for name in names if name != ground)
    return ground, target


def orbital_symmetry_labels(mol: gto.Mole, coeff: np.ndarray) -> List[str]:
    labels = symm.label_orb_symm(
        mol,
        mol.irrep_name,
        mol.symm_orb,
        coeff,
        check=False,
    )
    return [str(label) for label in labels]


def meta_lowdin_coefficients(
    mol: gto.Mole, coeff: np.ndarray, overlap: np.ndarray
) -> np.ndarray:
    x = lo.orth.orth_ao(mol, method="meta_lowdin", s=overlap)
    return x.T.conj() @ overlap @ coeff


def projected_p_population(
    mol: gto.Mole,
    orth_coeff: np.ndarray,
    atom_indices: Sequence[int],
    direction: np.ndarray,
) -> np.ndarray:
    """Population in atom-centered p functions projected along ``direction``.

    The projection is formed separately for every radial p shell, using the
    meta-Lowdin orthonormal AO coefficients.  This permits arbitrary molecular
    orientation rather than assuming that the symmetry-plane normal is exactly
    x, y, or z.
    """
    labels = mol.ao_labels(fmt=False)
    groups: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(dict)
    selected_atoms = set(int(i) for i in atom_indices)

    for iao, label in enumerate(labels):
        atom_id = int(label[0])
        if atom_id not in selected_atoms:
            continue
        # With fmt=False, PySCF returns (atom_id, symbol, nl, component),
        # for example (2, "O", "2p", "y").
        radial_key = str(label[2]).lower().replace(" ", "")
        component = str(label[3]).lower().replace(" ", "")
        if radial_key.endswith("p") and component in {"x", "y", "z"}:
            groups[(atom_id, radial_key)][component] = iao

    result = np.zeros(orth_coeff.shape[1])
    component_index = {"x": 0, "y": 1, "z": 2}
    for component_map in groups.values():
        amplitude = np.zeros(orth_coeff.shape[1], dtype=complex)
        for component, iao in component_map.items():
            amplitude += direction[component_index[component]] * orth_coeff[iao]
        result += np.abs(amplitude) ** 2
    return result.real


def spatial_moments(mol: gto.Mole, coeff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = mol.intor_symmetric("int1e_r", comp=3)
    r2 = mol.intor_symmetric("int1e_r2")
    centers = np.einsum("pi,xpq,qi->ix", coeff.conj(), r, coeff).real
    mean_r2 = np.einsum("pi,pq,qi->i", coeff.conj(), r2, coeff).real
    spreads = mean_r2 - np.einsum("ix,ix->i", centers, centers)
    return centers, np.maximum(spreads, 0.0)


def frontier_label(index: int, homo: int, lumo: int) -> str:
    if index <= homo:
        delta = homo - index
        return "HOMO" if delta == 0 else f"HOMO-{delta}"
    delta = index - lumo
    return "LUMO" if delta == 0 else f"LUMO+{delta}"


def canonical_orbital_diagnostics(
    mf: scf.hf.SCF,
    oxygen: int,
    carbonyl_carbon: int,
    normal: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    mol = mf.mol
    coeff = np.asarray(mf.mo_coeff)
    occupations = np.asarray(mf.mo_occ)
    energies = np.asarray(mf.mo_energy)
    overlap = mf.get_ovlp()
    irreps = orbital_symmetry_labels(mol, coeff)
    orth_coeff = meta_lowdin_coefficients(mol, coeff, overlap)
    weights = np.abs(orth_coeff) ** 2
    weights /= np.where(weights.sum(axis=0) > 0, weights.sum(axis=0), 1.0)

    ao_labels = mol.ao_labels(fmt=False)
    ao_atom_ids = np.asarray([int(label[0]) for label in ao_labels])
    oxygen_population = weights[ao_atom_ids == oxygen].sum(axis=0)
    carbonyl_normal_p = projected_p_population(
        mol,
        orth_coeff,
        [oxygen, carbonyl_carbon],
        normal,
    )
    oxygen_normal_p = projected_p_population(mol, orth_coeff, [oxygen], normal)
    carbon_normal_p = projected_p_population(
        mol, orth_coeff, [carbonyl_carbon], normal
    )
    centers, spreads = spatial_moments(mol, coeff)

    occupied = np.where(occupations > 0)[0]
    virtual = np.where(occupations == 0)[0]
    homo = int(occupied[-1])
    lumo = int(virtual[0])

    rows: List[Dict[str, Any]] = []
    for i in range(coeff.shape[1]):
        rows.append(
            {
                "mo_1based": i + 1,
                "frontier_label": frontier_label(i, homo, lumo),
                "occupation": float(occupations[i]),
                "energy_Eh": float(energies[i]),
                "energy_eV": float(energies[i] * HARTREE_TO_EV),
                "irrep": irreps[i],
                "oxygen_meta_lowdin_population": float(oxygen_population[i]),
                "carbonyl_normal_p_population": float(carbonyl_normal_p[i]),
                "oxygen_normal_p_population": float(oxygen_normal_p[i]),
                "carbon_normal_p_population": float(carbon_normal_p[i]),
                "spread_bohr2": float(spreads[i]),
                "center_x_bohr": float(centers[i, 0]),
                "center_y_bohr": float(centers[i, 1]),
                "center_z_bohr": float(centers[i, 2]),
                "selected_role": "",
            }
        )

    arrays = {
        "occupation": occupations,
        "energy": energies,
        "irrep": np.asarray(irreps, dtype=object),
        "oxygen_population": oxygen_population,
        "carbonyl_normal_p": carbonyl_normal_p,
        "spread": spreads,
    }
    return rows, arrays


def select_active_mos(
    mf: scf.hf.SCF,
    diagnostic_rows: List[Dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
    ground_irrep: str,
    target_irrep: str,
    manual_active_mos: List[int] | None,
    args: argparse.Namespace,
) -> Tuple[List[int], Dict[str, int]]:
    occupations = arrays["occupation"]
    energies = arrays["energy"]
    irreps = arrays["irrep"]
    oxygen_population = arrays["oxygen_population"]
    carbonyl_normal_p = arrays["carbonyl_normal_p"]
    nmo = len(occupations)

    if manual_active_mos is not None:
        zero_based = [value - 1 for value in manual_active_mos]
        if max(zero_based) >= nmo:
            raise RuntimeError(
                f"Manual active MO list {manual_active_mos} exceeds NMO={nmo}"
            )
        n_orb, pi_orb, pistar_orb = zero_based
    else:
        core_count = conventional_core_orbitals(mf.mol)
        occupied = [i for i in range(nmo) if occupations[i] > 0 and i >= core_count]
        virtual = [i for i in range(nmo) if occupations[i] == 0]

        n_candidates = [i for i in occupied if irreps[i] == ground_irrep]
        n_filtered = [
            i for i in n_candidates if oxygen_population[i] >= args.n_min_oxygen_population
        ]
        if n_filtered:
            # The relevant carbonyl n orbital is the highest-energy sufficiently
            # oxygen-centered occupied orbital of A' symmetry.
            n_orb = max(n_filtered, key=lambda i: energies[i])
        elif n_candidates:
            n_orb = max(n_candidates, key=lambda i: (oxygen_population[i], energies[i]))
            message = (
                "No A' occupied MO passed the oxygen-population threshold; "
                f"falling back to MO {n_orb + 1} with O population "
                f"{oxygen_population[n_orb]:.3f}."
            )
            if args.strict_active_selection:
                raise RuntimeError(message)
            print(f"WARNING: {message}", file=sys.stderr)
        else:
            raise RuntimeError(f"No occupied {ground_irrep} valence MOs were found")

        pi_candidates = [i for i in occupied if irreps[i] == target_irrep]
        if not pi_candidates:
            raise RuntimeError(f"No occupied {target_irrep} valence MOs were found")
        pi_orb = max(pi_candidates, key=lambda i: (carbonyl_normal_p[i], energies[i]))

        pistar_candidates = [i for i in virtual if irreps[i] == target_irrep]
        if not pistar_candidates:
            raise RuntimeError(f"No virtual {target_irrep} MOs were found")
        # Prioritize C=O normal-p character.  For very similar projections, the
        # lower-energy orbital is preferred over a high-energy polarization MO.
        pistar_orb = min(
            pistar_candidates,
            key=lambda i: (-carbonyl_normal_p[i], energies[i]),
        )

    selected = {"n": n_orb, "pi": pi_orb, "pi_star": pistar_orb}
    if len(set(selected.values())) != 3:
        raise RuntimeError(f"Active-orbital selector returned duplicates: {selected}")

    expected = {
        "n": (True, ground_irrep),
        "pi": (True, target_irrep),
        "pi_star": (False, target_irrep),
    }
    problems: List[str] = []
    for role, index in selected.items():
        should_be_occupied, expected_irrep = expected[role]
        is_occupied = bool(occupations[index] > 0)
        if is_occupied != should_be_occupied:
            problems.append(
                f"{role}=MO{index + 1} has occupation {occupations[index]:.1f}"
            )
        if str(irreps[index]) != expected_irrep:
            problems.append(
                f"{role}=MO{index + 1} has irrep {irreps[index]}, expected {expected_irrep}"
            )

    if carbonyl_normal_p[pi_orb] < args.pi_min_normal_p_population:
        problems.append(
            f"pi=MO{pi_orb + 1} has low projected C=O p-normal population "
            f"{carbonyl_normal_p[pi_orb]:.3f}"
        )
    if carbonyl_normal_p[pistar_orb] < args.pistar_min_normal_p_population:
        problems.append(
            f"pi*=MO{pistar_orb + 1} has low projected C=O p-normal population "
            f"{carbonyl_normal_p[pistar_orb]:.3f}"
        )
    if oxygen_population[n_orb] < args.n_min_oxygen_population:
        problems.append(
            f"n=MO{n_orb + 1} has low oxygen population "
            f"{oxygen_population[n_orb]:.3f}"
        )

    if problems:
        message = "Active-space validation warnings: " + "; ".join(problems)
        if args.strict_active_selection:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)

    for role, index in selected.items():
        diagnostic_rows[index]["selected_role"] = role

    # sort_mo preserves the order of the supplied CAS list.  We use n,pi,pi*.
    active_mos_1based = [n_orb + 1, pi_orb + 1, pistar_orb + 1]
    return active_mos_1based, selected


def prepare_active_space(
    mf: scf.hf.SCF,
    active_mos_1based: Sequence[int],
    ncas: int,
    nelecas_total: int,
) -> np.ndarray:
    mc_template = mcscf.CASSCF(mf, ncas, nelecas_total)
    # Keep PySCF's tagged-array orbital-symmetry metadata when available.
    return mcscf.sort_mo(
        mc_template, mf.mo_coeff, list(active_mos_1based), base=1
    )


# -----------------------------------------------------------------------------
# CASSCF, CASCI, and NEVPT2
# -----------------------------------------------------------------------------


def make_fci_solver(
    mol: gto.Mole,
    spin_2s: int,
    wfnsym: str,
    nroots: int,
    spin_penalty: float,
) -> Any:
    if spin_2s == 0:
        solver = fci.direct_spin0_symm.FCI(mol)
    else:
        solver = fci.direct_spin1_symm.FCI(mol)
    solver.spin = spin_2s
    solver.wfnsym = wfnsym
    solver.nroots = nroots
    if spin_2s > 0 and spin_penalty > 0:
        target_s = spin_2s / 2.0
        solver = fci.addons.fix_spin(
            solver,
            shift=spin_penalty,
            ss=target_s * (target_s + 1.0),
        )
        solver.spin = spin_2s
        solver.wfnsym = wfnsym
        solver.nroots = nroots
    return solver


def apply_state_average_mix(
    mc: Any, solvers: Sequence[Any], weights: Sequence[float]
) -> Any:
    if hasattr(mc, "state_average_mix_"):
        return mc.state_average_mix_(list(solvers), list(weights))
    return mcscf.state_average_mix_(mc, list(solvers), list(weights))


def run_state_averaged_casscf(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    mo_guess: np.ndarray,
    target: TargetState,
    ground_irrep: str,
    target_irrep: str,
    args: argparse.Namespace,
    state_dir: Path,
) -> Any:
    mc = mcscf.CASSCF(mf, ncas, nelecas_total)
    mc.chkfile = str(state_dir / "sa_casscf.chk")
    mc.conv_tol = args.casscf_conv_tol
    mc.conv_tol_grad = args.casscf_conv_tol_grad
    mc.max_cycle_macro = args.casscf_max_cycle_macro
    mc.max_memory = args.max_memory

    gs_solver = make_fci_solver(
        mf.mol,
        spin_2s=0,
        wfnsym=ground_irrep,
        nroots=1,
        spin_penalty=args.spin_penalty,
    )
    ex_solver = make_fci_solver(
        mf.mol,
        spin_2s=target.spin_2s,
        wfnsym=target_irrep,
        nroots=1,
        spin_penalty=args.spin_penalty,
    )
    mc = apply_state_average_mix(mc, [gs_solver, ex_solver], [0.5, 0.5])

    mc.kernel(mo_guess)
    if not mc.converged:
        raise RuntimeError(f"SA-CASSCF did not converge for {target.label}")
    return mc


def run_symmetry_casci(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    common_mo: np.ndarray,
    spin_2s: int,
    wfnsym: str,
    nroots: int,
    spin_penalty: float,
    chkfile: Path,
) -> Any:
    active_nelec = normalize_active_nelec(nelecas_total, spin_2s)
    ci_mc = mcscf.CASCI(mf, ncas, active_nelec)
    ci_mc.chkfile = str(chkfile)
    ci_mc.max_memory = mf.max_memory
    ci_mc.fcisolver = make_fci_solver(
        mf.mol,
        spin_2s,
        wfnsym,
        nroots,
        spin_penalty,
    )
    ci_mc.kernel(common_mo)
    convergence = getattr(ci_mc, "converged", True)
    if not bool(np.all(np.asarray(convergence))):
        raise RuntimeError(
            f"CASCI did not converge for spin 2S={spin_2s}, symmetry {wfnsym}"
        )
    return ci_mc


def nevpt2_total_energy(mc: Any, root: int) -> Tuple[float, float, float]:
    reference_energy = root_value(mc.e_tot, root)
    correction = float(mrpt.NEVPT(mc, root=root).kernel())
    return reference_energy, correction, reference_energy + correction


# -----------------------------------------------------------------------------
# CI/configuration and active-orbital analysis
# -----------------------------------------------------------------------------


def spatial_configuration_weights(
    ci_vector: np.ndarray,
    ncas: int,
    nelec: Tuple[int, int],
    determinant_cutoff: float = 0.0,
) -> List[Tuple[Tuple[int, ...], float]]:
    nalpha, nbeta = nelec
    alpha_occs = np.asarray(cistring.gen_occslst(range(ncas), nalpha), dtype=int)
    beta_occs = np.asarray(cistring.gen_occslst(range(ncas), nbeta), dtype=int)
    ci = np.asarray(ci_vector).reshape(len(alpha_occs), len(beta_occs))

    grouped: MutableMapping[Tuple[int, ...], float] = defaultdict(float)
    for ia, alpha_occ in enumerate(alpha_occs):
        for ib, beta_occ in enumerate(beta_occs):
            weight = float(abs(ci[ia, ib]) ** 2)
            if weight <= determinant_cutoff:
                continue
            spatial_occ = np.zeros(ncas, dtype=int)
            spatial_occ[alpha_occ] += 1
            spatial_occ[beta_occ] += 1
            grouped[tuple(int(x) for x in spatial_occ)] += weight

    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def occupation_text(occupation: Sequence[int]) -> str:
    return "[" + " ".join(str(int(x)) for x in occupation) + "]"


def single_excitation_configurations(
    ground_occupation: Sequence[int],
    target_configurations: Sequence[Tuple[Tuple[int, ...], float]],
) -> List[Tuple[Tuple[int, ...], float, int, int]]:
    ground = np.asarray(ground_occupation, dtype=int)
    results: List[Tuple[Tuple[int, ...], float, int, int]] = []
    for occupation, weight in target_configurations:
        delta = np.asarray(occupation, dtype=int) - ground
        if np.sum(np.abs(delta)) != 2:
            continue
        holes = np.where(delta == -1)[0]
        particles = np.where(delta == 1)[0]
        if len(holes) == 1 and len(particles) == 1:
            results.append(
                (occupation, float(weight), int(holes[0]), int(particles[0]))
            )
    return results


def active_natural_occupations(
    mc: Any, ci_vector: np.ndarray, nelec: Tuple[int, int]
) -> np.ndarray:
    dm1 = np.asarray(mc.fcisolver.make_rdm1(ci_vector, mc.ncas, nelec))
    dm1 = 0.5 * (dm1 + dm1.T.conj())
    return np.linalg.eigvalsh(dm1).real[::-1]


def state_spin_square(
    mc: Any, ci_vector: np.ndarray, nelec: Tuple[int, int]
) -> Tuple[float, float]:
    try:
        ss, multiplicity = mc.fcisolver.spin_square(ci_vector, mc.ncas, nelec)
        return float(ss), float(multiplicity)
    except Exception:
        return math.nan, math.nan


def active_orbital_irreps(mol: gto.Mole, active_coeff: np.ndarray) -> List[str]:
    try:
        labels = symm.label_orb_symm(
            mol,
            mol.irrep_name,
            mol.symm_orb,
            active_coeff,
            check=False,
        )
        return [str(label) for label in labels]
    except Exception as exc:
        return [f"UNLABELED:{type(exc).__name__}"] * active_coeff.shape[1]


def write_configuration_table(
    configurations: Sequence[Tuple[Tuple[int, ...], float]],
    path: Path,
    maximum: int,
) -> None:
    rows = [
        {
            "rank": rank,
            "active_spatial_occupation": occupation_text(occupation),
            "aggregated_weight": weight,
        }
        for rank, (occupation, weight) in enumerate(configurations[:maximum], start=1)
    ]
    write_csv(rows, path)


def write_single_excitation_table(
    configurations: Sequence[Tuple[Tuple[int, ...], float, int, int]],
    active_irreps: Sequence[str],
    inactive_count: int,
    path: Path,
) -> None:
    rows = []
    for rank, (occupation, weight, hole, particle) in enumerate(configurations, start=1):
        rows.append(
            {
                "rank": rank,
                "active_spatial_occupation": occupation_text(occupation),
                "aggregated_weight": weight,
                "hole_active_orbital_1based": hole + 1,
                "hole_full_mo_1based": inactive_count + hole + 1,
                "hole_irrep": active_irreps[hole],
                "particle_active_orbital_1based": particle + 1,
                "particle_full_mo_1based": inactive_count + particle + 1,
                "particle_irrep": active_irreps[particle],
            }
        )
    write_csv(rows, path)


def write_natural_occupation_table(
    ground_noons: Sequence[float],
    target_noons: Sequence[float],
    path: Path,
) -> None:
    rows = [
        {
            "natural_orbital_rank": i + 1,
            "ground_occupation": float(ground_noons[i]),
            "target_occupation": float(target_noons[i]),
        }
        for i in range(len(ground_noons))
    ]
    write_csv(rows, path)


# -----------------------------------------------------------------------------
# Orbital files and plots
# -----------------------------------------------------------------------------


def inferred_bonds(mol: gto.Mole) -> List[Tuple[int, int]]:
    coords_ang = np.asarray(mol.atom_coords()) * ANGSTROM_PER_BOHR
    bonds: List[Tuple[int, int]] = []
    for i in range(mol.natm):
        sym_i = mol.atom_symbol(i)
        ri = COVALENT_RADII_ANGSTROM.get(sym_i, 0.75)
        for j in range(i + 1, mol.natm):
            sym_j = mol.atom_symbol(j)
            rj = COVALENT_RADII_ANGSTROM.get(sym_j, 0.75)
            distance = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
            cutoff = 1.25 * (ri + rj) + 0.10
            if distance <= cutoff:
                bonds.append((i, j))
    return bonds


def plot_orbital_slice(
    mol: gto.Mole,
    coeff: np.ndarray,
    path: Path,
    title: str,
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    normal: np.ndarray,
    offset_bohr: float,
    grid_size: int,
    margin_bohr: float,
) -> None:
    if plt is None:
        raise RuntimeError("Matplotlib is not installed; use --no-plots or install it")

    atom_coords = np.asarray(mol.atom_coords())
    relative = atom_coords - center
    atom_u = relative @ u_axis
    atom_v = relative @ v_axis

    u_min, u_max = atom_u.min() - margin_bohr, atom_u.max() + margin_bohr
    v_min, v_max = atom_v.min() - margin_bohr, atom_v.max() + margin_bohr
    u_values = np.linspace(u_min, u_max, grid_size)
    v_values = np.linspace(v_min, v_max, grid_size)
    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")

    points = (
        center[None, :]
        + offset_bohr * normal[None, :]
        + uu.reshape(-1, 1) * u_axis[None, :]
        + vv.reshape(-1, 1) * v_axis[None, :]
    )
    ao = mol.eval_gto("GTOval_sph", points)
    values = (ao @ coeff).reshape(grid_size, grid_size)

    abs_values = np.abs(values)
    vmax = float(np.quantile(abs_values, 0.995))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = float(abs_values.max())
    if vmax < 1e-12:
        raise RuntimeError(
            f"Orbital values are numerically zero on slice offset {offset_bohr:.3f} bohr"
        )

    levels = np.linspace(-vmax, vmax, 41)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    contour = ax.contourf(uu, vv, values, levels=levels, cmap="seismic", extend="both")
    ax.contour(uu, vv, values, levels=[0.0], colors="black", linewidths=0.6)

    for i, j in inferred_bonds(mol):
        ax.plot(
            [atom_u[i], atom_u[j]],
            [atom_v[i], atom_v[j]],
            linewidth=1.3,
            color="black",
        )
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        ax.scatter([atom_u[ia]], [atom_v[ia]], s=30, color="black", zorder=3)
        ax.text(atom_u[ia] + 0.08, atom_v[ia] + 0.08, symbol, fontsize=9)

    fig.colorbar(contour, ax=ax, label="MO amplitude")
    ax.set_xlabel("C=O-axis coordinate / bohr")
    ax.set_ylabel("in-plane perpendicular coordinate / bohr")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nslice offset = {offset_bohr:.2f} bohr")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_orbital_outputs(
    mol: gto.Mole,
    common_mo: np.ndarray,
    inactive_count: int,
    active_index: int,
    irrep: str,
    role: str,
    state_dir: Path,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    preferred_offset: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    full_index = inactive_count + active_index
    coeff = np.asarray(common_mo[:, full_index])
    stem = f"{role}_active{active_index + 1}_mo{full_index + 1}_{slugify(irrep)}"
    cube_path = state_dir / f"{stem}.cube"
    png_path = state_dir / f"{stem}.png"

    cube_written = False
    png_written = False
    cube_error: str | None = None
    png_error: str | None = None

    if args.cubes:
        try:
            cubegen.orbital(
                mol,
                str(cube_path),
                coeff,
                resolution=args.cube_resolution,
                margin=args.cube_margin,
            )
            cube_written = True
        except Exception as exc:
            cube_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: could not write {role} cube: {cube_error}", file=sys.stderr)

    if args.plots:
        try:
            center, u_axis, v_axis, normal = frame
            plot_orbital_slice(
                mol,
                coeff,
                png_path,
                title=(
                    f"{role.capitalize()} orbital: active {active_index + 1}, "
                    f"full MO {full_index + 1}, {irrep}"
                ),
                center=center,
                u_axis=u_axis,
                v_axis=v_axis,
                normal=normal,
                offset_bohr=preferred_offset,
                grid_size=args.slice_grid,
                margin_bohr=args.slice_margin,
            )
            png_written = True
        except Exception as exc:
            png_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: could not write {role} PNG: {png_error}", file=sys.stderr)

    return {
        f"{role}_active_orbital_1based": active_index + 1,
        f"{role}_full_mo_1based": full_index + 1,
        f"{role}_irrep": irrep,
        f"{role}_slice_offset_bohr": preferred_offset,
        f"{role}_cube": str(cube_path) if cube_written else None,
        f"{role}_cube_error": cube_error,
        f"{role}_png": str(png_path) if png_written else None,
        f"{role}_png_error": png_error,
    }


# -----------------------------------------------------------------------------
# Per-state driver
# -----------------------------------------------------------------------------


def run_target_state(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    mo_guess: np.ndarray,
    target: TargetState,
    ground_irrep: str,
    target_irrep: str,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    outdir: Path,
) -> Dict[str, Any]:
    state_dir = outdir / target.key
    state_dir.mkdir(parents=True, exist_ok=True)

    sa_mc = run_state_averaged_casscf(
        mf,
        ncas,
        nelecas_total,
        mo_guess,
        target,
        ground_irrep,
        target_irrep,
        args,
        state_dir,
    )
    # Keep the tagged MO array for the subsequent symmetry-resolved CASCI jobs.
    common_mo = sa_mc.mo_coeff
    inactive_count = int(sa_mc.ncore)
    active_coeff = np.asarray(common_mo[:, inactive_count : inactive_count + ncas])
    active_irreps = active_orbital_irreps(mf.mol, active_coeff)

    try:
        molden.from_mo(
            mf.mol,
            str(state_dir / "sa_casscf_orbitals.molden"),
            common_mo,
        )
    except Exception as exc:
        print(
            f"WARNING: could not write Molden file for {target.key}: {exc}",
            file=sys.stderr,
        )

    active_rows = [
        {
            "active_orbital_1based": i + 1,
            "full_mo_1based": inactive_count + i + 1,
            "irrep": active_irreps[i],
        }
        for i in range(ncas)
    ]
    write_csv(active_rows, state_dir / "active_orbitals.csv")

    ground_mc = run_symmetry_casci(
        mf,
        ncas,
        nelecas_total,
        common_mo,
        spin_2s=0,
        wfnsym=ground_irrep,
        nroots=1,
        spin_penalty=args.spin_penalty,
        chkfile=state_dir / "ground_casci.chk",
    )
    target_mc = run_symmetry_casci(
        mf,
        ncas,
        nelecas_total,
        common_mo,
        spin_2s=target.spin_2s,
        wfnsym=target_irrep,
        nroots=1,
        spin_penalty=args.spin_penalty,
        chkfile=state_dir / "target_casci.chk",
    )
    ground_root = 0
    target_root = target.target_root

    ground_ref, ground_corr, ground_total = nevpt2_total_energy(ground_mc, ground_root)
    target_ref, target_corr, target_total = nevpt2_total_energy(target_mc, target_root)

    ground_nelec = normalize_active_nelec(nelecas_total, 0)
    target_nelec = normalize_active_nelec(nelecas_total, target.spin_2s)
    ground_ci = root_ci(ground_mc, ground_root)
    target_ci = root_ci(target_mc, target_root)

    ground_configs = spatial_configuration_weights(ground_ci, ncas, ground_nelec)
    target_configs = spatial_configuration_weights(target_ci, ncas, target_nelec)
    write_configuration_table(
        ground_configs,
        state_dir / "ground_spatial_configurations.csv",
        args.max_configurations,
    )
    write_configuration_table(
        target_configs,
        state_dir / "target_spatial_configurations.csv",
        args.max_configurations,
    )

    if not ground_configs or not target_configs:
        raise RuntimeError("No CI configurations were extracted")

    ground_occ, ground_weight = ground_configs[0]
    target_occ, target_weight = target_configs[0]
    single_excitations = single_excitation_configurations(ground_occ, target_configs)
    if not single_excitations:
        raise RuntimeError(
            "No one-electron spatial configuration was identified relative to "
            f"the dominant ground configuration {occupation_text(ground_occ)}"
        )
    single_occ, single_weight, hole_active, particle_active = single_excitations[0]
    total_single_weight = float(sum(item[1] for item in single_excitations))
    write_single_excitation_table(
        single_excitations,
        active_irreps,
        inactive_count,
        state_dir / "single_excitation_configurations.csv",
    )

    ground_noons = active_natural_occupations(ground_mc, ground_ci, ground_nelec)
    target_noons = active_natural_occupations(target_mc, target_ci, target_nelec)
    write_natural_occupation_table(
        ground_noons,
        target_noons,
        state_dir / "active_natural_occupations.csv",
    )

    ss, effective_multiplicity = state_spin_square(target_mc, target_ci, target_nelec)
    dominant_pair_matches_expected_npi = (
        active_irreps[hole_active] == ground_irrep
        and active_irreps[particle_active] == target_irrep
    )
    if not dominant_pair_matches_expected_npi:
        print(
            "WARNING: dominant single-excitation pair does not have the expected "
            f"{ground_irrep} -> {target_irrep} symmetry assignment: "
            f"hole={active_irreps[hole_active]}, "
            f"particle={active_irreps[particle_active]}",
            file=sys.stderr,
        )

    row: Dict[str, Any] = {
        **asdict(target),
        "basis": args.basis,
        "ground_wfnsym": ground_irrep,
        "target_wfnsym": target_irrep,
        "nevpt2_contraction": "strongly contracted (PySCF mrpt.NEVPT)",
        "active_space_electrons": nelecas_total,
        "active_space_orbitals": ncas,
        "inactive_doubly_occupied_orbitals": inactive_count,
        "sa_casscf_converged": bool(sa_mc.converged),
        "ground_casci_Eh": ground_ref,
        "target_casci_Eh": target_ref,
        "ground_sc_nevpt2_correction_Eh": ground_corr,
        "target_sc_nevpt2_correction_Eh": target_corr,
        "ground_sc_nevpt2_total_Eh": ground_total,
        "target_sc_nevpt2_total_Eh": target_total,
        "vertical_casci_eV": (target_ref - ground_ref) * HARTREE_TO_EV,
        "vertical_sc_nevpt2_eV": (target_total - ground_total) * HARTREE_TO_EV,
        "difference_from_paper_tbe_eV": (
            (target_total - ground_total) * HARTREE_TO_EV
            - target.paper_tbe_augccpvtz_eV
        ),
        "difference_from_paper_pc_nevpt2_eV": (
            (target_total - ground_total) * HARTREE_TO_EV
            - target.paper_pc_nevpt2_augccpvtz_eV
        ),
        "target_S2": ss,
        "target_effective_multiplicity": effective_multiplicity,
        "dominant_ground_spatial_configuration": occupation_text(ground_occ),
        "dominant_ground_configuration_weight": ground_weight,
        "dominant_target_spatial_configuration": occupation_text(target_occ),
        "dominant_target_configuration_weight": target_weight,
        "dominant_single_excitation_configuration": occupation_text(single_occ),
        "dominant_single_excitation_weight": single_weight,
        "dominant_pair_matches_expected_n_to_pi_star_symmetry": dominant_pair_matches_expected_npi,
        "total_single_excitation_weight_relative_to_dominant_ground_configuration": total_single_weight,
        "ground_active_natural_occupations": [float(x) for x in ground_noons],
        "target_active_natural_occupations": [float(x) for x in target_noons],
    }

    # Even (A') orbitals are informative in the C_s plane.  Odd (A'') orbitals
    # vanish there and are plotted on a displaced parallel slice.  This dynamic
    # choice also keeps plotting sensible if the dominant pair differs from the
    # expected n -> pi* assignment.
    hole_offset = 0.0 if active_irreps[hole_active] == ground_irrep else args.slice_offset
    particle_offset = (
        0.0 if active_irreps[particle_active] == ground_irrep else args.slice_offset
    )
    row.update(
        write_orbital_outputs(
            mf.mol,
            common_mo,
            inactive_count,
            hole_active,
            active_irreps[hole_active],
            "hole",
            state_dir,
            frame,
            preferred_offset=hole_offset,
            args=args,
        )
    )
    row.update(
        write_orbital_outputs(
            mf.mol,
            common_mo,
            inactive_count,
            particle_active,
            active_irreps[particle_active],
            "particle",
            state_dir,
            frame,
            preferred_offset=particle_offset,
            args=args,
        )
    )

    (state_dir / "state_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n=== Acetaldehyde n->pi* summary ===")
    header = (
        "State",
        "CASCI/eV",
        "SC-NEVPT2/eV",
        "Paper TBE/eV",
        "Paper PC-NEVPT2/eV",
        "Sum single wt",
        "Top pair wt",
        "Hole",
        "Particle",
        "<S^2>",
    )
    print("  ".join(header))
    for row in rows:
        if row.get("status") == "ERROR":
            print(f"{row.get('label', row.get('key'))}  ERROR: {row.get('error')}")
            continue
        values = (
            str(row["label"]),
            f"{float(row['vertical_casci_eV']):.6f}",
            f"{float(row['vertical_sc_nevpt2_eV']):.6f}",
            f"{float(row['paper_tbe_augccpvtz_eV']):.2f}",
            f"{float(row['paper_pc_nevpt2_augccpvtz_eV']):.2f}",
            f"{float(row['total_single_excitation_weight_relative_to_dominant_ground_configuration']):.4f}",
            f"{float(row['dominant_single_excitation_weight']):.4f}",
            f"a{int(row['hole_active_orbital_1based'])}/MO{int(row['hole_full_mo_1based'])}",
            f"a{int(row['particle_active_orbital_1based'])}/MO{int(row['particle_full_mo_1based'])}",
            f"{float(row['target_S2']):.6f}",
        )
        print("  ".join(values))


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def parse_state_selection(text: str) -> List[TargetState]:
    if text.strip().lower() == "all":
        keys = list(DEFAULT_STATE_ORDER)
    else:
        keys = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = [key for key in keys if key not in TARGET_STATES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown states {unknown}; choose from {sorted(TARGET_STATES)} or 'all'"
        )
    return [TARGET_STATES[key] for key in keys]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize acetaldehyde's lowest singlet and triplet n->pi* "
            "states with SA-CASSCF/CASCI and PySCF strongly contracted NEVPT2."
        )
    )
    parser.add_argument("coordinates", type=Path, help="Numbered Coordinates file")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    parser.add_argument(
        "--states",
        default="all",
        help="Comma-separated selection: singlet-app,triplet-app; default: all",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("acetaldehyde_sc_nevpt2"),
    )
    parser.add_argument("--max-memory", type=int, default=4000)
    parser.add_argument("--verbose", type=int, default=4)

    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument(
        "--newton-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry unconverged RHF with Newton/SOSCF",
    )

    parser.add_argument(
        "--active-mos",
        default=None,
        help=(
            "Manual comma-separated 1-based canonical MO indices in the order "
            "n,pi,pi*. Default: automatic symmetry/population selection"
        ),
    )
    parser.add_argument("--expected-ncas", type=int, default=3)
    parser.add_argument("--expected-nelecas", type=int, default=4)
    parser.add_argument("--n-min-oxygen-population", type=float, default=0.30)
    parser.add_argument("--pi-min-normal-p-population", type=float, default=0.20)
    parser.add_argument("--pistar-min-normal-p-population", type=float, default=0.12)
    parser.add_argument(
        "--strict-active-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the automatically/manual selected n,pi,pi* orbitals fail checks",
    )

    parser.add_argument("--casscf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--casscf-conv-tol-grad", type=float, default=1e-5)
    parser.add_argument("--casscf-max-cycle-macro", type=int, default=100)
    parser.add_argument(
        "--spin-penalty",
        type=float,
        default=0.20,
        help="FCI S^2 penalty for triplet targeting; set 0 to disable",
    )
    parser.add_argument("--max-configurations", type=int, default=20)

    parser.add_argument(
        "--cubes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write cube files for the dominant hole and particle orbitals",
    )
    parser.add_argument("--cube-resolution", type=float, default=0.20)
    parser.add_argument("--cube-margin", type=float, default=3.0)
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write signed 2-D orbital-slice PNGs",
    )
    parser.add_argument(
        "--slice-offset",
        type=float,
        default=0.70,
        help="Offset from the C_s plane for the pi* particle slice, in bohr",
    )
    parser.add_argument("--slice-grid", type=int, default=301)
    parser.add_argument("--slice-margin", type=float, default=2.5)
    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    selected_states = parse_state_selection(args.states)
    manual_active_mos = parse_active_mos(args.active_mos)
    args.outdir.mkdir(parents=True, exist_ok=True)

    molecule_index, molecule_name, atoms = find_named_molecule(
        parse_coordinate_file(args.coordinates), "Acetaldehyde"
    )
    mol = build_molecule(
        atoms,
        basis=args.basis,
        unit=args.unit,
        max_memory=args.max_memory,
        log_path=args.outdir / "pyscf.log",
        verbose=args.verbose,
    )
    ground_irrep, target_irrep = resolve_cs_irreps(mol)
    oxygen, carbonyl_carbon, methyl_carbon = identify_carbonyl_atoms(mol)
    frame = carbonyl_frame(mol, oxygen, carbonyl_carbon, methyl_carbon)
    normal = frame[3]

    mf = run_rhf(mol, args, args.outdir)
    try:
        molden.from_scf(mf, str(args.outdir / "rhf_canonical_orbitals.molden"))
    except Exception as exc:
        print(f"WARNING: could not write RHF Molden file: {exc}", file=sys.stderr)

    diagnostic_rows, arrays = canonical_orbital_diagnostics(
        mf,
        oxygen,
        carbonyl_carbon,
        normal,
    )
    diagnostic_path = args.outdir / "orbital_selection_diagnostics.csv"
    # Write the unmarked table first so it remains available even if a strict
    # automatic-selection check aborts below.
    write_csv(diagnostic_rows, diagnostic_path)
    active_mos_1based, selected_roles = select_active_mos(
        mf,
        diagnostic_rows,
        arrays,
        ground_irrep,
        target_irrep,
        manual_active_mos,
        args,
    )
    # Rewrite after selection to add the n/pi/pi* role labels.
    write_csv(diagnostic_rows, diagnostic_path)

    ncas = int(args.expected_ncas)
    nelecas_total = int(args.expected_nelecas)
    if (ncas, nelecas_total) != (3, 4):
        raise RuntimeError(
            "This script's n/pi/pi* logic assumes CAS(4,3). Keep "
            "--expected-ncas 3 and --expected-nelecas 4, or modify the script."
        )
    mo_guess = prepare_active_space(mf, active_mos_1based, ncas, nelecas_total)

    metadata = {
        "molecule_index": molecule_index,
        "molecule": molecule_name,
        "formula": hill_formula(atoms),
        "basis": args.basis,
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "unit": args.unit,
        "point_group": str(mol.groupname),
        "ground_irrep": ground_irrep,
        "target_irrep": target_irrep,
        "electrons": int(mol.nelectron),
        "nao_nmo": int(mol.nao_nr()),
        "rhf_energy_Eh": float(mf.e_tot),
        "oxygen_atom_1based": oxygen + 1,
        "carbonyl_carbon_atom_1based": carbonyl_carbon + 1,
        "methyl_carbon_atom_1based": methyl_carbon + 1,
        "symmetry_plane_normal": [float(x) for x in normal],
        "active_space_electrons": nelecas_total,
        "active_space_orbitals": ncas,
        "active_mos_1based_order_n_pi_pistar": active_mos_1based,
        "active_selection_mode": "manual" if manual_active_mos else "automatic",
        "selected_roles_zero_based": selected_roles,
        "states": [state.key for state in selected_states],
        "method_note": (
            "PySCF strongly contracted NEVPT2; paper main-table values are "
            "partially contracted NEVPT2/aug-cc-pVTZ"
        ),
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        f"Acetaldehyde: basis={args.basis}, point_group={mol.groupname}, "
        f"irreps=({ground_irrep},{target_irrep}), NMO={mol.nao_nr()}, "
        f"CAS({nelecas_total},{ncas})"
    )
    print(
        "Selected canonical MOs, order n,pi,pi*: "
        + ", ".join(f"MO{i}" for i in active_mos_1based)
    )
    for role, mo_index in selected_roles.items():
        row = diagnostic_rows[mo_index]
        print(
            f"  {role:7s} MO{mo_index + 1:3d} "
            f"occ={row['occupation']:.1f} irrep={row['irrep']} "
            f"E={row['energy_eV']:.3f} eV "
            f"O-pop={row['oxygen_meta_lowdin_population']:.3f} "
            f"C=O-pnormal={row['carbonyl_normal_p_population']:.3f}"
        )
    print(
        "NOTE: this is PySCF SC-NEVPT2. The paper's main NEVPT2 values are "
        "PC-NEVPT2/aug-cc-pVTZ."
    )

    results: List[Dict[str, Any]] = []
    for state in selected_states:
        print(f"\n=== {state.label} ===", flush=True)
        try:
            result = run_target_state(
                mf,
                ncas,
                nelecas_total,
                mo_guess,
                state,
                ground_irrep,
                target_irrep,
                frame,
                args,
                args.outdir,
            )
            result["status"] = "OK"
            results.append(result)
            print(
                f"Vertical SC-NEVPT2 energy: "
                f"{result['vertical_sc_nevpt2_eV']:.6f} eV; "
                f"summed single-excitation weight: "
                f"{result['total_single_excitation_weight_relative_to_dominant_ground_configuration']:.4f}; "
                f"top pair weight: {result['dominant_single_excitation_weight']:.4f}"
            )
        except Exception as exc:
            error_row: Dict[str, Any] = {
                **asdict(state),
                "basis": args.basis,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            results.append(error_row)
            print(error_row["error"], file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

    write_csv(results, args.outdir / "acetaldehyde_vertical_energies.csv")
    (args.outdir / "acetaldehyde_vertical_energies.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print_summary(results)
    print(f"\nResults written below: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
