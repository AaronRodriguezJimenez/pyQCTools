#!/usr/bin/env python3
"""Furan valence-excitation audit with SA-CASSCF and PySCF SC-NEVPT2.

This script is the next proof-of-concept step after the RHF orbital audit.  It
reads the supplied numbered ``Coordinates`` file, extracts furan, and targets
four valence pi -> pi* states discussed in Sarkar et al., JCTC 2022:

    1B2, 1A1, 3B2, and 3A1.

The default STO-3G calculation intentionally provides a small proof of concept.
The active space is generated with AVAS from the five out-of-plane 2px valence
AOs of the four C atoms and the O atom.  With the supplied orientation (the
molecular plane is x=0), this should produce a CAS(6e,5o) pi space.

For every targeted excited state the workflow is:

  1. RHF on the ground-state geometry.
  2. AVAS construction of the pi active space.
  3. A two-state state-averaged CASSCF containing the singlet A1 ground state
     and the excited state of interest.  For the excited singlet A1 state, the
     first two singlet A1 roots are averaged.
  4. A separate symmetry- and spin-resolved CASCI calculation in the optimized
     state-averaged orbitals.  PySCF requires this step because its NEVPT2
     driver does not accept the state-average FCI-solver wrapper directly.
  5. State-specific strongly contracted NEVPT2 corrections for the ground and
     excited states, followed by the vertical excitation energy.
  6. CI configuration analysis.  Spatial configurations are formed by summing
     |C_I|^2 over spin determinants with the same spatial occupation.  The
     total weight of all one-electron promotions relative to the dominant
     ground configuration is reported, together with the highest-weight
     hole/particle pair.  This is an orbital-basis-dependent CAS diagnostic,
     not the CC3 %T1 diagnostic used in some benchmark databases.
  7. Cube files and 2-D signed orbital slices for the identified hole and
     particle orbitals.

Important scope limitations
---------------------------
* PySCF's built-in ``mrpt.NEVPT`` is strongly contracted NEVPT2 (SC-NEVPT2).
  The main table in the paper reports partially contracted NEVPT2 (PC-NEVPT2).
* STO-3G has no diffuse Rydberg space.  Therefore this script deliberately
  targets only the four valence pi -> pi* states, not the 3s/3p Rydberg states.
* The paper used aug-cc-pVTZ and active spaces tailored to each transition.
  CAS(6,5)/STO-3G is a controlled proof of concept, not a numerical
  reproduction of the paper.

Example
-------

    python 04_furan_nevpt2_valence.py Coordinates \
        --basis sto-3g \
        --outdir furan_sc_nevpt2_sto3g

Later, the same valence-only workflow can be tested with a larger basis:

    python 04_furan_nevpt2_valence.py Coordinates \
        --basis aug-cc-pvtz \
        --outdir furan_sc_nevpt2_avtz

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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Plots can be disabled, but cube files still require PySCF.
    plt = None

try:
    import pyscf
    from pyscf import fci, gto, mcscf, mrpt, scf, symm
    from pyscf.fci import cistring
    from pyscf.mcscf import avas
    from pyscf.tools import cubegen, molden
except ImportError as exc:  # Give a clearer message than a deep import failure.
    raise SystemExit(
        "PySCF is required. Install it in the environment used to run this script, "
        "for example with `python -m pip install pyscf matplotlib`."
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

Atom = Tuple[str, Tuple[float, float, float]]
MoleculeRecord = Tuple[int, str, List[Atom]]


@dataclass(frozen=True)
class TargetState:
    key: str
    label: str
    multiplicity: int
    spin_2s: int
    wfnsym: str
    target_root: int
    paper_tbe_augccpvtz_eV: float
    paper_pc_nevpt2_augccpvtz_eV: float

    @property
    def is_excited_a1_singlet(self) -> bool:
        return self.spin_2s == 0 and self.wfnsym == "A1" and self.target_root == 1


# The values are from Tables 1-3 of the uploaded paper.  The PySCF values
# produced by this script are SC-NEVPT2 and, by default, STO-3G; direct numerical
# agreement with these aug-cc-pVTZ PC-NEVPT2 values is therefore not expected.
TARGET_STATES: Mapping[str, TargetState] = {
    "singlet-b2": TargetState(
        key="singlet-b2",
        label="1B2 (V; pi->pi*)",
        multiplicity=1,
        spin_2s=0,
        wfnsym="B2",
        target_root=0,
        paper_tbe_augccpvtz_eV=6.37,
        paper_pc_nevpt2_augccpvtz_eV=6.20,
    ),
    "singlet-a1": TargetState(
        key="singlet-a1",
        label="1A1 (V; pi->pi*)",
        multiplicity=1,
        spin_2s=0,
        wfnsym="A1",
        target_root=1,
        paper_tbe_augccpvtz_eV=6.56,
        paper_pc_nevpt2_augccpvtz_eV=6.77,
    ),
    "triplet-b2": TargetState(
        key="triplet-b2",
        label="3B2 (V; pi->pi*)",
        multiplicity=3,
        spin_2s=2,
        wfnsym="B2",
        target_root=0,
        paper_tbe_augccpvtz_eV=4.20,
        paper_pc_nevpt2_augccpvtz_eV=4.42,
    ),
    "triplet-a1": TargetState(
        key="triplet-a1",
        label="3A1 (V; pi->pi*)",
        multiplicity=3,
        spin_2s=2,
        wfnsym="A1",
        target_root=0,
        paper_tbe_augccpvtz_eV=5.46,
        paper_pc_nevpt2_augccpvtz_eV=5.60,
    ),
}

DEFAULT_STATE_ORDER = ("triplet-b2", "triplet-a1", "singlet-b2", "singlet-a1")

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
    """Parse the numbered molecule blocks used by the supplied Coordinates file."""
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

        # Permit descriptive text before the first molecule block.
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
    # A multi-root FCI vector is normally a list.  This fallback covers versions
    # or solvers that stack roots in the first axis.
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


# -----------------------------------------------------------------------------
# RHF, AVAS, CASSCF, CASCI, and NEVPT2
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


def construct_pi_active_space(
    mf: scf.hf.SCF, args: argparse.Namespace
) -> Tuple[int, int, np.ndarray]:
    ncas, nelecas, mo_guess = avas.avas(
        mf,
        list(args.avas_labels),
        threshold=args.avas_threshold,
        minao=args.avas_minao,
        with_iao=args.avas_with_iao,
        canonicalize=True,
        ncore=args.avas_excluded_core,
    )
    nelecas_total = sum(nelecas) if isinstance(nelecas, (tuple, list)) else int(nelecas)

    if (ncas, nelecas_total) != (args.expected_ncas, args.expected_nelecas):
        message = (
            "AVAS did not produce the expected furan pi space. "
            f"Expected CAS({args.expected_nelecas},{args.expected_ncas}); "
            f"obtained CAS({nelecas_total},{ncas}). "
            "Inspect the AVAS log and AO labels before continuing."
        )
        if args.strict_active_space:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)

    return int(ncas), int(nelecas_total), np.asarray(mo_guess)


def make_fci_solver(
    mol: gto.Mole,
    spin_2s: int,
    wfnsym: str,
    nroots: int,
) -> Any:
    if spin_2s == 0:
        solver = fci.direct_spin0_symm.FCI(mol)
    else:
        solver = fci.direct_spin1_symm.FCI(mol)
    solver.spin = spin_2s
    solver.wfnsym = wfnsym
    solver.nroots = nroots
    return solver


def run_state_averaged_casscf(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    mo_guess: np.ndarray,
    target: TargetState,
    args: argparse.Namespace,
    state_dir: Path,
) -> Any:
    mc = mcscf.CASSCF(mf, ncas, nelecas_total)
    mc.chkfile = str(state_dir / "sa_casscf.chk")
    mc.conv_tol = args.casscf_conv_tol
    mc.conv_tol_grad = args.casscf_conv_tol_grad
    mc.max_cycle_macro = args.casscf_max_cycle_macro
    mc.max_memory = args.max_memory

    if target.is_excited_a1_singlet:
        # Ground and excited A1 are roots 0 and 1 of the same symmetry/spin block.
        solver = make_fci_solver(mf.mol, spin_2s=0, wfnsym="A1", nroots=2)
        mc.fcisolver = solver
        mc = mc.state_average_([0.5, 0.5])
    else:
        gs_solver = make_fci_solver(mf.mol, spin_2s=0, wfnsym="A1", nroots=1)
        ex_solver = make_fci_solver(
            mf.mol,
            spin_2s=target.spin_2s,
            wfnsym=target.wfnsym,
            nroots=1,
        )
        mc = mcscf.state_average_mix_(mc, [gs_solver, ex_solver], [0.5, 0.5])

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
    chkfile: Path,
) -> Any:
    active_nelec = normalize_active_nelec(nelecas_total, spin_2s)
    ci_mc = mcscf.CASCI(mf, ncas, active_nelec)
    ci_mc.chkfile = str(chkfile)
    ci_mc.max_memory = mf.max_memory
    ci_mc.fcisolver = make_fci_solver(mf.mol, spin_2s, wfnsym, nroots)
    ci_mc.kernel(common_mo)
    convergence = getattr(ci_mc, "converged", True)
    if not bool(np.all(np.asarray(convergence))):
        raise RuntimeError(
            f"CASCI did not converge for spin 2S={spin_2s}, symmetry {wfnsym}"
        )
    return ci_mc


def nevpt2_total_energy(mc: Any, root: int) -> Tuple[float, float, float]:
    """Return CASCI energy, SC-NEVPT2 correction, and their sum."""
    reference_energy = root_value(mc.e_tot, root)
    correction = float(mrpt.NEVPT(mc, root=root).kernel())
    return reference_energy, correction, reference_energy + correction


# -----------------------------------------------------------------------------
# CI/configuration and orbital analysis
# -----------------------------------------------------------------------------


def spatial_configuration_weights(
    ci_vector: np.ndarray,
    ncas: int,
    nelec: Tuple[int, int],
    determinant_cutoff: float = 0.0,
) -> List[Tuple[Tuple[int, ...], float]]:
    """Aggregate determinant weights into spatial occupation configurations."""
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
    """Return all spatial configurations differing by one electron transfer.

    The comparison is made relative to the dominant spatial occupation of the
    ground-state CASCI wave function.  Each result contains the target
    occupation, aggregated configuration weight, active-space hole index, and
    active-space particle index.
    """
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
    """Return descending eigenvalues of the active-space spin-summed 1-RDM."""
    dm1 = np.asarray(mc.fcisolver.make_rdm1(ci_vector, mc.ncas, nelec))
    dm1 = 0.5 * (dm1 + dm1.T.conj())
    return np.linalg.eigvalsh(dm1).real[::-1]


def state_spin_square(mc: Any, ci_vector: np.ndarray, nelec: Tuple[int, int]) -> Tuple[float, float]:
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


def molecular_plane_axes(mol: gto.Mole) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return center, plane axes u/v, and normal from the heavy-atom geometry."""
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges())
    fit_coords = coords[charges > 1]
    center = fit_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(fit_coords - center, full_matrices=False)
    u_axis, v_axis, normal = vh[0], vh[1], vh[-1]

    # Deterministic signs make plots reproducible between linear algebra builds.
    for vector in (u_axis, v_axis, normal):
        largest = int(np.argmax(np.abs(vector)))
        if vector[largest] < 0:
            vector *= -1
    if np.dot(np.cross(u_axis, v_axis), normal) < 0:
        v_axis *= -1
    return center, u_axis, v_axis, normal


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
    offset_bohr: float,
    grid_size: int,
    margin_bohr: float,
) -> None:
    if plt is None:
        raise RuntimeError("Matplotlib is not installed; use --no-plots or install matplotlib")

    center, u_axis, v_axis, normal = molecular_plane_axes(mol)
    atom_coords = np.asarray(mol.atom_coords())
    rel = atom_coords - center
    atom_u = rel @ u_axis
    atom_v = rel @ v_axis

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
        raise RuntimeError("Orbital values are numerically zero on the selected slice")

    levels = np.linspace(-vmax, vmax, 41)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    contour = ax.contourf(uu, vv, values, levels=levels, cmap="seismic", extend="both")
    ax.contour(uu, vv, values, levels=[0.0], colors="black", linewidths=0.6)

    for i, j in inferred_bonds(mol):
        ax.plot([atom_u[i], atom_u[j]], [atom_v[i], atom_v[j]], linewidth=1.3, color="black")
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        ax.scatter([atom_u[ia]], [atom_v[ia]], s=30, color="black", zorder=3)
        ax.text(atom_u[ia] + 0.08, atom_v[ia] + 0.08, symbol, fontsize=9)

    fig.colorbar(contour, ax=ax, label="MO amplitude")
    ax.set_xlabel("in-plane coordinate u / bohr")
    ax.set_ylabel("in-plane coordinate v / bohr")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nplane offset = {offset_bohr:.2f} bohr")
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
    args: argparse.Namespace,
) -> Dict[str, Any]:
    full_index = inactive_count + active_index
    coeff = np.asarray(common_mo[:, full_index])
    stem = f"{role}_active{active_index + 1}_mo{full_index + 1}_{slugify(irrep)}"
    cube_path = state_dir / f"{stem}.cube"
    png_path = state_dir / f"{stem}.png"

    if args.cubes:
        cubegen.orbital(
            mol,
            str(cube_path),
            coeff,
            resolution=args.cube_resolution,
            margin=args.cube_margin,
        )
    if args.plots:
        plot_orbital_slice(
            mol,
            coeff,
            png_path,
            title=f"{role.capitalize()} orbital: active {active_index + 1}, full MO {full_index + 1}, {irrep}",
            offset_bohr=args.slice_offset,
            grid_size=args.slice_grid,
            margin_bohr=args.slice_margin,
        )

    return {
        f"{role}_active_orbital_1based": active_index + 1,
        f"{role}_full_mo_1based": full_index + 1,
        f"{role}_irrep": irrep,
        f"{role}_cube": str(cube_path) if args.cubes else None,
        f"{role}_png": str(png_path) if args.plots else None,
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
        args,
        state_dir,
    )
    common_mo = np.asarray(sa_mc.mo_coeff)
    inactive_count = int(sa_mc.ncore)
    active_coeff = common_mo[:, inactive_count : inactive_count + ncas]
    active_irreps = active_orbital_irreps(mf.mol, active_coeff)

    # Write the shared state-averaged orbitals for independent inspection.
    try:
        molden.from_mo(mf.mol, str(state_dir / "sa_casscf_orbitals.molden"), common_mo)
    except Exception as exc:
        print(f"WARNING: could not write Molden file for {target.key}: {exc}", file=sys.stderr)

    active_rows = [
        {
            "active_orbital_1based": i + 1,
            "full_mo_1based": inactive_count + i + 1,
            "irrep": active_irreps[i],
        }
        for i in range(ncas)
    ]
    write_csv(active_rows, state_dir / "active_orbitals.csv")

    if target.is_excited_a1_singlet:
        # One two-root A1 CASCI gives both the ground and target roots.
        a1_casci = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            spin_2s=0,
            wfnsym="A1",
            nroots=2,
            chkfile=state_dir / "a1_multiroot_casci.chk",
        )
        ground_mc = target_mc = a1_casci
        ground_root = 0
        target_root = 1
    else:
        ground_mc = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            spin_2s=0,
            wfnsym="A1",
            nroots=1,
            chkfile=state_dir / "ground_a1_casci.chk",
        )
        target_mc = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            spin_2s=target.spin_2s,
            wfnsym=target.wfnsym,
            nroots=1,
            chkfile=state_dir / "target_casci.chk",
        )
        ground_root = 0
        target_root = 0

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
            "No one-electron spatial configuration was identified relative to the "
            f"dominant ground configuration {occupation_text(ground_occ)}"
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
        ground_noons, target_noons, state_dir / "active_natural_occupations.csv"
    )

    ss, effective_multiplicity = state_spin_square(target_mc, target_ci, target_nelec)

    row: Dict[str, Any] = {
        **asdict(target),
        "basis": args.basis,
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
            (target_total - ground_total) * HARTREE_TO_EV - target.paper_tbe_augccpvtz_eV
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
        "total_single_excitation_weight_relative_to_dominant_ground_configuration": total_single_weight,
        "ground_active_natural_occupations": [float(x) for x in ground_noons],
        "target_active_natural_occupations": [float(x) for x in target_noons],
    }

    row.update(
        write_orbital_outputs(
            mf.mol,
            common_mo,
            inactive_count,
            hole_active,
            active_irreps[hole_active],
            "hole",
            state_dir,
            args,
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
            args,
        )
    )

    (state_dir / "state_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n=== Furan valence-excitation summary ===")
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
            "Characterize the four valence pi->pi* excitations of furan with "
            "SA-CASSCF/CASCI and PySCF strongly contracted NEVPT2."
        )
    )
    parser.add_argument("coordinates", type=Path, help="Numbered Coordinates file")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    parser.add_argument(
        "--states",
        default="all",
        help=(
            "Comma-separated selection: singlet-b2,singlet-a1,triplet-b2,"
            "triplet-a1; default: all"
        ),
    )
    parser.add_argument("--outdir", type=Path, default=Path("furan_sc_nevpt2"))
    parser.add_argument("--max-memory", type=int, default=4000, help="PySCF memory limit in MB")
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
        "--avas-labels",
        nargs="+",
        default=["C 2px", "O 2px"],
        help="AO labels defining the pi space; default: C 2px O 2px",
    )
    parser.add_argument("--avas-threshold", type=float, default=0.10)
    parser.add_argument("--avas-minao", default="minao")
    parser.add_argument(
        "--avas-with-iao",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--avas-excluded-core",
        type=int,
        default=5,
        help="Lowest MOs excluded from AVAS selection: four C 1s plus O 1s",
    )
    parser.add_argument("--expected-ncas", type=int, default=5)
    parser.add_argument("--expected-nelecas", type=int, default=6)
    parser.add_argument(
        "--strict-active-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort unless AVAS returns CAS(6,5) by default",
    )

    parser.add_argument("--casscf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--casscf-conv-tol-grad", type=float, default=1e-5)
    parser.add_argument("--casscf-max-cycle-macro", type=int, default=100)
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
        help="Write 2-D orbital slice PNGs",
    )
    parser.add_argument(
        "--slice-offset",
        type=float,
        default=0.70,
        help="Offset from the molecular nodal plane in bohr",
    )
    parser.add_argument("--slice-grid", type=int, default=301)
    parser.add_argument("--slice-margin", type=float, default=2.5)
    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort the batch at the first failed state instead of continuing",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    selected_states = parse_state_selection(args.states)
    args.outdir.mkdir(parents=True, exist_ok=True)

    molecule_index, molecule_name, atoms = find_named_molecule(
        parse_coordinate_file(args.coordinates), "Furan"
    )
    mol = build_molecule(
        atoms,
        basis=args.basis,
        unit=args.unit,
        max_memory=args.max_memory,
        log_path=args.outdir / "pyscf.log",
        verbose=args.verbose,
    )

    if "A1" not in mol.irrep_name or "B2" not in mol.irrep_name:
        raise RuntimeError(
            f"PySCF detected point group {mol.groupname} with irreps {mol.irrep_name}; "
            "the scripted A1/B2 state targeting assumes the supplied C2v orientation."
        )

    mf = run_rhf(mol, args, args.outdir)
    ncas, nelecas_total, mo_guess = construct_pi_active_space(mf, args)

    metadata = {
        "molecule_index": molecule_index,
        "molecule": molecule_name,
        "formula": hill_formula(atoms),
        "basis": args.basis,
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "unit": args.unit,
        "point_group": str(mol.groupname),
        "electrons": int(mol.nelectron),
        "nao_nmo": int(mol.nao_nr()),
        "rhf_energy_Eh": float(mf.e_tot),
        "active_space_electrons": nelecas_total,
        "active_space_orbitals": ncas,
        "avas_labels": list(args.avas_labels),
        "avas_threshold": args.avas_threshold,
        "states": [state.key for state in selected_states],
        "method_note": (
            "PySCF strongly contracted NEVPT2; paper main-table values are "
            "partially contracted NEVPT2/aug-cc-pVTZ"
        ),
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"Furan: basis={args.basis}, point_group={mol.groupname}, NMO={mol.nao_nr()}, "
        f"CAS({nelecas_total},{ncas})"
    )
    print(
        "NOTE: this is PySCF SC-NEVPT2. The paper's main NEVPT2 column is "
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
                args,
                args.outdir,
            )
            result["status"] = "OK"
            results.append(result)
            print(
                f"Vertical SC-NEVPT2 energy: {result['vertical_sc_nevpt2_eV']:.6f} eV; "
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

    write_csv(results, args.outdir / "furan_vertical_energies.csv")
    (args.outdir / "furan_vertical_energies.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print_summary(results)
    print(f"\nResults written below: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
