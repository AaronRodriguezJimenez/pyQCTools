#!/usr/bin/env python3
"""Configuration-driven SA-CASSCF/CASCI/SC-NEVPT2 workflow (v3).

This program covers the conservative 27-molecule, single-excitation-dominated
benchmark set discussed in this project, including Furan and Acetaldehyde.  It is
based on Sarkar et al., J. Chem. Theory Comput. 2022, 18, 2418-2436.

The central design choice is deliberate: benchmark-state metadata are stored in
a JSON registry, while the active-space definition is stored in an editable
per-molecule case file.  A single fully automatic active-space selector is not
chemically reliable for all of the benchmark families (pi systems, n->pi*,
sigma->pi*, linear degeneracies, and Rydberg states).  The workflow is thus:

  1. ``prepare``: run RHF only, write canonical-orbital diagnostics and a case
     template.
  2. Inspect the CSV/Molden output and enter ``active_mos_1based`` in case.json.
  3. ``run``: state-average CASSCF, symmetry/spin-resolved CASCI, strongly
     contracted NEVPT2, CI configuration analysis, and hole/particle plots.

The script supports arbitrary manual active spaces and arbitrary target states
contained in the registry.  It also supports root/symmetry overrides in the
case file, which are essential when PySCF lowers the molecular point group or
when a state root changes with basis/active-space choice.

Important limitations
---------------------
* PySCF ``mrpt.NEVPT`` is strongly contracted NEVPT2 (SC-NEVPT2), while the
  main NEVPT2 column in the benchmark paper is partially contracted NEVPT2.
* STO-3G is appropriate only for the current proof-of-concept workflow.  It has
  no physically meaningful diffuse Rydberg space.  Rydberg targets are blocked
  with a non-augmented basis unless explicitly allowed.
* The dominant single-excitation weight is obtained by aggregating CASCI
  determinant weights into spatial occupations.  It depends on the chosen
  active-orbital basis and is not a coupled-cluster %%T1 diagnostic.
* ``target_root_guess`` values in the registry are initial guesses.  Verify the
  state identity using symmetry, CI configurations, natural occupations, and
  the generated orbitals.
* Benchmark calculations may use a different active space and a different
  state-average for each target state.  Version 2 therefore supports per-state
  ``active_mos_1based``, ``sa_target_nroots``, ``casci_nroots``, and expected
  active-orbital irrep counts in ``state_overrides``.

Examples
--------
List registered molecules::

    python 06_general_single_excitation_nevpt2_v3.py list

Prepare Cyclopropene with STO-3G::

    python 06_general_single_excitation_nevpt2_v3.py prepare Coordinates \
        --molecule Cyclopropene --basis sto-3g \
        --outdir prepared_cases

After editing the generated case.json::

    python 06_general_single_excitation_nevpt2_v3.py run Coordinates \
        --case prepared_cases/03_Cyclopropene/case.json

A provisional frontier-space suggestion can be written during preparation::

    python 06_general_single_excitation_nevpt2_v3.py prepare Coordinates \
        --molecule Cyclopropene --suggest-frontier 2,2

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
except ImportError:
    plt = None

try:
    import pyscf
    from pyscf import fci, gto, lo, mcscf, mrpt, scf, symm
    from pyscf.fci import cistring
    from pyscf.tools import cubegen, molden
except ImportError:
    pyscf = None
    fci = gto = lo = mcscf = mrpt = scf = symm = None
    cistring = cubegen = molden = None


def require_pyscf() -> None:
    if pyscf is None:
        raise SystemExit(
            "PySCF is required for prepare/run. Install it in the environment "
            "used to run this script, for example with "
            "`python -m pip install pyscf matplotlib`."
        )


WORKFLOW_VERSION = "3.0"
HARTREE_TO_EV = 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903
DEFAULT_REGISTRY = Path(__file__).with_name("benchmark_single_excitation_cases.json")

HEADER_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$"
)

Atom = Tuple[str, Tuple[float, float, float]]
MoleculeRecord = Tuple[int, str, List[Atom]]

COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
}


@dataclass(frozen=True)
class TargetState:
    key: str
    label: str
    multiplicity: int
    spin_2s: int
    paper_wfnsym: str
    pyscf_wfnsym: str | None
    target_root_guess: int
    state_type: str
    excitation: str
    paper_tbe_augccpvtz_eV: float
    paper_pc_nevpt2_augccpvtz_eV: float
    minimal_basis_recommended: bool


# -----------------------------------------------------------------------------
# Generic file and registry utilities
# -----------------------------------------------------------------------------


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "item"


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


def parse_coordinate_file(path: Path) -> List[MoleculeRecord]:
    records: List[MoleculeRecord] = []
    current_index: int | None = None
    current_name: str | None = None
    current_atoms: List[Atom] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        header = HEADER_RE.match(line)
        if header:
            if current_index is not None:
                if not current_atoms:
                    raise ValueError(f"Molecule {current_index} has no atoms")
                assert current_name is not None
                records.append((current_index, current_name, current_atoms))
            current_index = int(header.group(1))
            current_name = header.group(2).strip()
            current_atoms = []
            continue
        atom = ATOM_RE.match(line)
        if atom and current_index is not None:
            symbol = atom.group(1)
            xyz = tuple(float(atom.group(i)) for i in (2, 3, 4))
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
    matches = [record for record in records if record[1].casefold() == name.casefold()]
    if len(matches) != 1:
        available = ", ".join(record[1] for record in records)
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
        order.extend(sorted(element for element in counts if element not in {"C", "H"}))
    else:
        order = sorted(counts)
    return "".join(
        element if counts[element] == 1 else f"{element}{counts[element]}"
        for element in order
    )


def load_registry(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported registry schema in {path}")
    if "molecules" not in data:
        raise ValueError(f"Registry {path} has no 'molecules' mapping")
    return data


def registry_molecule(registry: Mapping[str, Any], name: str) -> Dict[str, Any]:
    molecules = registry["molecules"]
    matches = [key for key in molecules if key.casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError(
            f"Molecule {name!r} not found uniquely in registry. "
            f"Available: {', '.join(molecules)}"
        )
    return dict(molecules[matches[0]])


def target_states_from_entry(entry: Mapping[str, Any]) -> List[TargetState]:
    return [TargetState(**state) for state in entry["states"]]


def select_target_states(
    states: Sequence[TargetState], selection: str | None
) -> List[TargetState]:
    if selection is None or selection.strip().lower() == "valence":
        selected = [state for state in states if state.state_type.upper() == "V"]
    elif selection.strip().lower() == "all":
        selected = list(states)
    elif selection.strip().lower() == "rydberg":
        selected = [state for state in states if state.state_type.upper() == "R"]
    else:
        wanted = {part.strip() for part in selection.split(",") if part.strip()}
        by_key = {state.key: state for state in states}
        missing = sorted(wanted - set(by_key))
        if missing:
            raise ValueError(f"Unknown state keys: {missing}")
        selected = [state for state in states if state.key in wanted]
    if not selected:
        raise ValueError("The state selection is empty")
    return selected


def parse_mo_list(text: str | None) -> List[int] | None:
    if text is None:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--active-mos did not contain any MO indices")
    if min(values) < 1 or len(set(values)) != len(values):
        raise ValueError("Active MO indices must be unique positive 1-based integers")
    return values


def parse_frontier_suggestion(text: str | None) -> Tuple[int, int] | None:
    if text is None:
        return None
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 2 or min(parts) < 0 or sum(parts) == 0:
        raise ValueError("--suggest-frontier must be NOCC,NVIRT, e.g. 2,2")
    return parts[0], parts[1]


def basis_looks_diffuse(name: str) -> bool:
    lower = name.lower().replace("_", "-")
    return lower.startswith("aug-") or "d-aug" in lower or "diffuse" in lower


# -----------------------------------------------------------------------------
# RHF and canonical-orbital diagnostics
# -----------------------------------------------------------------------------


def build_molecule(
    atoms: Sequence[Atom],
    basis: str,
    unit: str,
    charge: int,
    max_memory: int,
    log_path: Path,
    verbose: int,
    symmetry: bool,
    symmetry_subgroup: str | None,
) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = unit
    mol.basis = basis
    mol.charge = charge
    mol.spin = 0
    mol.cart = False
    mol.symmetry = symmetry
    if symmetry_subgroup:
        mol.symmetry_subgroup = symmetry_subgroup
    mol.max_memory = max_memory
    mol.verbose = verbose
    mol.output = str(log_path)
    mol.build()
    return mol


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
        dm0 = mf.make_rdm1() if mf.mo_coeff is not None else None
        nmf = mf.newton()
        nmf.chkfile = str(chkfile)
        nmf.conv_tol = conv_tol
        nmf.max_cycle = max_cycle
        nmf.max_memory = mol.max_memory
        nmf.kernel(dm0)
        mf = nmf
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def orbital_symmetry_labels(mol: gto.Mole, coeff: np.ndarray) -> List[str]:
    if not mol.symmetry:
        return [""] * coeff.shape[1]
    try:
        labels = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, coeff, check=False
        )
        return [str(label) for label in labels]
    except Exception as exc:
        return [f"UNLABELED:{type(exc).__name__}"] * coeff.shape[1]


def meta_lowdin_weights(mol: gto.Mole, coeff: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    x = lo.orth.orth_ao(mol, method="meta_lowdin", s=overlap)
    c_orth = x.T.conj() @ overlap @ coeff
    weights = np.abs(c_orth) ** 2
    denominator = weights.sum(axis=0)
    denominator[denominator == 0] = 1.0
    return weights / denominator


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


def canonical_orbital_rows(mf: scf.hf.SCF) -> List[Dict[str, Any]]:
    mol = mf.mol
    coeff = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    energy = np.asarray(mf.mo_energy)
    overlap = mf.get_ovlp()
    weights = meta_lowdin_weights(mol, coeff, overlap)
    irreps = orbital_symmetry_labels(mol, coeff)
    centers, spreads = spatial_moments(mol, coeff)
    labels = mol.ao_labels(fmt=False)
    atom_ids = np.asarray([int(label[0]) for label in labels])
    nl = [str(label[2]).lower() for label in labels]
    occupied = np.where(occ > 0)[0]
    virtual = np.where(occ == 0)[0]
    homo = int(occupied[-1])
    lumo = int(virtual[0])

    rows: List[Dict[str, Any]] = []
    for imo in range(coeff.shape[1]):
        atom_pop = np.asarray(
            [weights[atom_ids == ia, imo].sum() for ia in range(mol.natm)]
        )
        top_atoms = np.argsort(atom_pop)[::-1][:3]
        top_text = ";".join(
            f"{ia + 1}:{mol.atom_symbol(int(ia))}={atom_pop[ia]:.3f}"
            for ia in top_atoms
            if atom_pop[ia] > 1e-4
        )
        row: Dict[str, Any] = {
            "mo_1based": imo + 1,
            "frontier_label": frontier_label(imo, homo, lumo),
            "occupation": float(occ[imo]),
            "energy_Eh": float(energy[imo]),
            "energy_eV": float(energy[imo] * HARTREE_TO_EV),
            "irrep": irreps[imo],
            "spread_bohr2": float(spreads[imo]),
            "center_x_bohr": float(centers[imo, 0]),
            "center_y_bohr": float(centers[imo, 1]),
            "center_z_bohr": float(centers[imo, 2]),
            "top_atoms_meta_lowdin": top_text,
        }
        for angular in "spdfgh":
            mask = np.asarray([value.endswith(angular) for value in nl])
            row[f"{angular}_character"] = float(weights[mask, imo].sum()) if mask.any() else 0.0
        rows.append(row)
    return rows


def totally_symmetric_irrep(mol: gto.Mole) -> str:
    if not mol.symmetry:
        raise RuntimeError("A symmetry-resolved calculation requires mol.symmetry=True")
    ids = [int(value) for value in mol.irrep_id]
    names = [str(value) for value in mol.irrep_name]
    if 0 in ids:
        return names[ids.index(0)]
    return names[0]


def provisional_frontier_active_mos(mf: scf.hf.SCF, nocc: int, nvirt: int) -> List[int]:
    occ = np.asarray(mf.mo_occ)
    occupied = np.where(occ > 0)[0]
    virtual = np.where(occ == 0)[0]
    if nocc > len(occupied) or nvirt > len(virtual):
        raise ValueError("Requested frontier suggestion exceeds occupied/virtual space")
    selected = list(occupied[-nocc:]) + list(virtual[:nvirt])
    return [int(index) + 1 for index in selected]


# -----------------------------------------------------------------------------
# Case preparation
# -----------------------------------------------------------------------------


def prepare_case(args: argparse.Namespace) -> None:
    require_pyscf()
    registry = load_registry(args.registry)
    entry = registry_molecule(registry, args.molecule)
    states = select_target_states(target_states_from_entry(entry), args.states)
    record = find_named_molecule(parse_coordinate_file(args.coordinates), args.molecule)
    index, name, atoms = record
    charge = int(entry.get("charge", 0))

    outdir = args.outdir / f"{index:02d}_{slugify(name)}"
    outdir.mkdir(parents=True, exist_ok=True)
    mol = build_molecule(
        atoms,
        basis=args.basis,
        unit=args.unit,
        charge=charge,
        max_memory=args.max_memory,
        log_path=outdir / "prepare.log",
        verbose=args.verbose,
        symmetry=args.symmetry,
        symmetry_subgroup=args.symmetry_subgroup,
    )
    mf = run_rhf(
        mol,
        outdir / "rhf.chk",
        args.scf_conv_tol,
        args.scf_max_cycle,
        args.newton_fallback,
    )

    rows = canonical_orbital_rows(mf)
    write_csv(rows, outdir / "canonical_orbitals.csv")
    molden.from_scf(mf, str(outdir / "canonical_orbitals.molden"))

    active_mos: List[int] = []
    status = "EDIT_REQUIRED"
    suggestion = parse_frontier_suggestion(args.suggest_frontier)
    if suggestion:
        active_mos = provisional_frontier_active_mos(mf, *suggestion)
        status = "PROVISIONAL_FRONTIER_SUGGESTION_REQUIRES_INSPECTION"

    ground_irrep = totally_symmetric_irrep(mol)
    available_irreps = [str(value) for value in mol.irrep_name]
    state_overrides: Dict[str, Any] = {}
    for state in states:
        wfnsym = state.pyscf_wfnsym
        actual_root_guess = int(state.target_root_guess)
        if state.spin_2s == 0 and wfnsym == ground_irrep:
            actual_root_guess += 1
        state_overrides[state.key] = {
            "pyscf_wfnsym": wfnsym,
            "target_root": actual_root_guess,
            "active_mos_1based": [],
            "sa_target_nroots": None,
            "casci_nroots": None,
            "expected_active_irrep_counts": {},
            "comment": (
                "Verify symmetry, target root, state-average root count, and "
                "state-specific active space. Empty active_mos_1based inherits "
                "the molecule-level active space."
            ),
        }

    case = {
        "schema_version": 1,
        "registry_file": str(args.registry.resolve()),
        "coordinates_file": str(args.coordinates.resolve()),
        "molecule": name,
        "coordinates_index": index,
        "formula": hill_formula(atoms),
        "charge": charge,
        "basis": args.basis,
        "unit": args.unit,
        "symmetry": bool(args.symmetry),
        "symmetry_subgroup": args.symmetry_subgroup,
        "detected_topgroup": str(getattr(mol, "topgroup", "")),
        "detected_group": str(mol.groupname),
        "available_irreps": available_irreps,
        "ground_wfnsym": ground_irrep,
        "active_mos_1based": active_mos,
        "active_space_status": status,
        "selected_state_keys": [state.key for state in states],
        "state_overrides": state_overrides,
        "benchmark_notes": list(entry.get("notes", [])),
        "instructions": [
            "Inspect canonical_orbitals.csv and canonical_orbitals.molden.",
            "Replace active_mos_1based with a chemically justified default active space.",
            "Use state_overrides[STATE].active_mos_1based when a target needs a different space.",
            "Check every pyscf_wfnsym against available_irreps.",
            "Check target_root using root_scan.csv; registry values are guesses.",
            "Set sa_target_nroots when extra roots were included in the benchmark state average.",
            "Rydberg states require diffuse functions and suitable diffuse active orbitals.",
        ],
    }
    (outdir / "case.json").write_text(json.dumps(case, indent=2) + "\n")

    print(f"Prepared {name} in {outdir.resolve()}")
    print(f"Detected symmetry: topgroup={case['detected_topgroup']}, group={case['detected_group']}")
    print(f"Available irreps: {', '.join(available_irreps)}")
    print(f"Ground-state irrep: {ground_irrep}")
    print(f"Active-space status: {status}")
    if active_mos:
        print(f"Provisional active MOs: {active_mos}")
    print("Edit case.json before running the correlated calculation.")


# -----------------------------------------------------------------------------
# CASSCF, CASCI, and NEVPT2 helpers
# -----------------------------------------------------------------------------


def as_root_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return [value]
    if array.ndim == 1:
        return [item for item in array]
    return [value]


def root_value(value: Any, root: int) -> float:
    values = as_root_list(value)
    if root >= len(values):
        raise IndexError(f"Root {root} is unavailable; object contains {len(values)} roots")
    return float(np.asarray(values[root]))


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


def selected_mo_irrep_counts(
    mol: gto.Mole, coeff: np.ndarray, active_mos_1based: Sequence[int]
) -> Dict[str, int]:
    """Count selected canonical MOs by point-group irrep."""
    labels = orbital_symmetry_labels(mol, np.asarray(coeff))
    counts = Counter(labels[int(index) - 1] for index in active_mos_1based)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def validate_expected_irrep_counts(
    actual: Mapping[str, int], expected: Mapping[str, Any] | None
) -> None:
    """Fail early if a state-specific active space has the wrong irrep content."""
    if not expected:
        return
    normalized_expected = {str(key): int(value) for key, value in expected.items()}
    all_irreps = sorted(set(actual) | set(normalized_expected))
    mismatches = [
        f"{irrep}: selected={int(actual.get(irrep, 0))}, "
        f"expected={int(normalized_expected.get(irrep, 0))}"
        for irrep in all_irreps
        if int(actual.get(irrep, 0)) != int(normalized_expected.get(irrep, 0))
    ]
    if mismatches:
        raise RuntimeError(
            "Active-space irrep composition does not match the configured "
            "benchmark recipe: " + "; ".join(mismatches)
        )


def prepare_active_space(
    mf: scf.hf.SCF, active_mos_1based: Sequence[int]
) -> Tuple[int, int, np.ndarray]:
    nmo = np.asarray(mf.mo_coeff).shape[1]
    active = sorted(int(value) for value in active_mos_1based)
    if not active or active[0] < 1 or active[-1] > nmo or len(set(active)) != len(active):
        raise ValueError(f"Invalid active MO list {active}; NMO={nmo}")
    zero = np.asarray(active, dtype=int) - 1
    occ = np.asarray(mf.mo_occ)
    nelecas_total_float = float(occ[zero].sum())
    nelecas_total = int(round(nelecas_total_float))
    if abs(nelecas_total_float - nelecas_total) > 1e-8:
        raise RuntimeError("Active electron count inferred from RHF occupations is not integral")
    if nelecas_total <= 0 or nelecas_total >= 2 * len(active) + 1:
        raise RuntimeError(
            f"Unreasonable inferred active space CAS({nelecas_total},{len(active)})"
        )
    if not np.any(occ[zero] > 0) or not np.any(occ[zero] == 0):
        raise RuntimeError("The active space should contain both occupied and virtual RHF MOs")
    template = mcscf.CASSCF(mf, len(active), nelecas_total)
    mo_guess = mcscf.sort_mo(template, mf.mo_coeff, active, base=1)
    return len(active), nelecas_total, np.asarray(mo_guess)


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
            solver, shift=spin_penalty, ss=target_s * (target_s + 1.0)
        )
        solver.spin = spin_2s
        solver.wfnsym = wfnsym
        solver.nroots = nroots
    return solver


def apply_state_average_mix(mc: Any, solvers: Sequence[Any], weights: Sequence[float]) -> Any:
    if hasattr(mc, "state_average_mix_"):
        return mc.state_average_mix_(list(solvers), list(weights))
    return mcscf.state_average_mix_(mc, list(solvers), list(weights))


def run_state_averaged_casscf(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    mo_guess: np.ndarray,
    ground_irrep: str,
    target_spin_2s: int,
    target_irrep: str,
    target_root: int,
    sa_target_nroots: int | None,
    args: argparse.Namespace,
    state_dir: Path,
) -> Any:
    mc = mcscf.CASSCF(mf, ncas, nelecas_total)
    mc.chkfile = str(state_dir / "sa_casscf.chk")
    mc.conv_tol = args.casscf_conv_tol
    mc.conv_tol_grad = args.casscf_conv_tol_grad
    mc.max_cycle_macro = args.casscf_max_cycle_macro
    mc.max_memory = args.max_memory

    requested_sa_roots = int(sa_target_nroots) if sa_target_nroots is not None else 0
    if requested_sa_roots < 0:
        raise ValueError("sa_target_nroots must be non-negative")

    if target_spin_2s == 0 and target_irrep == ground_irrep:
        # This count includes the ground-state root because all states belong to
        # the same spin/symmetry block.
        nroots = max(target_root + 1, requested_sa_roots, 2)
        solver = make_fci_solver(mf.mol, 0, ground_irrep, nroots, args.spin_penalty)
        mc.fcisolver = solver
        weights = [1.0 / nroots] * nroots
        mc = mc.state_average_(weights)
    else:
        # Here the singlet ground state is represented by its own solver. The
        # configured count refers only to roots of the target block.
        gs_solver = make_fci_solver(mf.mol, 0, ground_irrep, 1, args.spin_penalty)
        target_nroots = max(target_root + 1, requested_sa_roots, 1)
        ex_solver = make_fci_solver(
            mf.mol, target_spin_2s, target_irrep, target_nroots, args.spin_penalty
        )
        nstates = 1 + target_nroots
        weights = [1.0 / nstates] * nstates
        mc = apply_state_average_mix(mc, [gs_solver, ex_solver], weights)

    mc.kernel(mo_guess)
    if not mc.converged:
        raise RuntimeError("SA-CASSCF did not converge")
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
    # SymAdaptedCASCI does not expose an SCF-style ``chkfile`` attribute.
    # Assigning it only produces a PySCF warning and does not create a usable
    # checkpoint, so v3 writes a compact JSON root summary after the kernel.
    ci_mc.max_memory = mf.max_memory
    ci_mc.fcisolver = make_fci_solver(
        mf.mol, spin_2s, wfnsym, nroots, spin_penalty
    )
    ci_mc.kernel(common_mo)
    convergence = getattr(ci_mc, "converged", True)
    if not bool(np.all(np.asarray(convergence))):
        raise RuntimeError(
            f"CASCI did not converge for 2S={spin_2s}, symmetry={wfnsym}"
        )
    summary_path = Path(chkfile).with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "spin_2s": int(spin_2s),
                "wfnsym": str(wfnsym),
                "nroots": int(nroots),
                "e_tot_Eh": np.asarray(ci_mc.e_tot).reshape(-1).tolist(),
                "note": "CASCI metadata only; PySCF SymAdaptedCASCI has no chkfile attribute.",
            },
            indent=2,
        )
        + "\n"
    )
    return ci_mc


def nevpt2_total_energy(mc: Any, root: int) -> Tuple[float, float, float]:
    reference = root_value(mc.e_tot, root)
    correction = float(mrpt.NEVPT(mc, root=root).kernel())
    return reference, correction, reference + correction


# -----------------------------------------------------------------------------
# CI and orbital characterization
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
            occupation = np.zeros(ncas, dtype=int)
            occupation[alpha_occ] += 1
            occupation[beta_occ] += 1
            grouped[tuple(int(value) for value in occupation)] += weight
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def occupation_text(occupation: Sequence[int]) -> str:
    return "[" + " ".join(str(int(value)) for value in occupation) + "]"


def single_excitation_configurations(
    ground_occupation: Sequence[int],
    target_configurations: Sequence[Tuple[Tuple[int, ...], float]],
) -> List[Tuple[Tuple[int, ...], float, int, int]]:
    ground = np.asarray(ground_occupation, dtype=int)
    results: List[Tuple[Tuple[int, ...], float, int, int]] = []
    for occupation, weight in target_configurations:
        delta = np.asarray(occupation, dtype=int) - ground
        holes = np.where(delta == -1)[0]
        particles = np.where(delta == 1)[0]
        if np.sum(np.abs(delta)) == 2 and len(holes) == 1 and len(particles) == 1:
            results.append((occupation, float(weight), int(holes[0]), int(particles[0])))
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
    return orbital_symmetry_labels(mol, active_coeff)


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
    rows: List[Dict[str, Any]] = []
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
    ground: Sequence[float], target: Sequence[float], path: Path
) -> None:
    write_csv(
        [
            {
                "natural_orbital_rank": index + 1,
                "ground_occupation": float(ground[index]),
                "target_occupation": float(target[index]),
            }
            for index in range(len(ground))
        ],
        path,
    )


def casci_root_scan_rows(
    target_mc: Any,
    ground_reference_Eh: float,
    ground_occupation: Sequence[int],
    target_nelec: Tuple[int, int],
    ncas: int,
    active_irreps: Sequence[str],
    inactive_count: int,
    same_block_ground_root: int | None,
) -> List[Dict[str, Any]]:
    """Characterize every computed CASCI root before choosing a target root."""
    energies = as_root_list(target_mc.e_tot)
    rows: List[Dict[str, Any]] = []
    for root in range(len(energies)):
        ci_vector = root_ci(target_mc, root)
        configs = spatial_configuration_weights(ci_vector, ncas, target_nelec)
        if not configs:
            continue
        dominant_occ, dominant_weight = configs[0]
        singles = single_excitation_configurations(ground_occupation, configs)
        total_single_weight = float(sum(item[1] for item in singles))
        ss, multiplicity = state_spin_square(target_mc, ci_vector, target_nelec)
        row: Dict[str, Any] = {
            "root_zero_based": root,
            "is_same_block_ground_reference": bool(
                same_block_ground_root is not None and root == same_block_ground_root
            ),
            "absolute_casci_Eh": root_value(target_mc.e_tot, root),
            "vertical_casci_eV_from_ground_reference": (
                root_value(target_mc.e_tot, root) - ground_reference_Eh
            )
            * HARTREE_TO_EV,
            "S2": ss,
            "effective_multiplicity": multiplicity,
            "dominant_spatial_configuration": occupation_text(dominant_occ),
            "dominant_configuration_weight": float(dominant_weight),
            "total_single_excitation_weight_relative_to_ground": total_single_weight,
            "dominant_single_excitation_weight": None,
            "hole_active_orbital_1based": None,
            "hole_full_mo_1based": None,
            "hole_irrep": None,
            "particle_active_orbital_1based": None,
            "particle_full_mo_1based": None,
            "particle_irrep": None,
        }
        if singles:
            _, weight, hole, particle = singles[0]
            row.update(
                {
                    "dominant_single_excitation_weight": float(weight),
                    "hole_active_orbital_1based": hole + 1,
                    "hole_full_mo_1based": inactive_count + hole + 1,
                    "hole_irrep": active_irreps[hole],
                    "particle_active_orbital_1based": particle + 1,
                    "particle_full_mo_1based": inactive_count + particle + 1,
                    "particle_irrep": active_irreps[particle],
                }
            )
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Orbital files and plots
# -----------------------------------------------------------------------------


def molecular_plane_axes(mol: gto.Mole) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges())
    fit = coords[charges > 1]
    if len(fit) < 2:
        fit = coords
    center = fit.mean(axis=0)
    centered = fit - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=True)
    if len(fit) == 2 or (len(singular) >= 2 and singular[1] < 1e-8):
        # Linear system: choose the molecular axis as u and construct a stable
        # perpendicular pair for plotting.
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
        u_axis, v_axis, normal = vh[0], vh[1], vh[-1]
    for vector in (u_axis, v_axis, normal):
        largest = int(np.argmax(np.abs(vector)))
        if vector[largest] < 0:
            vector *= -1
    if np.dot(np.cross(u_axis, v_axis), normal) < 0:
        v_axis *= -1
    return center, u_axis, v_axis, normal


def inferred_bonds(mol: gto.Mole) -> List[Tuple[int, int]]:
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
        raise RuntimeError("Matplotlib is unavailable; use --no-plots")
    center, u_axis, v_axis, normal = molecular_plane_axes(mol)
    atom_coords = np.asarray(mol.atom_coords())
    relative = atom_coords - center
    atom_u = relative @ u_axis
    atom_v = relative @ v_axis
    u_values = np.linspace(atom_u.min() - margin_bohr, atom_u.max() + margin_bohr, grid_size)
    v_values = np.linspace(atom_v.min() - margin_bohr, atom_v.max() + margin_bohr, grid_size)
    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    points = (
        center[None, :]
        + offset_bohr * normal[None, :]
        + uu.reshape(-1, 1) * u_axis[None, :]
        + vv.reshape(-1, 1) * v_axis[None, :]
    )
    ao = mol.eval_gto("GTOval_sph", points)
    values = (ao @ coeff).reshape(grid_size, grid_size)
    absolute = np.abs(values)
    vmax = float(np.quantile(absolute, 0.995))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = float(absolute.max())
    if vmax < 1e-12:
        raise RuntimeError("Orbital is numerically zero on the requested slice")
    levels = np.linspace(-vmax, vmax, 41)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    contour = ax.contourf(uu, vv, values, levels=levels, cmap="seismic", extend="both")
    ax.contour(uu, vv, values, levels=[0.0], colors="black", linewidths=0.6)
    for i, j in inferred_bonds(mol):
        ax.plot([atom_u[i], atom_u[j]], [atom_v[i], atom_v[j]], color="black", linewidth=1.3)
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        ax.scatter([atom_u[ia]], [atom_v[ia]], s=30, color="black", zorder=3)
        ax.text(atom_u[ia] + 0.08, atom_v[ia] + 0.08, symbol, fontsize=9)
    fig.colorbar(contour, ax=ax, label="MO amplitude")
    ax.set_xlabel("u / bohr")
    ax.set_ylabel("v / bohr")
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
    plane_png = state_dir / f"{stem}_plane.png"
    offset_png = state_dir / f"{stem}_offset.png"
    if args.cubes:
        cubegen.orbital(
            mol,
            str(cube_path),
            coeff,
            resolution=args.cube_resolution,
            margin=args.cube_margin,
        )
    if args.plots:
        for path, offset in ((plane_png, 0.0), (offset_png, args.slice_offset)):
            try:
                plot_orbital_slice(
                    mol,
                    coeff,
                    path,
                    title=(
                        f"{role.capitalize()} orbital: active {active_index + 1}, "
                        f"full MO {full_index + 1}, {irrep}"
                    ),
                    offset_bohr=offset,
                    grid_size=args.slice_grid,
                    margin_bohr=args.slice_margin,
                )
            except RuntimeError as exc:
                print(f"WARNING: {path.name}: {exc}", file=sys.stderr)
    return {
        f"{role}_active_orbital_1based": active_index + 1,
        f"{role}_full_mo_1based": full_index + 1,
        f"{role}_irrep": irrep,
        f"{role}_cube": str(cube_path) if args.cubes else None,
        f"{role}_plane_png": str(plane_png) if args.plots and plane_png.exists() else None,
        f"{role}_offset_png": str(offset_png) if args.plots and offset_png.exists() else None,
    }


# -----------------------------------------------------------------------------
# Per-state and case execution
# -----------------------------------------------------------------------------


def run_target_state(
    mf: scf.hf.SCF,
    ncas: int,
    nelecas_total: int,
    mo_guess: np.ndarray,
    ground_irrep: str,
    target: TargetState,
    target_irrep: str,
    target_root: int,
    active_mos_1based: Sequence[int],
    active_irrep_counts: Mapping[str, int],
    sa_target_nroots: int | None,
    casci_nroots: int | None,
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
        ground_irrep,
        target.spin_2s,
        target_irrep,
        target_root,
        sa_target_nroots,
        args,
        state_dir,
    )
    common_mo = np.asarray(sa_mc.mo_coeff)
    sa_weights = list(getattr(sa_mc, "weights", []))
    sa_total_states_used = len(sa_weights)
    sa_target_nroots_used = (
        sa_total_states_used
        if target.spin_2s == 0 and target_irrep == ground_irrep
        else max(sa_total_states_used - 1, 0)
    )
    inactive_count = int(sa_mc.ncore)
    active_coeff = common_mo[:, inactive_count : inactive_count + ncas]
    active_irreps = active_orbital_irreps(mf.mol, active_coeff)
    molden.from_mo(mf.mol, str(state_dir / "sa_casscf_orbitals.molden"), common_mo)
    write_csv(
        [
            {
                "active_orbital_1based": index + 1,
                "full_mo_1based": inactive_count + index + 1,
                "irrep": active_irreps[index],
            }
            for index in range(ncas)
        ],
        state_dir / "active_orbitals.csv",
    )

    requested_casci_roots = int(casci_nroots) if casci_nroots is not None else 0
    if requested_casci_roots < 0:
        raise ValueError("casci_nroots must be non-negative")
    requested_sa_roots = int(sa_target_nroots) if sa_target_nroots is not None else 0

    if target.spin_2s == 0 and target_irrep == ground_irrep:
        same_block_nroots = max(target_root + 1, requested_casci_roots, requested_sa_roots, 2)
        multiroot = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            0,
            ground_irrep,
            same_block_nroots,
            args.spin_penalty,
            state_dir / "same_symmetry_multiroot_casci.chk",
        )
        ground_mc = target_mc = multiroot
        ground_root = 0
        casci_nroots_used = same_block_nroots
    else:
        ground_mc = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            0,
            ground_irrep,
            1,
            args.spin_penalty,
            state_dir / "ground_casci.chk",
        )
        target_block_nroots = max(
            target_root + 1, requested_casci_roots, requested_sa_roots, 1
        )
        target_mc = run_symmetry_casci(
            mf,
            ncas,
            nelecas_total,
            common_mo,
            target.spin_2s,
            target_irrep,
            target_block_nroots,
            args.spin_penalty,
            state_dir / "target_multiroot_casci.chk",
        )
        ground_root = 0
        casci_nroots_used = target_block_nroots

    ground_ref, ground_corr, ground_total = nevpt2_total_energy(ground_mc, ground_root)
    target_ref, target_corr, target_total = nevpt2_total_energy(target_mc, target_root)

    ground_nelec = normalize_active_nelec(nelecas_total, 0)
    target_nelec = normalize_active_nelec(nelecas_total, target.spin_2s)
    ground_ci = root_ci(ground_mc, ground_root)
    target_ci = root_ci(target_mc, target_root)
    ground_configs = spatial_configuration_weights(ground_ci, ncas, ground_nelec)
    target_configs = spatial_configuration_weights(target_ci, ncas, target_nelec)
    write_configuration_table(
        ground_configs, state_dir / "ground_spatial_configurations.csv", args.max_configurations
    )
    write_configuration_table(
        target_configs, state_dir / "target_spatial_configurations.csv", args.max_configurations
    )
    if not ground_configs or not target_configs:
        raise RuntimeError("No CI configurations were extracted")

    ground_occ, ground_weight = ground_configs[0]
    target_occ, target_weight = target_configs[0]

    root_scan = casci_root_scan_rows(
        target_mc=target_mc,
        ground_reference_Eh=ground_ref,
        ground_occupation=ground_occ,
        target_nelec=target_nelec,
        ncas=ncas,
        active_irreps=active_irreps,
        inactive_count=inactive_count,
        same_block_ground_root=(
            0 if target.spin_2s == 0 and target_irrep == ground_irrep else None
        ),
    )
    for scan_row in root_scan:
        scan_row["selected_target_root"] = (
            int(scan_row["root_zero_based"]) == int(target_root)
        )
    write_csv(root_scan, state_dir / "root_scan.csv")

    singles = single_excitation_configurations(ground_occ, target_configs)
    write_single_excitation_table(
        singles, active_irreps, inactive_count, state_dir / "single_excitation_configurations.csv"
    )
    total_single_weight = float(sum(item[1] for item in singles))
    if total_single_weight < float(args.warn_single_weight_below):
        print(
            f"WARNING: {target.key}: selected root {target_root} has total "
            f"single-excitation weight {total_single_weight:.3f}, below "
            f"the audit threshold {args.warn_single_weight_below:.3f}. "
            "Inspect root_scan.csv before accepting the state assignment.",
            file=sys.stderr,
        )

    ground_noons = active_natural_occupations(ground_mc, ground_ci, ground_nelec)
    target_noons = active_natural_occupations(target_mc, target_ci, target_nelec)
    write_natural_occupation_table(
        ground_noons, target_noons, state_dir / "active_natural_occupations.csv"
    )
    ss, effective_multiplicity = state_spin_square(target_mc, target_ci, target_nelec)

    row: Dict[str, Any] = {
        **asdict(target),
        "pyscf_target_wfnsym_used": target_irrep,
        "target_root_used": target_root,
        "sa_total_states_used": sa_total_states_used,
        "sa_target_nroots_used": sa_target_nroots_used,
        "casci_nroots_used": casci_nroots_used,
        "active_mos_1based": [int(value) for value in active_mos_1based],
        "selected_active_irrep_counts": {
            str(key): int(value) for key, value in active_irrep_counts.items()
        },
        "root_scan_file": str(state_dir / "root_scan.csv"),
        "basis": args.basis,
        "nevpt2_contraction": "strongly contracted (PySCF mrpt.NEVPT)",
        "active_space_electrons": nelecas_total,
        "active_space_orbitals": ncas,
        "inactive_doubly_occupied_orbitals": inactive_count,
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
        "single_excitation_found": bool(singles),
        "total_single_excitation_weight_relative_to_dominant_ground_configuration": total_single_weight,
        "ground_active_natural_occupations": [float(value) for value in ground_noons],
        "target_active_natural_occupations": [float(value) for value in target_noons],
    }

    if singles:
        single_occ, single_weight, hole_active, particle_active = singles[0]
        row.update(
            {
                "dominant_single_excitation_configuration": occupation_text(single_occ),
                "dominant_single_excitation_weight": single_weight,
            }
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
    else:
        print(
            f"WARNING: {target.key}: no single-electron spatial occupation was "
            "identified relative to the dominant ground configuration.",
            file=sys.stderr,
        )

    (state_dir / "state_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def run_case(args: argparse.Namespace) -> None:
    require_pyscf()
    case = json.loads(args.case.read_text())
    if int(case.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported case schema")
    registry_path = args.registry or Path(case["registry_file"])
    registry = load_registry(registry_path)
    molecule_name = str(case["molecule"])
    entry = registry_molecule(registry, molecule_name)
    all_states = target_states_from_entry(entry)
    selection = args.states
    if selection is None:
        wanted = set(case.get("selected_state_keys", []))
        states = [state for state in all_states if state.key in wanted]
    else:
        states = select_target_states(all_states, selection)
    if not states:
        raise ValueError("No states selected in case file")

    coordinates = args.coordinates
    record = find_named_molecule(parse_coordinate_file(coordinates), molecule_name)
    index, name, atoms = record
    basis = args.basis or str(case["basis"])
    unit = args.unit or str(case.get("unit", "Bohr"))
    charge = int(case.get("charge", entry.get("charge", 0)))
    symmetry = bool(case.get("symmetry", True))
    subgroup = args.symmetry_subgroup
    if subgroup is None:
        subgroup = case.get("symmetry_subgroup")

    outdir = args.outdir or (args.case.parent / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    mol = build_molecule(
        atoms,
        basis=basis,
        unit=unit,
        charge=charge,
        max_memory=args.max_memory,
        log_path=outdir / "pyscf.log",
        verbose=args.verbose,
        symmetry=symmetry,
        symmetry_subgroup=subgroup,
    )
    mf = run_rhf(
        mol,
        outdir / "rhf.chk",
        args.scf_conv_tol,
        args.scf_max_cycle,
        args.newton_fallback,
    )
    molden.from_scf(mf, str(outdir / "rhf_canonical_orbitals.molden"))
    write_csv(canonical_orbital_rows(mf), outdir / "canonical_orbitals.csv")

    # ``--active-mos`` is an explicit all-state override. Otherwise, each state
    # may provide its own list and fall back to the molecule-level default.
    cli_active_mos = parse_mo_list(args.active_mos)
    default_active_mos = [int(value) for value in case.get("active_mos_1based", [])]
    args.basis = basis  # ensure per-state summaries record the resolved basis

    ground_irrep = args.ground_wfnsym or str(case.get("ground_wfnsym") or totally_symmetric_irrep(mol))
    available_irreps = [str(value) for value in mol.irrep_name]
    if ground_irrep not in available_irreps:
        raise RuntimeError(
            f"Ground irrep {ground_irrep!r} is unavailable. Available: {available_irreps}"
        )

    overrides = dict(case.get("state_overrides", {}))
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for target in states:
        if target.state_type.upper() == "R" and not basis_looks_diffuse(basis):
            if not args.allow_rydberg_without_diffuse:
                print(
                    f"SKIP {target.key}: Rydberg state with non-diffuse basis {basis}. "
                    "Use an augmented basis or --allow-rydberg-without-diffuse.",
                    file=sys.stderr,
                )
                continue
        override = dict(overrides.get(target.key, {}))

        if cli_active_mos is not None:
            state_active_mos = list(cli_active_mos)
        else:
            raw_state_active = override.get("active_mos_1based", [])
            if isinstance(raw_state_active, str):
                state_active_mos = parse_mo_list(raw_state_active) or []
            else:
                state_active_mos = [int(value) for value in raw_state_active]
            if not state_active_mos:
                state_active_mos = list(default_active_mos)

        if not state_active_mos:
            failures.append(
                {
                    "key": target.key,
                    "error": (
                        "No active MOs are defined for this state. Set "
                        "state_overrides[STATE].active_mos_1based, the global "
                        "active_mos_1based, or pass --active-mos."
                    ),
                }
            )
            continue

        actual_irrep_counts = selected_mo_irrep_counts(
            mol, np.asarray(mf.mo_coeff), state_active_mos
        )
        expected_irrep_counts = override.get("expected_active_irrep_counts", {})
        try:
            validate_expected_irrep_counts(actual_irrep_counts, expected_irrep_counts)
            ncas, nelecas_total, mo_guess = prepare_active_space(mf, state_active_mos)
        except Exception as exc:
            failures.append({"key": target.key, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise
            continue

        sa_target_nroots = override.get("sa_target_nroots")
        if sa_target_nroots is not None:
            sa_target_nroots = int(sa_target_nroots)
        casci_nroots = override.get("casci_nroots")
        if casci_nroots is not None:
            casci_nroots = int(casci_nroots)

        target_irrep = override.get("pyscf_wfnsym", target.pyscf_wfnsym)
        target_root = override.get("target_root")
        if target_root is None:
            target_root = int(target.target_root_guess)
            if target.spin_2s == 0 and target_irrep == ground_irrep:
                target_root += 1
        target_root = int(target_root)
        if target_irrep is None:
            failures.append(
                {
                    "key": target.key,
                    "error": "pyscf_wfnsym is undefined; edit state_overrides in case.json",
                }
            )
            continue
        target_irrep = str(target_irrep)
        if target_irrep not in available_irreps:
            failures.append(
                {
                    "key": target.key,
                    "error": (
                        f"Target irrep {target_irrep!r} unavailable in {mol.groupname}; "
                        f"available={available_irreps}"
                    ),
                }
            )
            continue
        requested_sa_display = (
            int(sa_target_nroots) if sa_target_nroots is not None else "auto"
        )
        requested_casci_display = (
            int(casci_nroots) if casci_nroots is not None else "auto"
        )
        print(f"\n=== {target.key}: {target.label} ===", flush=True)
        print(
            f"Active space: CAS({nelecas_total},{ncas}); "
            f"RHF MOs={state_active_mos}; irreps={actual_irrep_counts}",
            flush=True,
        )
        print(
            f"Target: symmetry={target_irrep}, zero-based root={target_root}; "
            f"SA target roots={requested_sa_display}; "
            f"CASCI roots={requested_casci_display}",
            flush=True,
        )
        try:
            rows.append(
                run_target_state(
                    mf,
                    ncas,
                    nelecas_total,
                    mo_guess,
                    ground_irrep,
                    target,
                    target_irrep,
                    target_root,
                    state_active_mos,
                    actual_irrep_counts,
                    sa_target_nroots,
                    casci_nroots,
                    args,
                    outdir,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            failures.append({"key": target.key, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise

    write_csv(rows, outdir / "vertical_energies.csv")
    (outdir / "vertical_energies.json").write_text(json.dumps(rows, indent=2) + "\n")
    (outdir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    metadata = {
        "workflow_version": WORKFLOW_VERSION,
        "molecule": name,
        "coordinates_index": index,
        "formula": hill_formula(atoms),
        "basis": basis,
        "unit": unit,
        "charge": charge,
        "detected_topgroup": str(getattr(mol, "topgroup", "")),
        "detected_group": str(mol.groupname),
        "available_irreps": available_irreps,
        "ground_wfnsym": ground_irrep,
        "molecule_default_active_mos_1based": default_active_mos,
        "cli_active_mos_override": cli_active_mos,
        "state_active_spaces": {
            row["key"]: {
                "active_mos_1based": row.get("active_mos_1based"),
                "active_space_electrons": row.get("active_space_electrons"),
                "active_space_orbitals": row.get("active_space_orbitals"),
                "selected_active_irrep_counts": row.get("selected_active_irrep_counts"),
                "sa_target_nroots_used": row.get("sa_target_nroots_used"),
                "casci_nroots_used": row.get("casci_nroots_used"),
            }
            for row in rows
        },
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "successful_states": [row["key"] for row in rows],
        "failed_states": failures,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n=== Summary ===")
    print("state  CASCI/eV  SC-NEVPT2/eV  single-weight  S2")
    for row in rows:
        print(
            f"{row['key']}  {row['vertical_casci_eV']:.6f}  "
            f"{row['vertical_sc_nevpt2_eV']:.6f}  "
            f"{row['total_single_excitation_weight_relative_to_dominant_ground_configuration']:.6f}  "
            f"{row['target_S2']:.6f}"
        )
    if failures:
        print(f"{len(failures)} state(s) failed or require manual symmetry/root edits; see failures.json")
    print(f"Results written below {outdir.resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_calculation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-memory", type=int, default=4000, help="PySCF memory limit in MB")
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument(
        "--newton-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--symmetry-subgroup")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="General single-excitation SA-CASSCF/CASCI/SC-NEVPT2 driver"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List registered molecules or states")
    list_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    list_parser.add_argument("--molecule")

    prepare = sub.add_parser("prepare", help="Run RHF and write an editable case template")
    prepare.add_argument("coordinates", type=Path)
    prepare.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    prepare.add_argument("--molecule", required=True)
    prepare.add_argument("--states", default="valence", help="valence, rydberg, all, or comma-separated keys")
    prepare.add_argument("--basis", default="sto-3g")
    prepare.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    prepare.add_argument("--outdir", type=Path, default=Path("prepared_cases"))
    prepare.add_argument(
        "--suggest-frontier",
        help="Write a provisional NOCC,NVIRT frontier active space, e.g. 2,2",
    )
    add_common_calculation_arguments(prepare)

    run = sub.add_parser("run", help="Run a prepared molecule case")
    run.add_argument("coordinates", type=Path)
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--registry", type=Path)
    run.add_argument("--states", help="Override case state selection")
    run.add_argument("--basis", help="Override basis in case file")
    run.add_argument("--unit", choices=["Bohr", "Angstrom"], help="Override coordinate unit")
    run.add_argument("--outdir", type=Path)
    run.add_argument("--active-mos", help="Override active_mos_1based with comma-separated indices")
    run.add_argument("--ground-wfnsym")
    run.add_argument(
        "--allow-rydberg-without-diffuse",
        action="store_true",
        help="Permit Rydberg targets in a non-augmented basis (not physically recommended)",
    )
    run.add_argument("--casscf-conv-tol", type=float, default=1e-8)
    run.add_argument("--casscf-conv-tol-grad", type=float, default=3e-5)
    run.add_argument("--casscf-max-cycle-macro", type=int, default=100)
    run.add_argument("--spin-penalty", type=float, default=0.5)
    run.add_argument(
        "--warn-single-weight-below",
        type=float,
        default=0.70,
        help=(
            "Warn when the selected root has less than this total CASCI "
            "single-excitation configuration weight (default: 0.70)"
        ),
    )
    run.add_argument("--max-configurations", type=int, default=50)
    run.add_argument("--cubes", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--cube-resolution", type=float, default=0.18)
    run.add_argument("--cube-margin", type=float, default=4.0)
    run.add_argument("--slice-offset", type=float, default=0.70)
    run.add_argument("--slice-grid", type=int, default=181)
    run.add_argument("--slice-margin", type=float, default=3.0)
    run.add_argument("--fail-fast", action="store_true")
    add_common_calculation_arguments(run)

    return parser


def list_registry(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    if args.molecule:
        entry = registry_molecule(registry, args.molecule)
        print(args.molecule)
        for state in target_states_from_entry(entry):
            suffix = "minimal-basis valence target" if state.minimal_basis_recommended else "requires diffuse basis"
            print(
                f"  {state.key:42s} {state.label:28s} "
                f"TBE={state.paper_tbe_augccpvtz_eV:.2f}  "
                f"PC-NEVPT2={state.paper_pc_nevpt2_augccpvtz_eV:.2f}  {suffix}"
            )
        for note in entry.get("notes", []):
            print(f"  NOTE: {note}")
    else:
        print("#  Molecule  Valence states  Rydberg states")
        total_valence = 0
        total_rydberg = 0
        for name, entry in registry["molecules"].items():
            states = target_states_from_entry(entry)
            nval = sum(state.state_type == "V" for state in states)
            nryd = sum(state.state_type == "R" for state in states)
            total_valence += nval
            total_rydberg += nryd
            print(f"{entry['index']:2d}  {name:28s}  {nval:3d}  {nryd:3d}")
        print(
            f"\nRegistry totals: {len(registry['molecules'])} molecules; "
            f"{total_valence + total_rydberg} states "
            f"({total_valence} valence, {total_rydberg} Rydberg)."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list":
        list_registry(args)
    elif args.command == "prepare":
        prepare_case(args)
    elif args.command == "run":
        # The case file can specify symmetry=False, but the target workflow is
        # fundamentally symmetry-resolved.  Keep the CLI flag available for
        # diagnostics and fail clearly later if no irreps exist.
        run_case(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
