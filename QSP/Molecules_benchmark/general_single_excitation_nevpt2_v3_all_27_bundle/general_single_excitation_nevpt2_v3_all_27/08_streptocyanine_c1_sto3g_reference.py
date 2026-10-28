#!/usr/bin/env python3
"""STO-3G reference protocol for the lowest B2 excitations of Streptocyanine-C1.

This script is deliberately more rigorous than selecting a HOMO/LUMO window and
calling one CASPT2/NEVPT2 result a reference.  It provides three complementary
levels of calculation at the fixed ground-state geometry:

1. All-electron and frozen-core EOM-EE-CCSD root scans.  These calculations do
   not require an active-space choice and are the default practical reference
   for the strongly single-excitation-dominated 1B2 and 3B2 states.
2. A symmetry-resolved pi-space SA-CASSCF/CASCI/strongly-contracted NEVPT2
   calculation using the chemically complete three-orbital N-C-N pi manifold.
   For the supplied geometry and STO-3G orbitals, this is expected to be
   MO 11 (bonding pi, B1), MO 12 (nonbonding pi, A2), and MO 13 (pi*, B1).
3. Optional frozen-core full-valence FCI: CASCI(18e,17o), with the three 1s core
   orbitals frozen.  This is exact within the frozen-core STO-3G Hamiltonian,
   but it can require many GB of memory and substantial CPU time.

The target B2 state is identified by spatial symmetry, not by visual appearance
of a single orbital.  In C2v, A2 x B1 = B2, whereas B1 x B1 = A1.  Consequently,
when the orbital labels are (MO11,MO12,MO13)=(B1,A2,B1), the benchmark B2 state
is expected to be dominated by MO12 -> MO13.  MO11 -> MO13 is a legitimate
bonding-pi -> pi* excitation, but it belongs to the A1 symmetry block.

Examples
--------
Fast, recommended first run (EOM-CCSD + pi-space SC-NEVPT2):

    python 08_streptocyanine_c1_sto3g_reference.py Coordinates \
        --outdir streptocyanine_c1_sto3g_reference

Run the optional frozen-core full-valence FCI calculations as well:

    python 08_streptocyanine_c1_sto3g_reference.py Coordinates \
        --outdir streptocyanine_c1_sto3g_reference_fci \
        --run-frozen-core-fci --max-memory 64000

Override the three pi orbitals after inspecting rhf_orbitals.csv:

    python 08_streptocyanine_c1_sto3g_reference.py Coordinates \
        --pi-mos 11,12,13

Outputs
-------
* rhf_orbitals.csv
* eom_ccsd_all_electron_roots.csv
* eom_ccsd_frozen_core_roots.csv
* pi_space_results.csv
* reference_summary.csv
* recommended_reference.json
* optional frozen_core_fci_results.csv
* PySCF log/checkpoint/Molden files

Notes
-----
* PySCF's built-in mrpt.NEVPT is strongly contracted NEVPT2.
* The optional full-valence FCI freezes three heavy-atom 1s orbitals.  It is not
  all-electron FCI.
* If full-valence FCI is not run, the script calls all-electron EOM-CCSD the
  "practical reference", not an exact finite-basis result.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    import pyscf
    from pyscf import cc, fci, gto, mcscf, mrpt, scf, symm
    from pyscf.cc import eom_rccsd
    from pyscf.fci import cistring
    from pyscf.scf import stability as scf_stability
    from pyscf.tools import molden
except ImportError as exc:  # pragma: no cover - useful message on non-PySCF hosts
    raise SystemExit(
        "PySCF is required. Install it in the active environment, for example: "
        "python -m pip install pyscf"
    ) from exc


HARTREE_TO_EV = 27.211386245988
MOLECULE_NAME = "Streptocyanine-C1"
MOLECULE_INDEX = 8
EXPECTED_CHARGE = +1
EXPECTED_GROUP = "C2v"
TARGET_IRREP = "B2"
GROUND_IRREP = "A1"

# PySCF's conventional C2v ordering supports direct products through XOR.
C2V_ID = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
C2V_NAME = {value: key for key, value in C2V_ID.items()}

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


@dataclass
class EOMRoot:
    method: str
    multiplicity: int
    root: int
    converged: bool
    energy_Eh: float
    energy_eV: float
    r1_weight: float
    dominant_excitation_irrep: str
    dominant_irrep_fraction_of_r1: float
    b2_r1_weight: float
    b2_fraction_of_r1: float
    dominant_hole_mo_1based: int
    dominant_particle_mo_1based: int
    dominant_hole_irrep: str
    dominant_particle_irrep: str
    dominant_pair_irrep: str
    dominant_r1_amplitude: float
    dominant_r1_weight: float
    mo11_to_mo13_amplitude: float | None
    mo11_to_mo13_weight: float | None
    mo12_to_mo13_amplitude: float | None
    mo12_to_mo13_weight: float | None


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                records.append((current_index, str(current_name), current_atoms))
            current_index = int(header.group(1))
            current_name = header.group(2).strip()
            current_atoms = []
            continue
        atom_match = ATOM_RE.match(line)
        if atom_match and current_index is not None:
            symbol = atom_match.group(1)
            if symbol not in ATOMIC_NUMBER:
                raise ValueError(f"Unsupported element {symbol!r} at line {lineno}")
            xyz = tuple(float(atom_match.group(i)) for i in (2, 3, 4))
            current_atoms.append((symbol, xyz))
            continue
        if current_index is None:  # descriptive preamble
            continue
        raise ValueError(f"Could not parse line {lineno}: {raw!r}")

    if current_index is not None:
        if not current_atoms:
            raise ValueError(f"Molecule {current_index} has no atoms")
        records.append((current_index, str(current_name), current_atoms))
    if not records:
        raise ValueError(f"No molecule blocks found in {path}")
    return records


def find_streptocyanine(records: Sequence[MoleculeRecord]) -> MoleculeRecord:
    for record in records:
        index, name, _ = record
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        if index == MOLECULE_INDEX or normalized == "streptocyaninec1":
            return record
    raise ValueError(
        f"Could not find molecule {MOLECULE_INDEX} ({MOLECULE_NAME}) in coordinate file"
    )


def parse_mo_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(x) for x in token.split("-", 1))
            if hi < lo:
                lo, hi = hi, lo
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(token))
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate MO indices in {text!r}")
    return values


def normalize_roots(values: Any) -> List[Any]:
    if isinstance(values, (list, tuple)):
        return list(values)
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return [values]
    return [arr[i] for i in range(arr.shape[0])]


def normalize_energies(values: Any) -> List[float]:
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return [float(x) for x in arr]


def normalize_status(values: Any, nroots: int) -> List[bool]:
    if isinstance(values, (bool, np.bool_)):
        return [bool(values)] * nroots
    if values is None:
        return [True] * nroots
    seq = list(np.atleast_1d(values))
    if len(seq) == 1 and nroots > 1:
        return [bool(seq[0])] * nroots
    return [bool(x) for x in seq[:nroots]]


def c2v_product(irrep_a: str, irrep_b: str) -> str:
    try:
        return C2V_NAME[C2V_ID[irrep_a] ^ C2V_ID[irrep_b]]
    except KeyError as exc:
        raise ValueError(
            f"C2v product requested for unsupported irreps {irrep_a!r}, {irrep_b!r}"
        ) from exc


def mo_symmetry_labels(mol: gto.Mole, coeff: np.ndarray) -> List[str]:
    labels = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, coeff, check=False
    )
    return [str(value) for value in labels]


def core_count_from_atoms(atoms: Sequence[Atom]) -> int:
    # Conventional 1s core: one doubly occupied core orbital per C/N/O atom.
    # Streptocyanine-C1 has C + 2N -> three frozen core orbitals.
    return sum(1 for symbol, _ in atoms if symbol in {"C", "N", "O"})


# -----------------------------------------------------------------------------
# Molecule and RHF
# -----------------------------------------------------------------------------


def build_molecule(
    atoms: Sequence[Atom],
    basis: str,
    unit: str,
    max_memory: int,
    log_path: Path,
    verbose: int,
) -> gto.Mole:
    nelec = sum(ATOMIC_NUMBER[symbol] for symbol, _ in atoms) - EXPECTED_CHARGE
    if nelec % 2:
        raise ValueError(f"Expected a closed-shell even electron count; obtained {nelec}")

    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = unit
    mol.basis = basis
    mol.charge = EXPECTED_CHARGE
    mol.spin = 0
    mol.cart = False
    mol.symmetry = True
    mol.max_memory = max_memory
    mol.output = str(log_path)
    mol.verbose = verbose
    mol.build()

    if str(mol.groupname) != EXPECTED_GROUP:
        raise RuntimeError(
            f"Expected {EXPECTED_GROUP} symmetry for the supplied orientation, but PySCF "
            f"detected {mol.groupname}. Do not assign B2 roots until the axis convention is checked."
        )
    return mol


def run_rhf(mol: gto.Mole, outdir: Path, args: argparse.Namespace) -> scf.hf.SCF:
    mf = scf.RHF(mol)
    mf.chkfile = str(outdir / "rhf.chk")
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle
    mf.max_memory = args.max_memory
    mf.kernel()
    if not mf.converged and args.newton_fallback:
        dm = mf.make_rdm1() if mf.mo_coeff is not None else None
        mf = mf.newton()
        mf.chkfile = str(outdir / "rhf.chk")
        mf.conv_tol = args.scf_conv_tol
        mf.max_cycle = args.scf_max_cycle
        mf.max_memory = args.max_memory
        mf.kernel(dm)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def write_rhf_orbitals(mf: scf.hf.SCF, outdir: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    labels = mo_symmetry_labels(mf.mol, np.asarray(mf.mo_coeff))
    rows: List[Dict[str, Any]] = []
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    for i, (occ, energy, irrep) in enumerate(
        zip(mf.mo_occ, mf.mo_energy, labels), start=1
    ):
        if i <= nocc:
            frontier = "HOMO" if i == nocc else f"HOMO-{nocc-i}"
        else:
            frontier = "LUMO" if i == nocc + 1 else f"LUMO+{i-nocc-1}"
        rows.append(
            {
                "mo_1based": i,
                "frontier_label": frontier,
                "occupation": float(occ),
                "energy_Eh": float(energy),
                "energy_eV": float(energy * HARTREE_TO_EV),
                "irrep": irrep,
            }
        )
    write_csv(rows, outdir / "rhf_orbitals.csv")
    molden.from_scf(mf, str(outdir / "rhf_orbitals.molden"))
    return labels, rows


def run_stability(mf: scf.hf.SCF, outdir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not args.stability:
        return result
    _, internal = scf_stability.rhf_internal(
        mf,
        with_symmetry=False,
        return_status=True,
        nroots=args.stability_nroots,
        tol=args.stability_tol,
    )
    _, external = scf_stability.rhf_external(
        mf,
        with_symmetry=False,
        return_status=True,
        nroots=args.stability_nroots,
        tol=args.stability_tol,
    )
    result["rhf_internal_stable"] = bool(internal)
    result["rhf_to_uhf_stable"] = bool(external)
    (outdir / "rhf_stability.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# -----------------------------------------------------------------------------
# EOM-CCSD reference calculation and root characterization
# -----------------------------------------------------------------------------


def active_full_mo_indices(mycc: cc.ccsd.CCSD) -> Tuple[np.ndarray, np.ndarray]:
    nmo_full = np.asarray(mycc.mo_coeff).shape[1]
    mask = np.ones(nmo_full, dtype=bool)
    frozen = mycc.frozen
    if frozen is None:
        pass
    elif isinstance(frozen, (int, np.integer)):
        mask[: int(frozen)] = False
    else:
        mask[np.asarray(list(frozen), dtype=int)] = False
    active = np.where(mask)[0]
    nocc = int(mycc.nocc)
    if len(active) != int(mycc.nmo):
        raise RuntimeError(
            f"Internal frozen-orbital map inconsistency: active={len(active)}, CC nmo={mycc.nmo}"
        )
    return active[:nocc], active[nocc:]


def root_amplitude(
    r1: np.ndarray,
    occ_full: np.ndarray,
    vir_full: np.ndarray,
    hole_1based: int,
    particle_1based: int,
) -> float | None:
    hole0 = hole_1based - 1
    particle0 = particle_1based - 1
    i_match = np.where(occ_full == hole0)[0]
    a_match = np.where(vir_full == particle0)[0]
    if len(i_match) != 1 or len(a_match) != 1:
        return None
    return float(np.real_if_close(r1[int(i_match[0]), int(a_match[0])]))


def analyze_eom_roots(
    method: str,
    multiplicity: int,
    eom: Any,
    energies: Any,
    vectors: Any,
    mo_irreps: Sequence[str],
) -> List[EOMRoot]:
    energy_list = normalize_energies(energies)
    vector_list = normalize_roots(vectors)
    status_list = normalize_status(getattr(eom, "converged", None), len(energy_list))
    occ_full, vir_full = active_full_mo_indices(eom._cc)

    rows: List[EOMRoot] = []
    for root, (energy, vector, converged) in enumerate(
        zip(energy_list, vector_list, status_list)
    ):
        r1, _ = eom.vector_to_amplitudes(vector)
        r1 = np.asarray(r1)
        weights = np.abs(r1) ** 2
        r1_weight = float(weights.sum())

        by_irrep: MutableMapping[str, float] = defaultdict(float)
        for i, full_i in enumerate(occ_full):
            for a, full_a in enumerate(vir_full):
                product = c2v_product(mo_irreps[full_i], mo_irreps[full_a])
                by_irrep[product] += float(weights[i, a])

        dominant_irrep = max(by_irrep, key=by_irrep.get) if by_irrep else ""
        dominant_fraction = (
            float(by_irrep[dominant_irrep] / r1_weight) if r1_weight > 0 else 0.0
        )
        b2_weight = float(by_irrep.get(TARGET_IRREP, 0.0))
        b2_fraction = b2_weight / r1_weight if r1_weight > 0 else 0.0

        flat = int(np.argmax(weights))
        i_dom, a_dom = np.unravel_index(flat, weights.shape)
        full_i = int(occ_full[i_dom])
        full_a = int(vir_full[a_dom])
        hole_irrep = mo_irreps[full_i]
        particle_irrep = mo_irreps[full_a]
        pair_irrep = c2v_product(hole_irrep, particle_irrep)

        amp_11_13 = root_amplitude(r1, occ_full, vir_full, 11, 13)
        amp_12_13 = root_amplitude(r1, occ_full, vir_full, 12, 13)

        rows.append(
            EOMRoot(
                method=method,
                multiplicity=multiplicity,
                root=root,
                converged=converged,
                energy_Eh=energy,
                energy_eV=energy * HARTREE_TO_EV,
                r1_weight=r1_weight,
                dominant_excitation_irrep=dominant_irrep,
                dominant_irrep_fraction_of_r1=dominant_fraction,
                b2_r1_weight=b2_weight,
                b2_fraction_of_r1=b2_fraction,
                dominant_hole_mo_1based=full_i + 1,
                dominant_particle_mo_1based=full_a + 1,
                dominant_hole_irrep=hole_irrep,
                dominant_particle_irrep=particle_irrep,
                dominant_pair_irrep=pair_irrep,
                dominant_r1_amplitude=float(np.real_if_close(r1[i_dom, a_dom])),
                dominant_r1_weight=float(weights[i_dom, a_dom]),
                mo11_to_mo13_amplitude=amp_11_13,
                mo11_to_mo13_weight=None if amp_11_13 is None else amp_11_13**2,
                mo12_to_mo13_amplitude=amp_12_13,
                mo12_to_mo13_weight=None if amp_12_13 is None else amp_12_13**2,
            )
        )
    return rows


def choose_b2_root(rows: Sequence[EOMRoot], purity_threshold: float) -> EOMRoot:
    strict = [
        row
        for row in rows
        if row.converged
        and row.dominant_excitation_irrep == TARGET_IRREP
        and row.b2_fraction_of_r1 >= purity_threshold
    ]
    if strict:
        return min(strict, key=lambda row: row.energy_Eh)
    converged = [row for row in rows if row.converged]
    if not converged:
        raise RuntimeError("No converged EOM roots")
    # Fallback: maximize the B2 part of R1, then prefer lower energy.
    best = max(converged, key=lambda row: (row.b2_r1_weight, -row.energy_Eh))
    print(
        f"WARNING: no EOM root passed B2 purity threshold {purity_threshold:.2f}; "
        f"using root {best.root} with B2 fraction {best.b2_fraction_of_r1:.3f}",
        file=sys.stderr,
    )
    return best


def run_eom_ccsd(
    mf: scf.hf.SCF,
    frozen: int | None,
    method_label: str,
    mo_irreps: Sequence[str],
    outdir: Path,
    args: argparse.Namespace,
) -> Tuple[List[EOMRoot], Dict[int, EOMRoot], Dict[str, Any]]:
    mycc = cc.RCCSD(mf, frozen=frozen)
    mycc.conv_tol = args.cc_conv_tol
    mycc.max_cycle = args.cc_max_cycle
    mycc.max_memory = args.max_memory
    e_corr, _, _ = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError(f"{method_label}: RCCSD did not converge")

    all_rows: List[EOMRoot] = []
    selected: Dict[int, EOMRoot] = {}
    for multiplicity, eom_cls in (
        (1, eom_rccsd.EOMEESinglet),
        (3, eom_rccsd.EOMEETriplet),
    ):
        eom = eom_cls(mycc)
        eom.conv_tol = args.eom_conv_tol
        eom.max_cycle = args.eom_max_cycle
        eom.max_space = args.eom_max_space
        energies, vectors = eom.kernel(nroots=args.eom_nroots)
        rows = analyze_eom_roots(
            method_label, multiplicity, eom, energies, vectors, mo_irreps
        )
        all_rows.extend(rows)
        selected[multiplicity] = choose_b2_root(rows, args.eom_symmetry_purity)

    write_csv(
        [asdict(row) for row in all_rows],
        outdir / f"{method_label.replace('-', '_').lower()}_roots.csv",
    )
    metadata = {
        "method": method_label,
        "frozen_orbitals": frozen,
        "rhf_energy_Eh": float(mf.e_tot),
        "ccsd_correlation_energy_Eh": float(e_corr),
        "ccsd_total_energy_Eh": float(mycc.e_tot),
        "ccsd_t1_diagnostic": (
            float(mycc.get_t1_diagnostic())
            if hasattr(mycc, "get_t1_diagnostic")
            else None
        ),
    }
    return all_rows, selected, metadata


# -----------------------------------------------------------------------------
# Pi-space SA-CASSCF/CASCI/SC-NEVPT2
# -----------------------------------------------------------------------------


def make_fci_solver(
    mol: gto.Mole,
    spin_2s: int,
    wfnsym: str,
    nroots: int = 1,
    fix_spin_shift: float = 0.5,
) -> Any:
    if spin_2s == 0:
        solver = fci.direct_spin0_symm.FCI(mol)
    else:
        solver = fci.direct_spin1_symm.FCI(mol)
    solver.spin = spin_2s
    solver.wfnsym = wfnsym
    solver.nroots = nroots
    if spin_2s > 0:
        target_s = spin_2s / 2.0
        solver = fci.addons.fix_spin_(
            solver, shift=fix_spin_shift, ss=target_s * (target_s + 1.0)
        )
        solver.spin = spin_2s
        solver.wfnsym = wfnsym
        solver.nroots = nroots
    return solver


def apply_state_average_mix(mc: Any, solvers: Sequence[Any], weights: Sequence[float]) -> Any:
    if hasattr(mc, "state_average_mix_"):
        return mc.state_average_mix_(list(solvers), list(weights))
    return mcscf.state_average_mix_(mc, list(solvers), list(weights))


def root_value(value: Any, root: int) -> float:
    array = np.atleast_1d(value)
    if root >= len(array):
        raise IndexError(f"Requested root {root}; only {len(array)} values are present")
    return float(array[root])


def root_ci(mc: Any, root: int) -> np.ndarray:
    ci = mc.ci
    if isinstance(ci, (list, tuple)):
        return np.asarray(ci[root])
    arr = np.asarray(ci)
    if arr.ndim == 3:
        return np.asarray(arr[root])
    if root != 0:
        raise IndexError("Only one CI root is present")
    return arr


def spatial_configuration_weights(
    ci_vector: np.ndarray,
    ncas: int,
    nelec: Tuple[int, int],
) -> List[Tuple[Tuple[int, ...], float]]:
    nalpha, nbeta = nelec
    alpha_occs = np.asarray(cistring.gen_occslst(range(ncas), nalpha), dtype=int)
    beta_occs = np.asarray(cistring.gen_occslst(range(ncas), nbeta), dtype=int)
    ci = np.asarray(ci_vector).reshape(len(alpha_occs), len(beta_occs))
    grouped: MutableMapping[Tuple[int, ...], float] = defaultdict(float)
    for ia, alpha_occ in enumerate(alpha_occs):
        for ib, beta_occ in enumerate(beta_occs):
            weight = float(abs(ci[ia, ib]) ** 2)
            if weight < 1e-14:
                continue
            occupation = np.zeros(ncas, dtype=int)
            occupation[alpha_occ] += 1
            occupation[beta_occ] += 1
            grouped[tuple(int(value) for value in occupation)] += weight
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def single_excitation_list(
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
            results.append((occupation, weight, int(holes[0]), int(particles[0])))
    return results


def occupation_text(values: Sequence[int]) -> str:
    return "[" + " ".join(str(int(x)) for x in values) + "]"


def run_casci_state(
    mf: scf.hf.SCF,
    ncas: int,
    nelec: Tuple[int, int],
    ncore: int,
    common_mo: np.ndarray,
    spin_2s: int,
    wfnsym: str,
    nroots: int = 1,
) -> Any:
    cas = mcscf.CASCI(mf, ncas, nelec, ncore=ncore)
    cas.max_memory = mf.max_memory
    cas.fcisolver = make_fci_solver(mf.mol, spin_2s, wfnsym, nroots=nroots)
    cas.kernel(common_mo)
    converged = getattr(cas, "converged", True)
    if not bool(np.all(np.asarray(converged))):
        raise RuntimeError(f"CASCI failed for 2S={spin_2s}, symmetry={wfnsym}")
    return cas


def nevpt2_total(mc: Any, root: int = 0) -> Tuple[float, float, float]:
    ref = root_value(mc.e_tot, root)
    corr = float(mrpt.NEVPT(mc, root=root).kernel())
    return ref, corr, ref + corr


def run_pi_space_state(
    mf: scf.hf.SCF,
    pi_mos: Sequence[int],
    mo_irreps: Sequence[str],
    multiplicity: int,
    outdir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    state_dir = outdir / ("pi_space_1B2" if multiplicity == 1 else "pi_space_3B2")
    state_dir.mkdir(parents=True, exist_ok=True)

    ncas = 3
    nelecas_total = 4
    spin_2s = multiplicity - 1
    target_nelec = (2, 2) if multiplicity == 1 else (3, 1)

    mc = mcscf.CASSCF(mf, ncas, nelecas_total)
    mc.conv_tol = args.casscf_conv_tol
    mc.conv_tol_grad = args.casscf_conv_tol_grad
    mc.max_cycle_macro = args.casscf_max_cycle
    mc.max_memory = args.max_memory
    mo_guess = mcscf.sort_mo(mc, mf.mo_coeff, list(pi_mos), base=1)

    ground_solver = make_fci_solver(mf.mol, 0, GROUND_IRREP, 1)
    target_solver = make_fci_solver(mf.mol, spin_2s, TARGET_IRREP, 1)
    mc = apply_state_average_mix(mc, [ground_solver, target_solver], [0.5, 0.5])
    mc.kernel(mo_guess)
    if not mc.converged:
        raise RuntimeError(f"SA-CASSCF did not converge for {multiplicity}B2")

    common_mo = np.asarray(mc.mo_coeff)
    ncore = int(mc.ncore)
    molden.from_mo(
        mf.mol,
        str(state_dir / "sa_casscf_orbitals.molden"),
        common_mo,
    )

    ground = run_casci_state(
        mf, ncas, (2, 2), ncore, common_mo, 0, GROUND_IRREP, 1
    )
    target = run_casci_state(
        mf, ncas, target_nelec, ncore, common_mo, spin_2s, TARGET_IRREP, 1
    )

    g_ref, g_corr, g_total = nevpt2_total(ground)
    t_ref, t_corr, t_total = nevpt2_total(target)

    g_ci = root_ci(ground, 0)
    t_ci = root_ci(target, 0)
    g_cfg = spatial_configuration_weights(g_ci, ncas, (2, 2))
    t_cfg = spatial_configuration_weights(t_ci, ncas, target_nelec)
    singles = single_excitation_list(g_cfg[0][0], t_cfg)

    active_coeff = common_mo[:, ncore : ncore + ncas]
    active_irreps = mo_symmetry_labels(mf.mol, active_coeff)
    active_rows = []
    for iact, irrep in enumerate(active_irreps):
        # Identify the largest overlap with an original RHF MO to retain a useful label.
        overlap = mf.get_ovlp()
        projections = np.abs(mf.mo_coeff.T.conj() @ overlap @ active_coeff[:, iact]) ** 2
        parent = int(np.argmax(projections))
        active_rows.append(
            {
                "active_orbital_1based": iact + 1,
                "parent_rhf_mo_1based": parent + 1,
                "parent_overlap2": float(projections[parent]),
                "irrep": irrep,
            }
        )
    write_csv(active_rows, state_dir / "active_orbitals.csv")

    dominant_single: Dict[str, Any] = {}
    total_single_weight = float(sum(item[1] for item in singles))
    if singles:
        occupation, weight, hole, particle = singles[0]
        dominant_single = {
            "dominant_single_configuration": occupation_text(occupation),
            "dominant_single_weight": float(weight),
            "hole_active_orbital_1based": hole + 1,
            "particle_active_orbital_1based": particle + 1,
            "hole_parent_rhf_mo_1based": active_rows[hole]["parent_rhf_mo_1based"],
            "particle_parent_rhf_mo_1based": active_rows[particle]["parent_rhf_mo_1based"],
            "hole_irrep": active_irreps[hole],
            "particle_irrep": active_irreps[particle],
            "excitation_irrep": c2v_product(active_irreps[hole], active_irreps[particle]),
        }

    ss, effective_mult = target.fcisolver.spin_square(t_ci, ncas, target_nelec)
    row: Dict[str, Any] = {
        "method": "pi-space SA-CASSCF/CASCI/SC-NEVPT2",
        "state": f"{multiplicity}B2",
        "multiplicity": multiplicity,
        "active_space": "CAS(4e,3o)",
        "input_rhf_pi_mos_1based": list(pi_mos),
        "input_rhf_pi_irreps": [mo_irreps[i - 1] for i in pi_mos],
        "ground_casci_Eh": g_ref,
        "target_casci_Eh": t_ref,
        "vertical_casci_eV": (t_ref - g_ref) * HARTREE_TO_EV,
        "ground_sc_nevpt2_correction_Eh": g_corr,
        "target_sc_nevpt2_correction_Eh": t_corr,
        "vertical_sc_nevpt2_eV": (t_total - g_total) * HARTREE_TO_EV,
        "dominant_ground_configuration": occupation_text(g_cfg[0][0]),
        "dominant_ground_weight": float(g_cfg[0][1]),
        "dominant_target_configuration": occupation_text(t_cfg[0][0]),
        "dominant_target_weight": float(t_cfg[0][1]),
        "total_single_excitation_weight": total_single_weight,
        "target_S2": float(ss),
        "target_effective_multiplicity": float(effective_mult),
        **dominant_single,
    }
    (state_dir / "state_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def validate_pi_mos(
    mf: scf.hf.SCF,
    pi_mos: Sequence[int],
    mo_irreps: Sequence[str],
) -> None:
    if len(pi_mos) != 3:
        raise ValueError("Exactly three pi MOs are required, in bonding/nonbonding/antibonding order")
    nmo = mf.mo_coeff.shape[1]
    if min(pi_mos) < 1 or max(pi_mos) > nmo:
        raise ValueError(f"Invalid pi MO list {pi_mos}; NMO={nmo}")
    occ = [float(mf.mo_occ[i - 1]) for i in pi_mos]
    irreps = [mo_irreps[i - 1] for i in pi_mos]
    if occ != [2.0, 2.0, 0.0]:
        raise RuntimeError(
            f"Expected pi orbital occupations [2,2,0], obtained {occ} for MOs {pi_mos}"
        )
    if irreps != ["B1", "A2", "B1"]:
        raise RuntimeError(
            "For this C2v orientation, the expected bonding/nonbonding/antibonding "
            f"pi irreps are [B1,A2,B1], but MOs {pi_mos} have {irreps}. "
            "Inspect rhf_orbitals.csv and override --pi-mos if needed."
        )


# -----------------------------------------------------------------------------
# Optional frozen-core full-valence FCI
# -----------------------------------------------------------------------------


def determinant_irrep_counts(orbsym_ids: Sequence[int], nelec: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for occupation in itertools.combinations(range(len(orbsym_ids)), nelec):
        irrep = 0
        for orbital in occupation:
            irrep ^= int(orbsym_ids[orbital])
        counts[irrep] += 1
    return counts


def determinant_count_for_wfnsym(
    orbsym_names: Sequence[str],
    nelec: Tuple[int, int],
    wfnsym: str,
) -> int:
    ids = [C2V_ID[name] for name in orbsym_names]
    alpha = determinant_irrep_counts(ids, nelec[0])
    beta = determinant_irrep_counts(ids, nelec[1])
    target = C2V_ID[wfnsym]
    return sum(alpha[ia] * beta[ia ^ target] for ia in C2V_ID.values())


def run_full_valence_fci_state(
    mf: scf.hf.SCF,
    ncore: int,
    multiplicity: int,
    wfnsym: str,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    nmo = mf.mo_coeff.shape[1]
    ncas = nmo - ncore
    nelecas_total = mf.mol.nelectron - 2 * ncore
    if multiplicity == 1:
        nelec = (nelecas_total // 2, nelecas_total // 2)
        spin_2s = 0
    elif multiplicity == 3:
        nelec = (nelecas_total // 2 + 1, nelecas_total // 2 - 1)
        spin_2s = 2
    else:
        raise ValueError("Only singlet and triplet FCI states are supported")

    cas = mcscf.CASCI(mf, ncas, nelec, ncore=ncore)
    solver = make_fci_solver(mf.mol, spin_2s, wfnsym, 1)
    solver.conv_tol = args.fci_conv_tol
    solver.max_cycle = args.fci_max_cycle
    solver.max_space = args.fci_max_space
    solver.pspace_size = args.fci_pspace_size
    cas.fcisolver = solver
    cas.max_memory = args.max_memory
    cas.kernel(mf.mo_coeff)
    converged = getattr(cas, "converged", True)
    if not bool(np.all(np.asarray(converged))):
        raise RuntimeError(f"Frozen-core FCI did not converge for {multiplicity}{wfnsym}")
    ci = root_ci(cas, 0)
    ss, effective_mult = cas.fcisolver.spin_square(ci, ncas, nelec)
    return float(cas.e_tot), float(ss), float(effective_mult)


def run_frozen_core_fci(
    mf: scf.hf.SCF,
    ncore: int,
    mo_irreps: Sequence[str],
    outdir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    nmo = mf.mo_coeff.shape[1]
    active_irreps = list(mo_irreps[ncore:])
    ncas = nmo - ncore
    nelecas_total = mf.mol.nelectron - 2 * ncore
    singlet_nelec = (nelecas_total // 2, nelecas_total // 2)
    triplet_nelec = (nelecas_total // 2 + 1, nelecas_total // 2 - 1)

    estimates = []
    for state, nelec, wfnsym in (
        ("1A1", singlet_nelec, GROUND_IRREP),
        ("1B2", singlet_nelec, TARGET_IRREP),
        ("3B2", triplet_nelec, TARGET_IRREP),
    ):
        ndet = determinant_count_for_wfnsym(active_irreps, nelec, wfnsym)
        vector_gib = ndet * 8 / 1024**3
        estimates.append(
            {
                "state": state,
                "ncore": ncore,
                "ncas": ncas,
                "nelecas": list(nelec),
                "wfnsym": wfnsym,
                "Ms_determinants_in_target_irrep": ndet,
                "one_real_CI_vector_GiB": vector_gib,
                "rough_Davidson_workspace_GiB": vector_gib * (args.fci_max_space + 4),
            }
        )
    write_csv(estimates, outdir / "frozen_core_fci_size_estimates.csv")

    print("\n=== Frozen-core full-valence FCI size estimate ===")
    for row in estimates:
        print(
            f"{row['state']}: {row['Ms_determinants_in_target_irrep']:,} determinants; "
            f"one vector ~{row['one_real_CI_vector_GiB']:.2f} GiB; "
            f"rough workspace ~{row['rough_Davidson_workspace_GiB']:.1f} GiB"
        )

    if not args.run_frozen_core_fci:
        return {}, []

    print("\nRunning frozen-core full-valence FCI. This can be expensive.")
    ground_energy, ground_s2, ground_mult = run_full_valence_fci_state(
        mf, ncore, 1, GROUND_IRREP, args
    )
    rows: List[Dict[str, Any]] = [
        {
            "state": "1A1",
            "multiplicity": 1,
            "wfnsym": GROUND_IRREP,
            "total_energy_Eh": ground_energy,
            "S2": ground_s2,
            "effective_multiplicity": ground_mult,
            "vertical_energy_eV": 0.0,
        }
    ]
    excitations: Dict[int, float] = {}
    for multiplicity in (1, 3):
        energy, ss, mult = run_full_valence_fci_state(
            mf, ncore, multiplicity, TARGET_IRREP, args
        )
        excitation = (energy - ground_energy) * HARTREE_TO_EV
        excitations[multiplicity] = excitation
        rows.append(
            {
                "state": f"{multiplicity}B2",
                "multiplicity": multiplicity,
                "wfnsym": TARGET_IRREP,
                "total_energy_Eh": energy,
                "S2": ss,
                "effective_multiplicity": mult,
                "vertical_energy_eV": excitation,
            }
        )
    write_csv(rows, outdir / "frozen_core_fci_results.csv")
    return excitations, rows


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STO-3G reference protocol for Streptocyanine-C1 1B2/3B2 excitations"
    )
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("streptocyanine_c1_sto3g_reference"))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    parser.add_argument(
        "--pi-mos",
        default="11,12,13",
        help="One-based bonding-pi, nonbonding-pi, pi* RHF MOs (default: 11,12,13)",
    )
    parser.add_argument("--max-memory", type=int, default=16000, help="PySCF memory limit in MB")
    parser.add_argument("--verbose", type=int, default=4)

    parser.add_argument("--scf-conv-tol", type=float, default=1e-12)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument(
        "--newton-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stability-nroots", type=int, default=3)
    parser.add_argument("--stability-tol", type=float, default=1e-8)

    parser.add_argument("--cc-conv-tol", type=float, default=1e-10)
    parser.add_argument("--cc-max-cycle", type=int, default=200)
    parser.add_argument("--eom-conv-tol", type=float, default=1e-9)
    parser.add_argument("--eom-max-cycle", type=int, default=200)
    parser.add_argument("--eom-max-space", type=int, default=50)
    parser.add_argument("--eom-nroots", type=int, default=8)
    parser.add_argument("--eom-symmetry-purity", type=float, default=0.80)

    parser.add_argument("--casscf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--casscf-conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--casscf-max-cycle", type=int, default=200)
    parser.add_argument(
        "--run-pi-nevpt2", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--run-frozen-core-fci",
        action="store_true",
        help="Run exact frozen-core full-valence FCI after printing the size estimate",
    )
    parser.add_argument(
        "--ncore", type=int, default=None, help="Frozen core orbitals for FCI/FC-EOM (default: one per C/N/O)"
    )
    parser.add_argument("--fci-conv-tol", type=float, default=1e-9)
    parser.add_argument("--fci-max-cycle", type=int, default=100)
    parser.add_argument("--fci-max-space", type=int, default=8)
    parser.add_argument("--fci-pspace-size", type=int, default=400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.basis.lower().replace("_", "-") not in {"sto-3g", "sto3g"}:
        print(
            f"WARNING: this protocol was designed to establish an STO-3G reference; "
            f"the requested basis is {args.basis!r}.",
            file=sys.stderr,
        )

    index, name, atoms = find_streptocyanine(parse_coordinate_file(args.coordinates))
    if index != MOLECULE_INDEX:
        print(f"WARNING: matched {name!r} at index {index}, expected index 8", file=sys.stderr)

    mol = build_molecule(
        atoms,
        args.basis,
        args.unit,
        args.max_memory,
        args.outdir / "pyscf.log",
        args.verbose,
    )
    ncore = args.ncore if args.ncore is not None else core_count_from_atoms(atoms)
    if ncore != 3:
        print(
            f"WARNING: conventional C/N 1s core count for this molecule is 3, but ncore={ncore}",
            file=sys.stderr,
        )

    mf = run_rhf(mol, args.outdir, args)
    mo_irreps, orbital_rows = write_rhf_orbitals(mf, args.outdir)
    stability = run_stability(mf, args.outdir, args)

    pi_mos = parse_mo_list(args.pi_mos)
    validate_pi_mos(mf, pi_mos, mo_irreps)

    # Print the decisive symmetry products explicitly.
    mo11_ir = mo_irreps[10]
    mo12_ir = mo_irreps[11]
    mo13_ir = mo_irreps[12]
    print("\n=== RHF pi-manifold assignment ===")
    print(f"MO 11: occ={mf.mo_occ[10]:.1f}, irrep={mo11_ir}  (bonding pi candidate)")
    print(f"MO 12: occ={mf.mo_occ[11]:.1f}, irrep={mo12_ir}  (nonbonding pi candidate)")
    print(f"MO 13: occ={mf.mo_occ[12]:.1f}, irrep={mo13_ir}  (pi* candidate)")
    print(f"MO 11 -> MO 13: {mo11_ir} x {mo13_ir} = {c2v_product(mo11_ir, mo13_ir)}")
    print(f"MO 12 -> MO 13: {mo12_ir} x {mo13_ir} = {c2v_product(mo12_ir, mo13_ir)}")

    # Active-space-independent practical reference.
    ae_rows, ae_selected, ae_meta = run_eom_ccsd(
        mf,
        frozen=None,
        method_label="AE-EOM-CCSD",
        mo_irreps=mo_irreps,
        outdir=args.outdir,
        args=args,
    )
    fc_rows, fc_selected, fc_meta = run_eom_ccsd(
        mf,
        frozen=ncore,
        method_label="FC-EOM-CCSD",
        mo_irreps=mo_irreps,
        outdir=args.outdir,
        args=args,
    )

    pi_rows: List[Dict[str, Any]] = []
    if args.run_pi_nevpt2:
        for multiplicity in (1, 3):
            pi_rows.append(
                run_pi_space_state(
                    mf, pi_mos, mo_irreps, multiplicity, args.outdir, args
                )
            )
        write_csv(pi_rows, args.outdir / "pi_space_results.csv")

    # Always print the FCI size estimate; execute only when explicitly requested.
    fci_exc, _ = run_frozen_core_fci(
        mf, ncore, mo_irreps, args.outdir, args
    )

    summary_rows: List[Dict[str, Any]] = []
    for multiplicity in (1, 3):
        state = f"{multiplicity}B2"
        ae = ae_selected[multiplicity]
        fc = fc_selected[multiplicity]
        summary_rows.extend(
            [
                {
                    "state": state,
                    "method": "AE-EOM-CCSD",
                    "vertical_energy_eV": ae.energy_eV,
                    "status": "practical active-space-independent reference",
                    "dominant_transition": (
                        f"MO{ae.dominant_hole_mo_1based}({ae.dominant_hole_irrep})->"
                        f"MO{ae.dominant_particle_mo_1based}({ae.dominant_particle_irrep})"
                    ),
                    "single_excitation_R1_weight": ae.r1_weight,
                },
                {
                    "state": state,
                    "method": "FC-EOM-CCSD",
                    "vertical_energy_eV": fc.energy_eV,
                    "status": f"frozen {ncore} core orbitals",
                    "dominant_transition": (
                        f"MO{fc.dominant_hole_mo_1based}({fc.dominant_hole_irrep})->"
                        f"MO{fc.dominant_particle_mo_1based}({fc.dominant_particle_irrep})"
                    ),
                    "single_excitation_R1_weight": fc.r1_weight,
                },
            ]
        )
        for row in pi_rows:
            if int(row["multiplicity"]) == multiplicity:
                summary_rows.append(
                    {
                        "state": state,
                        "method": "pi-CAS(4,3)-SC-NEVPT2",
                        "vertical_energy_eV": row["vertical_sc_nevpt2_eV"],
                        "status": "active-space method; diagnostic, not basis-exact",
                        "dominant_transition": (
                            f"RHF-parent MO{row.get('hole_parent_rhf_mo_1based')}->"
                            f"MO{row.get('particle_parent_rhf_mo_1based')}"
                        ),
                        "single_excitation_R1_weight": row["total_single_excitation_weight"],
                    }
                )
        if multiplicity in fci_exc:
            summary_rows.append(
                {
                    "state": state,
                    "method": "FC-FCI/STO-3G",
                    "vertical_energy_eV": fci_exc[multiplicity],
                    "status": "exact within the frozen-core STO-3G Hamiltonian",
                    "dominant_transition": "state selected by B2 symmetry",
                    "single_excitation_R1_weight": None,
                }
            )
            core_shift = ae.energy_eV - fc.energy_eV
            summary_rows.append(
                {
                    "state": state,
                    "method": "FC-FCI + EOM-CCSD core correction",
                    "vertical_energy_eV": fci_exc[multiplicity] + core_shift,
                    "status": "recommended near-exact all-electron STO-3G composite",
                    "dominant_transition": "state selected by B2 symmetry",
                    "single_excitation_R1_weight": None,
                }
            )

    write_csv(summary_rows, args.outdir / "reference_summary.csv")

    recommendation: Dict[str, Any] = {
        "molecule": name,
        "basis": args.basis,
        "geometry_unit": args.unit,
        "charge": EXPECTED_CHARGE,
        "point_group": mol.groupname,
        "electron_count": mol.nelectron,
        "nmo": mol.nao_nr(),
        "frozen_core_orbitals": ncore,
        "pi_mos_1based": pi_mos,
        "pi_mo_irreps": [mo_irreps[i - 1] for i in pi_mos],
        "rhf_stability": stability,
        "eom_ccsd_metadata": {"all_electron": ae_meta, "frozen_core": fc_meta},
        "states": {},
        "interpretation": (
            "For the detected C2v orientation, MO12(A2)->MO13(B1) has B2 symmetry; "
            "MO11(B1)->MO13(B1) has A1 symmetry."
        ),
    }
    for multiplicity in (1, 3):
        state = f"{multiplicity}B2"
        ae = ae_selected[multiplicity]
        fc = fc_selected[multiplicity]
        state_rec: Dict[str, Any] = {
            "AE_EOM_CCSD_eV": ae.energy_eV,
            "FC_EOM_CCSD_eV": fc.energy_eV,
            "EOM_core_correlation_shift_eV": ae.energy_eV - fc.energy_eV,
            "AE_EOM_dominant_transition": (
                f"MO{ae.dominant_hole_mo_1based}({ae.dominant_hole_irrep})->"
                f"MO{ae.dominant_particle_mo_1based}({ae.dominant_particle_irrep})"
            ),
            "AE_EOM_R1_weight": ae.r1_weight,
        }
        if multiplicity in fci_exc:
            state_rec["FC_FCI_eV"] = fci_exc[multiplicity]
            state_rec["recommended_all_electron_composite_eV"] = (
                fci_exc[multiplicity] + ae.energy_eV - fc.energy_eV
            )
            state_rec["recommended_method"] = "FC-FCI + EOM-CCSD core correction"
        else:
            state_rec["recommended_practical_reference_eV"] = ae.energy_eV
            state_rec["recommended_method"] = "all-electron EOM-CCSD"
            state_rec["qualification"] = (
                "High-level and active-space independent, but not exact FCI in the full STO-3G basis."
            )
        recommendation["states"][state] = state_rec

    (args.outdir / "recommended_reference.json").write_text(
        json.dumps(recommendation, indent=2) + "\n"
    )
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(
            {
                "pyscf_version": getattr(pyscf, "__version__", "unknown"),
                "coordinates_file": str(args.coordinates.resolve()),
                "arguments": vars(args) | {"coordinates": str(args.coordinates), "outdir": str(args.outdir)},
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    print("\n=== STO-3G reference summary ===")
    print("state  method                              energy/eV  dominant transition")
    for row in summary_rows:
        print(
            f"{row['state']:5s}  {row['method']:<34s} "
            f"{float(row['vertical_energy_eV']):10.6f}  {row['dominant_transition']}"
        )
    print(f"\nResults written below: {args.outdir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
