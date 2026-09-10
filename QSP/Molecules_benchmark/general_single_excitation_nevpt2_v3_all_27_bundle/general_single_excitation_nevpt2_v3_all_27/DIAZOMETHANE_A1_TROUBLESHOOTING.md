# Diazomethane A1 state: benchmark-aligned configuration

The first version of the general driver allowed only one molecule-level active
space and inferred the state-average size from the target root. That is not
sufficient for diazomethane because the benchmark uses a different active space
and state average for the singlet A1 valence state.

## Benchmark recipes

The symmetry order is `(A1, B1, B2, A2)`.

| Target | Active orbitals by irrep | State average | CASSCF/eV | SC-NEVPT2/eV | PC-NEVPT2/eV |
|---|---:|---:|---:|---:|---:|
| 1A2 pi->pi* | (4,3,2,0) | (1,0,0,1) | 3.271 | 3.134 | 3.088 |
| 1A1 pi->pi* | (4,4,2,0) | (3,0,0,0) | 5.653 | 6.281 | 6.232 |
| 3A2 pi->pi* | (4,3,2,0) | (1,0,0,1) | 3.023 | 2.882 | 2.835 |
| 3A1 pi->pi* | (4,3,2,0) | (2,0,0,0) | 4.268 | 4.116 | 4.068 |

The state-average counts include the singlet A1 ground state. Therefore, in the
v3 driver's convention:

- same-symmetry singlet 1A1: `sa_target_nroots = 3` (ground plus two excited A1 roots);
- A2 singlet or triplet: `sa_target_nroots = 1` (one target-block root plus the separately represented A1 ground state);
- A1 triplet: `sa_target_nroots = 1` (one triplet A1 root plus the separately represented singlet A1 ground state).

## Required edits to case.json

Use the nine-orbital `(4,3,2,0)` space as the molecule-level default. Insert the
actual one-based PySCF RHF MO indices after inspecting `canonical_orbitals.csv`
and `canonical_orbitals.molden`:

```json
"active_mos_1based": [
  0, 0, 0, 0,
  0, 0, 0,
  0, 0
]
```

Replace all zeros. The selected list must contain exactly four A1, three B1,
two B2, and zero A2 orbitals. For an all-valence choice this will normally
contain all occupied valence orbitals plus the relevant pi-star orbital; verify
the electron count printed by the driver rather than assuming it.

Then use the following state overrides. For the singlet A1 state, replace the
empty list with the ten-orbital `(4,4,2,0)` list. It is normally the nine-orbital
valence list plus the additional B1 virtual needed to represent the nearby A1
3p Rydberg state and its mixing with the valence A1 state.

```json
"state_overrides": {
  "s-a2-pi-to-pistar": {
    "pyscf_wfnsym": "A2",
    "target_root": 0,
    "active_mos_1based": [],
    "sa_target_nroots": 1,
    "casci_nroots": 1,
    "expected_active_irrep_counts": {
      "A1": 4,
      "B1": 3,
      "B2": 2,
      "A2": 0
    }
  },
  "s-a1-pi-to-pistar": {
    "pyscf_wfnsym": "A1",
    "target_root": 1,
    "active_mos_1based": [
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0
    ],
    "sa_target_nroots": 3,
    "casci_nroots": 3,
    "expected_active_irrep_counts": {
      "A1": 4,
      "B1": 4,
      "B2": 2,
      "A2": 0
    }
  },
  "t-a2-pi-to-pistar": {
    "pyscf_wfnsym": "A2",
    "target_root": 0,
    "active_mos_1based": [],
    "sa_target_nroots": 1,
    "casci_nroots": 1,
    "expected_active_irrep_counts": {
      "A1": 4,
      "B1": 3,
      "B2": 2,
      "A2": 0
    }
  },
  "t-a1-pi-to-pistar": {
    "pyscf_wfnsym": "A1",
    "target_root": 0,
    "active_mos_1based": [],
    "sa_target_nroots": 1,
    "casci_nroots": 1,
    "expected_active_irrep_counts": {
      "A1": 4,
      "B1": 3,
      "B2": 2,
      "A2": 0
    }
  }
}
```

Again, replace all zeros in the singlet-A1 list. Empty lists for the other
states deliberately inherit the nine-orbital molecule-level default.

## Run command

```bash
python 06_general_single_excitation_nevpt2_v3.py run Coordinates \
  --case prepared_cases/04_Diazomethane/case.json \
  --basis aug-cc-pvtz \
  --states s-a1-pi-to-pistar,t-a1-pi-to-pistar
```

## Root validation

Version 2 writes, for every state:

```text
STATE_DIRECTORY/root_scan.csv
```

For the singlet A1 calculation, inspect all three roots. The desired valence
state should be selected by its pi->pi-star hole-particle character, not by
energy alone. The target is normally zero-based root 1, but the valence and 3p
Rydberg A1 states can mix and may exchange character.

A selected singlet root with only about 0.36 total one-electron-promotion weight
is a strong warning that the benchmark valence state has not been isolated.
The QUEST single-excitation diagnostic for this state is about 91 percent at
CC3/aug-cc-pVTZ.

## PySCF warnings

`SymAdaptedCASCI does not have attributes chkfile` came from assigning an
SCF-style `chkfile` attribute to a CASCI object. Version 2 removes that
assignment and writes a compact JSON root-energy summary instead.

The message saying that the first state is used for the Fock matrix and natural
orbitals is produced by PySCF's post-CASCI canonicalization for a multiroot
calculation. It does not mean that only the first root was solved. The root
energies and CI vectors remain available. The benchmark mismatch was already
present at the CASSCF/CASCI level and is primarily an active-space/state-average
and state-identification problem.
