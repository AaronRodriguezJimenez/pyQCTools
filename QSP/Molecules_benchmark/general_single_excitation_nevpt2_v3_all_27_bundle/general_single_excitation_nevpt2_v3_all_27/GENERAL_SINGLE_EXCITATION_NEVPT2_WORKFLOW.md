# General PySCF workflow for the 27 single-excitation-dominated benchmark molecules

## Scope

This bundle covers the conservative molecule-level set discussed in the project:
27 molecules and 202 reported singlet/triplet states.  The registry contains 143
valence states and 59 Rydberg states.  Furan and Acetaldehyde are included in the
general registry and their specialized scripts are retained as worked examples.

The following eight molecules are deliberately excluded from this molecule-level
set because the benchmark explicitly identifies at least one pure, significant,
or mixed double-excitation state for each of them:

- Carbon trimer
- Nitrosomethane
- Acrolein
- Butadiene
- Glyoxal
- Cyclopentadiene
- Benzene
- Tetrazine

This is a conservative organizational choice.  It does not imply that every
reported state of an excluded molecule is double-excitation dominated.

## Files

- `06_general_single_excitation_nevpt2_v3.py`: configuration-driven engine.
- `benchmark_single_excitation_cases.json`: all 27 included molecules and 202
  benchmark states, with TBE and paper PC-NEVPT2 values.
- `04_furan_nevpt2_valence.py`: specialized Furan valence workflow.
- `05_acetaldehyde_nevpt2_npi.py`: specialized Acetaldehyde n-to-pi-star workflow.
- `DIAZOMETHANE_A1_TROUBLESHOOTING.md`: state-specific guidance for the difficult
  Diazomethane singlet A1 state.
- `BENCHMARK_SCOPE.md`: machine/project scope, counts, and molecule list.

The general engine intentionally separates the benchmark state metadata from the
active-space definition.  A single automatic active-space rule is not reliable
for all pi systems, lone-pair excitations, sigma-to-pi-star states, linear
molecules, degeneracies, and Rydberg states.

## 1. Inspect the registry

```bash
python 06_general_single_excitation_nevpt2_v3.py list
```

For one molecule:

```bash
python 06_general_single_excitation_nevpt2_v3.py list --molecule Furan
python 06_general_single_excitation_nevpt2_v3.py list --molecule Acetaldehyde
```

The default registry is expected beside the driver.  Use `--registry` to select
another registry.

## 2. Prepare a molecule

Example for Furan:

```bash
python 06_general_single_excitation_nevpt2_v3.py prepare Coordinates \
    --molecule Furan \
    --basis sto-3g \
    --states valence \
    --outdir prepared_cases
```

Example for Acetaldehyde:

```bash
python 06_general_single_excitation_nevpt2_v3.py prepare Coordinates \
    --molecule Acetaldehyde \
    --basis sto-3g \
    --states valence \
    --outdir prepared_cases
```

This is an RHF-only stage.  It writes a checkpoint, canonical-orbital table,
Molden file, and editable `case.json`.  It does not run CASSCF or NEVPT2.

A provisional frontier-space suggestion can be requested:

```bash
python 06_general_single_excitation_nevpt2_v3.py prepare Coordinates \
    --molecule Cyclopropene \
    --suggest-frontier 2,2
```

A suggested frontier space is only a starting point and must be inspected.

## 3. Edit `case.json`

Enter a chemically complete active space using one-based RHF MO indices:

```json
"active_mos_1based": [8, 9, 10, 11]
```

Inspect `canonical_orbitals.csv` and `canonical_orbitals.molden` first.  The
program infers the number of active electrons from the RHF occupations.

Check the ground-state symmetry and each state override:

```json
"ground_wfnsym": "A1",
"state_overrides": {
  "s-b2-pi-to-pistar": {
    "pyscf_wfnsym": "B2",
    "target_root": 0,
    "active_mos_1based": [],
    "sa_target_nroots": 1,
    "casci_nroots": 1,
    "expected_active_irrep_counts": {}
  }
}
```

`target_root` is zero-based within the selected spin/symmetry CASCI block.  The
registry root values are initial guesses, not guaranteed identities.  Root order
can change with basis, active space, state averaging, and orbital optimization.

Version 3 retains the version-2 state-specific controls:

- per-state `active_mos_1based`;
- `sa_target_nroots`;
- `casci_nroots`;
- `expected_active_irrep_counts`.

These controls are essential for cases such as the Diazomethane singlet A1
state, where a state-dependent active space and a larger state average are
needed.

## 4. Run the correlated calculation

```bash
python 06_general_single_excitation_nevpt2_v3.py run Coordinates \
    --case prepared_cases/25_Furan/case.json
```

A command-line active-space override is also possible:

```bash
python 06_general_single_excitation_nevpt2_v3.py run Coordinates \
    --case prepared_cases/01_Acetaldehyde/case.json \
    --active-mos 8,9,10
```

State selection can be overridden with:

```bash
--states valence
--states rydberg
--states all
--states s-b2-pi-to-pistar,t-b2-pi-to-pistar
```

For each target, the driver performs:

```text
RHF -> state-averaged CASSCF -> symmetry/spin-resolved CASCI
    -> strongly contracted NEVPT2 -> CI/orbital analysis
```

## 5. Outputs

The result directory contains top-level RHF files and one directory per state.
The principal summaries are:

- `vertical_energies.csv`
- `vertical_energies.json`
- `failures.json`
- `run_metadata.json`
- `<state-key>/root_scan.csv`
- `<state-key>/state_summary.json`

For each successful state, the engine reports:

- CASCI vertical energy;
- SC-NEVPT2 vertical energy;
- differences from the paper TBE and PC-NEVPT2 values;
- target-state `<S^2>` and effective multiplicity;
- active natural occupations;
- dominant ground and excited spatial configurations;
- total and dominant single-excitation configuration weights;
- dominant active-space hole and particle orbitals;
- cube files and complementary molecular-plane/offset slices.

The CASCI configuration weight is an orbital-basis-dependent diagnostic and is
not the same quantity as a coupled-cluster percent-T1 diagnostic.

## Rydberg states

Rydberg entries are retained in the registry.  A non-augmented minimal basis is
not physically suitable for the benchmark's 3s and 3p states, so the driver
skips them unless the basis name looks diffuse.  The software restriction can
be bypassed only for testing:

```bash
--allow-rydberg-without-diffuse
```

Use an augmented basis and include the relevant diffuse orbitals in the active
space for a meaningful Rydberg calculation.

## NEVPT2 contraction

PySCF's built-in `mrpt.NEVPT` is strongly contracted NEVPT2.  The principal
NEVPT2 column in the benchmark paper is partially contracted NEVPT2.  Therefore
exact numerical agreement is not expected even with a benchmark-like basis and
active space.

## Symmetry caveats

PySCF may lower a high molecular point group to a supported subgroup.  The
prepared case records the detected top group, computational group, and available
irreducible representations.  Verify every target symmetry after preparation.

For linear Delta states, one component is targeted and the perpendicular partner
should be degenerate.  For Triazine, the paper's D3h labels must be mapped to the
supported subgroup detected by PySCF.

## Recommended validation order

1. Acetaldehyde and Furan specialized scripts as reference examples.
2. Thiopropynal and Streptocyanine-C1.
3. Cyclopropene, Propynal, and Cyanoformaldehyde.
4. Methylenecyclopropene and compact linear pi systems.
5. Pyrimidine, Pyridazine, and the larger multi-root cases.
6. Degenerate linear states and Triazine subgroup mappings.
