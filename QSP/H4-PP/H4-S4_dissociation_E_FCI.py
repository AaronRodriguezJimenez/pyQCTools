"""
 Integrals for H2-H2 Chains
 This idea comes from https://arxiv.org/abs/2405.12289
 We perform the fragment separation of the H4 linear chain into two H2 fragments
"""
import os
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf,fci
from pyscf.lib import logger
from src.pyqctools.int_fcns import get_restricted_active_space_integrals
import matplotlib.pyplot as plt
import pandas as pd
from pyscf.lib.exceptions import LinearDependencyError

n = 4
d = 0.7414  # fixed H-H distance inside each H2 fragment, Angstrom


def make_S4_paldus_geometry(r_frag):
    """
    Paldus S4 model: square H4 dissociated into four H atoms.

    The four H atoms are placed at the corners of a square with side length
    r_frag. The square is centered at the origin in the xy-plane.

        H-------H
        |       |
        |       |   side length r_frag
        |       |
        H-------H
    """
    if r_frag <= 0:
        raise ValueError("r_frag must be positive.")

    half = 0.5 * r_frag

    coords = [
        (-half, -half, 0.0),
        ( half, -half, 0.0),
        ( half,  half, 0.0),
        (-half,  half, 0.0),
    ]

    return "\n".join(
        f"H {x:.10f} {y:.10f} {z:.10f}"
        for x, y, z in coords
    )

def status_label(x):
    if x is True:
        return "stable"
    if x is False:
        return "unstable"
    return "failed/unknown"


def check_stability(
    mf,
    internal=True,
    external=True,
    nroots=5,
    tol=1e-7,
):
    """
    Safely check internal and/or external SCF stability.

    Returns a dictionary containing orbitals and boolean stability flags.
    """
    try:
        mo_i, mo_e, stable_i, stable_e = mf.stability(
            internal=internal,
            external=external,
            return_status=True,
            nroots=nroots,
            tol=tol,
        )

        return {
            "mo_i": mo_i,
            "mo_e": mo_e,
            "stable_i": stable_i,
            "stable_e": stable_e,
            "error": "",
        }

    except LinearDependencyError as err:
        return {
            "mo_i": None,
            "mo_e": None,
            "stable_i": None,
            "stable_e": None,
            "error": f"LinearDependencyError: {err}",
        }

    except Exception as err:
        return {
            "mo_i": None,
            "mo_e": None,
            "stable_i": None,
            "stable_e": None,
            "error": repr(err),
        }


def stable_opt_internal(
    mf,
    stability_cycles=10,
    nroots=5,
    tol=1e-7,
):
    """
    Optimize orbitals until the solution is internally stable.

    This does NOT optimize external stability.

    RHF internal stability: rotations within RHF space.
    UHF internal stability: rotations within UHF space.

    External stability is checked separately after optimization.
    """
    log = logger.new_logger(mf)

    if not mf.converged:
        mf.kernel()

    stable_i = None
    stab_error = ""

    for cyc in range(stability_cycles + 1):
        stab = check_stability(
            mf,
            internal=True,
            external=False,
            nroots=nroots,
            tol=tol,
        )

        mo_i = stab["mo_i"]
        stable_i = stab["stable_i"]
        stab_error = stab["error"]

        print(
            f"{mf.__class__.__name__} internal stability "
            f"cycle {cyc}: {status_label(stable_i)}"
        )

        if stable_i is True:
            break

        if mo_i is None:
            log.note("Internal stability check failed; cannot update orbitals.")
            break

        if cyc == stability_cycles:
            log.note("Stability optimization failed after %d attempts", cyc)
            break

        log.note("Try to optimize orbitals until internally stable, attempt %d", cyc)

        dm1 = mf.make_rdm1(mo_i, mf.mo_occ)
        mf.kernel(dm0=dm1)

    return mf, stable_i, cyc, stab_error


def renormalize_spin_dm(dm, s, nelec):
    tr = np.einsum("ij,ji->", dm, s)
    if abs(tr) < 1e-12:
        return dm
    return dm * (nelec / tr)


def make_afm_uhf_guess(mol, rhf_mf, amp=0.03):
    """
    Small antiferromagnetic perturbation to help UHF escape RHF.

    Pattern:
        H1 alpha-rich
        H2 beta-rich
        H3 alpha-rich
        H4 beta-rich
    """
    dm_tot = rhf_mf.make_rdm1()
    dm_a = 0.5 * dm_tot.copy()
    dm_b = 0.5 * dm_tot.copy()

    aoslices = mol.aoslice_by_atom()

    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        sign = 1.0 if ia % 2 == 0 else -1.0

        dm_a[p0:p1, p0:p1] += amp * sign * np.eye(p1 - p0)
        dm_b[p0:p1, p0:p1] -= amp * sign * np.eye(p1 - p0)

    s = mol.intor_symmetric("int1e_ovlp")
    nelec_a, nelec_b = mol.nelec

    dm_a = renormalize_spin_dm(dm_a, s, nelec_a)
    dm_b = renormalize_spin_dm(dm_b, s, nelec_b)

    return dm_a, dm_b

def compute_fci_energy(
    r_frag,
    basis="sto-3g",
    stability_cycles=10,
    break_uhf_symmetry=True,
):
    atom = make_S4_paldus_geometry(r_frag)

    print("=" * 80)
    print(f"H2-H2 separation R = {r_frag} Angstrom")
    print(atom)
    print("=" * 80)

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 4
    mol.build()

    # -------------------------
    # RHF
    # -------------------------
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()

    mf, rhf_stable_i, rhf_stab_cycles, rhf_stab_error = stable_opt_internal(
        mf,
        stability_cycles=stability_cycles,
        nroots=1,
        tol=1e-7,
    )

    rhf_final_stab = check_stability(
        mf,
        internal=True,
        external=True,
        nroots=1,
        tol=1e-7,
    )

    # -------------------------
    # FCI from RHF object
    # -------------------------
    cisolver = fci.FCI(mf)
    cisolver.nroots = 1
    efci = cisolver.kernel()[0]

    # -------------------------
    # UHF
    # -------------------------
    umf = scf.UHF(mol)
    umf.conv_tol = 1e-12
    umf.max_cycle = 200

    if break_uhf_symmetry:
        dm0 = make_afm_uhf_guess(mol, mf, amp=0.03)
        umf.kernel(dm0=dm0)
    else:
        umf.kernel()

    umf, uhf_stable_i, uhf_stab_cycles, uhf_stab_error = stable_opt_internal(
        umf,
        stability_cycles=stability_cycles,
        nroots=1,
        tol=1e-7,
    )

    uhf_final_stab = check_stability(
        umf,
        internal=True,
        external=True,
        nroots=5,
        tol=1e-7,
    )

    try:
        uhf_s2, uhf_mult = umf.spin_square()
    except Exception:
        uhf_s2, uhf_mult = np.nan, np.nan

    row = {
        "R": float(r_frag),
        "basis": basis,

        "e_rhf": mf.e_tot,
        "e_uhf": umf.e_tot,
        "e_fci": efci,

        "rhf_converged": bool(mf.converged),
        "uhf_converged": bool(umf.converged),

        "rhf_internal_stable": rhf_final_stab["stable_i"],
        "rhf_external_stable": rhf_final_stab["stable_e"],
        "uhf_internal_stable": uhf_final_stab["stable_i"],
        "uhf_external_stable": uhf_final_stab["stable_e"],

        "rhf_internal_status": status_label(rhf_final_stab["stable_i"]),
        "rhf_external_status": status_label(rhf_final_stab["stable_e"]),
        "uhf_internal_status": status_label(uhf_final_stab["stable_i"]),
        "uhf_external_status": status_label(uhf_final_stab["stable_e"]),

        "rhf_stability_cycles": rhf_stab_cycles,
        "uhf_stability_cycles": uhf_stab_cycles,

        "uhf_s2": uhf_s2,
        "uhf_multiplicity": uhf_mult,

        "rhf_stability_error": rhf_final_stab["error"] or rhf_stab_error,
        "uhf_stability_error": uhf_final_stab["error"] or uhf_stab_error,
    }

    return mf, umf, efci, row

def plot_energies(results, savepath=None, show=True):
    """
    Plot resulting energies as a function of H2-H2 inter-fragment separation.

    Parameters
    ----------
    results : dict
        Dictionary keyed by r_frag, where each value contains "hf_energy".
    savepath : str or Path, optional
        If provided, saves the plot to this path.
    show : bool
        If True, displays the plot.
    """
    r_values = np.array(sorted(results.keys()), dtype=float)
    hf_energies = np.array([results[r]["ehf"] for r in r_values])
    fci_energies = np.array([results[r]["efci"] for r in r_values])
    uhf_energies = np.array([results[r]["euhf"] for r in r_values])

    plt.figure(figsize=(6, 4))
    plt.plot(r_values, hf_energies, marker="o", label="RHF")
    plt.plot(r_values, uhf_energies, marker="x", label="UHF")
    plt.plot(r_values, fci_energies, marker="o", label="FCI")
    plt.xlabel("H2-H2 separation R / Angstrom")
    plt.ylabel("RHF energy / Hartree")
    plt.title("H4 as H2-H2 Fragment Separation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"HF energy plot saved to {savepath}")

    if show:
        plt.show()

    return r_values, hf_energies

# Selected H2-H2 central separations.
# R = d gives the original equally spaced H4 chain.
r_fragment_values = np.array([
    0.5,
    0.6,
    d,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
])

outdir = Path.cwd()
results = {}
rows = []

for r_frag in r_fragment_values:
    mf, umf, efci, row = compute_fci_energy(
        r_frag=r_frag,
        basis="sto-3g",
        stability_cycles=10,
        break_uhf_symmetry=True,
    )

    results[r_frag] = {
        "ehf": row["e_rhf"],
        "efci": row["e_fci"],
        "euhf": row["e_uhf"],
    }

    rows.append(row)

stability_df = pd.DataFrame(rows)

table_cols = [
    "R",
    "e_rhf",
    "e_uhf",
    "e_fci",
    "rhf_internal_status",
    "rhf_external_status",
    "uhf_internal_status",
    "uhf_external_status",
    "uhf_s2",
    "rhf_stability_cycles",
    "uhf_stability_cycles",
]

print("\nSCF stability summary")
print(stability_df[table_cols].to_string(index=False))

stability_df.to_csv(outdir / "h4_S4_stability_table.csv", index=False)
print(f"\nStability table saved to {outdir / 'h4_S4_stability_table.csv'}")

r_values, hf_energies = plot_energies(
    results,
    savepath=outdir / "energy_vs_H4_S4_separation.png",
)
