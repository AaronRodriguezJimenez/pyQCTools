"""
Integrals for H2-H2 / H4 Chains using CCSD Natural orbitals

This idea comes from https://arxiv.org/abs/2405.12289
here we include some Paldus models: Phys. Rev. A 47, 4 (1993)

We support three H4 geometries:

- LIN4:
    Linear chain H-H-H-H dissociated into 4 H atoms

        H -- R -- H -- R -- H -- R -- H

- D4:
    Paldus model: two H2 fragments at dissociation

        H -- d -- H -- R -- H -- d -- H

- S4:
    Paldus model: square H4 dissociated into 4 H atoms

        H-------H
        |       |
        |       |   side length R
        |       |
        H-------H
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, mcscf, cc
from pyscf.mcscf import addons as mcscf_addons
from pyscf.lib.exceptions import LinearDependencyError
from pyscf.lib import logger

from src.pyqctools.int_fcns import get_restricted_NO_integrals


n = 4
d = 0.7414  # fixed H-H distance inside each H2 fragment, Angstrom


def format_float_tag(x):
    """Make a float safe for filenames."""
    return f"{x:.6g}".replace("-", "m").replace(".", "p")


def make_H4_lin_diss_geometry(r_frag):
    """
    Linear H4 dissociated into four H atoms:

        H -- r_frag -- H -- r_frag -- H -- r_frag -- H

    Here r_frag is the nearest-neighbor H-H separation.
    """
    if r_frag <= 0:
        raise ValueError("r_frag must be positive.")

    z = np.array([
        0.0,
        r_frag,
        2.0 * r_frag,
        3.0 * r_frag,
    ])

    return "\n".join(f"H 0.0 0.0 {zi:.10f}" for zi in z)


def make_D4_paldus_geometry(d_hh, r_frag):
    """
    Paldus D4 model: linear H4 split into two H2 fragments:

        H -- d_hh -- H -- r_frag -- H -- d_hh -- H

    Here r_frag is the central H-H separation between the two H2 fragments.
    """
    if d_hh <= 0:
        raise ValueError("d_hh must be positive.")
    if r_frag <= 0:
        raise ValueError("r_frag must be positive.")

    z = np.array([
        0.0,
        d_hh,
        d_hh + r_frag,
        2.0 * d_hh + r_frag,
    ])

    return "\n".join(f"H 0.0 0.0 {zi:.10f}" for zi in z)


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


def make_H4_geometry(model, r_frag, d_hh=0.7414):
    """
    Dispatch function for H4 geometry construction.

    Parameters
    ----------
    model : str
        One of {"LIN4", "D4", "S4"}.
    r_frag : float
        Dissociation/separation parameter in Angstrom.
    d_hh : float
        Fixed H-H bond length for the D4 H2-H2 geometry.
    """
    model = model.upper()

    if model in {"LIN4", "LINEAR", "LINEAR4", "CHAIN"}:
        return make_H4_lin_diss_geometry(r_frag)

    if model == "D4":
        return make_D4_paldus_geometry(d_hh, r_frag)

    if model == "S4":
        return make_S4_paldus_geometry(r_frag)

    raise ValueError(
        f"Unknown model '{model}'. Expected one of: LIN4, D4, S4."
    )

def safe_rhf_stability(
    mf,
    internal=True,
    external=False,
    return_status=True,
    nroots=1,
    tol=1e-7,
):
    """
    Safe wrapper around mf.stability.

    External RHF stability can fail for near-degenerate stretched H systems
    because the Davidson initial space may become linearly dependent.
    """
    try:
        return mf.stability(
            internal=internal,
            external=external,
            return_status=return_status,
            nroots=nroots,
            tol=tol,
        )

    except LinearDependencyError as err:
        print("WARNING: Stability analysis failed with LinearDependencyError.")
        print(f"Reason: {err}")

        if return_status:
            mo_i = mf.mo_coeff if internal else None
            mo_e = None
            stable_i = None
            stable_e = None
            return mo_i, mo_e, stable_i, stable_e

        return None, None


def run_stable_rhf(
    mol,
    dm0=None,
    conv_tol=1e-10,
    conv_tol_grad=1e-7,
    max_cycle=300,
    stability_cycles=20,
    stability_tol=1e-7,
    check_external=True,
    use_newton_if_needed=False,
    level_shift=0.2,
    damping=0.2,
    raise_on_unstable=False,
):
    """
    Run RHF and optimize until internally stable.

    External stability is treated as a diagnostic only. For stretched H chains,
    RHF is often externally unstable because a spin-broken UHF solution exists.
    """

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_cycle
    mf.diis_space = 12

    # These help avoid oscillations near degeneracies.
    mf.level_shift = level_shift
    mf.damp = damping

    print("Running initial RHF...")
    mf.kernel(dm0=dm0)

    # If it converged with level shift/damping, remove them and polish.
    if mf.converged:
        print("Polishing RHF without level shift/damping...")
        dm_polish = mf.make_rdm1()
        mf.level_shift = 0.0
        mf.damp = 0.0
        mf.kernel(dm0=dm_polish)

    if not mf.converged and use_newton_if_needed:
        print("RHF did not converge. Retrying with Newton solver...")
        try:
            dm_retry = mf.make_rdm1()
            mf_newton = mf.newton()
            mf_newton.conv_tol = conv_tol
            mf_newton.conv_tol_grad = conv_tol_grad
            mf_newton.max_cycle = max_cycle
            mf_newton.kernel(dm0=dm_retry)
            mf = mf_newton
        except Exception as err:
            print("WARNING: Newton solver failed.")
            print(f"Reason: {err}")

    if not mf.converged:
        msg = "RHF did not converge."
        if raise_on_unstable:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    info = {
        "converged": bool(mf.converged),
        "internal_stable": None,
        "external_stable": None,
        "stability_cycles": 0,
    }

    # Internal stability optimization only.
    for cyc in range(stability_cycles):
        print(f"RHF internal stability check cycle {cyc}")

        mo_i, _, stable_i, _ = safe_rhf_stability(
            mf,
            internal=True,
            external=False,
            return_status=True,
            nroots=1,
            tol=stability_tol,
        )

        info["internal_stable"] = stable_i
        info["stability_cycles"] = cyc

        print("  Internal stable?", stable_i)

        if stable_i is True:
            break

        if stable_i is None:
            print("  Internal stability check failed. Keeping current RHF solution.")
            break

        print("  Re-optimizing RHF from internally improved orbitals...")
        dm1 = mf.make_rdm1(mo_i, mf.mo_occ)

        mf.level_shift = level_shift
        mf.damp = damping
        mf.kernel(dm0=dm1)

        if mf.converged:
            dm_polish = mf.make_rdm1()
            mf.level_shift = 0.0
            mf.damp = 0.0
            mf.kernel(dm0=dm_polish)

    # External stability as diagnostic only.
    if check_external:
        print("Running external RHF stability diagnostic...")

        _, mo_e, _, stable_e = safe_rhf_stability(
            mf,
            internal=False,
            external=True,
            return_status=True,
            nroots=1,
            tol=stability_tol,
        )

        info["external_stable"] = stable_e

        print("  External stable?", stable_e)

        if stable_e is False:
            print(
                "WARNING: RHF is externally unstable. "
                "For stretched H systems this usually means a lower-energy "
                "spin-broken UHF solution exists."
            )

        if stable_e is None:
            print(
                "WARNING: External stability diagnostic failed. "
                "This does not necessarily mean the RHF calculation failed."
            )

    if info["internal_stable"] is not True:
        msg = "RHF solution was not confirmed internally stable."

        if raise_on_unstable:
            raise RuntimeError(msg)

        print("WARNING:", msg)

    return mf, info

def run_ccsd_natural_orbitals_pyscf(
    mf,
    cc_conv_tol=1e-10,
    cc_conv_tol_normt=1e-8,
    cc_max_cycle=200,
    cc_diis_space=12,
    cc_level_shift=0.0,
    require_ccsd_converged=False,
):
    """
    Run RHF-CCSD and generate CCSD natural orbitals using PySCF's
    mcscf.addons.make_natural_orbitals helper.

    Returns
    -------
    mycc : pyscf.cc.CCSD
        CCSD object.

    noons : ndarray
        Natural orbital occupation numbers.

    natorbs : ndarray
        Natural orbital coefficients in the AO basis.

    cc_info : dict
        CCSD diagnostics.
    """
    print("\nRunning CCSD for PySCF natural orbitals...")

    mycc = cc.CCSD(mf)
    mycc.conv_tol = cc_conv_tol
    mycc.conv_tol_normt = cc_conv_tol_normt
    mycc.max_cycle = cc_max_cycle
    mycc.diis_space = cc_diis_space
    mycc.level_shift = cc_level_shift

    cc_info = {
        "ccsd_converged": False,
        "ccsd_energy": None,
        "ccsd_corr_energy": None,
        "noons": None,
    }

    try:
        mycc.kernel()

        cc_info["ccsd_converged"] = bool(mycc.converged)
        cc_info["ccsd_energy"] = mycc.e_tot
        cc_info["ccsd_corr_energy"] = mycc.e_corr

        print(f"CCSD converged? {mycc.converged}")
        print(f"CCSD total energy:       {mycc.e_tot: .12f}")
        print(f"CCSD correlation energy: {mycc.e_corr: .12f}")

    except Exception as err:
        print("WARNING: CCSD failed.")
        print(f"Reason: {err}")

        if require_ccsd_converged:
            raise

        print("Falling back to RHF orbitals.")
        noons = mf.mo_occ.copy()
        natorbs = mf.mo_coeff.copy()
        cc_info["noons"] = noons.copy()
        return mycc, noons, natorbs, cc_info

    if not mycc.converged:
        msg = "CCSD did not fully converge."

        if require_ccsd_converged:
            raise RuntimeError(msg)

        print("WARNING:", msg)
        print("Using best available CCSD amplitudes for natural orbitals.")

    try:
        # PySCF-supported construction of natural occupations and orbitals.
        #noons, natorbs = mcscf_addons.make_natural_orbitals(mycc)
        # Manual diagonalization
        # Calculate Natural Orbitals
        print("Solving Lambda equations for 1-RDM...")
        mycc.solve_lambda()
        rdm1_mo = mycc.make_rdm1()

        # Diagonalize 1-RDM
        noons, natorbs = np.linalg.eigh(rdm1_mo)

        # Sort descending
        idx = np.argsort(noons)[::-1]
        noons = noons[idx]
        natorbs = natorbs[:, idx]

    except Exception as err:
        print("WARNING: PySCF make_natural_orbitals failed.")
        print(f"Reason: {err}")

        if require_ccsd_converged:
            raise

        print("Falling back to RHF orbitals.")
        noons = mf.mo_occ.copy()
        natorbs = mf.mo_coeff.copy()

    noons = np.asarray(noons)
    natorbs = np.asarray(natorbs)

    cc_info["noons"] = noons.copy()

    print("PySCF CCSD natural occupations:")
    for i, occ in enumerate(noons):
        print(f"  NO {i:3d}: {occ: .10f}")

    # Sanity check: natorbs should be AO-basis MO coefficients.
    s = mf.mol.intor_symmetric("int1e_ovlp")
    orth_err = np.linalg.norm(natorbs.T @ s @ natorbs - np.eye(natorbs.shape[1]))
    print(f"Natural orbital orthonormality error: {orth_err:.3e}")

    return mycc, noons, natorbs, cc_info


def save_tensors(label, h0, h1, h2, folder="tensors"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = folder / f"{label}_tensors.npz"
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Tensors saved to {filename}")

def compute_integrals_for_separation(
    r_frag,
    model="D4",
    d_hh=0.7414,
    basis="sto-3g",
    localized=False,
    outdir=".",
    stability_cycles=20,
    prev_mol=None,
    prev_dm=None,
    check_external=True,
    use_ccsd_natural_orbitals=True,
    require_ccsd_converged=False,
):
    model = model.upper()

    outdir = Path(outdir)
    tensor_dir = outdir / "NO-CCSD_tensors"
    log_dir = outdir / "NO-CCSD_logs"

    tensor_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    atom = make_H4_geometry(model=model, r_frag=r_frag, d_hh=d_hh)
    r_tag = format_float_tag(r_frag)

    print("=" * 80)
    print(f"Model = {model}")
    print(f"H4 separation parameter R = {r_frag} Angstrom")
    print(atom)
    print("=" * 80)

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.verbose = 4
    mol.output = str(log_dir / f"H4_{model}_R_{r_tag}.out")
    mol.build()

    dm0 = None
    if prev_mol is not None and prev_dm is not None:
        print("Projecting previous density matrix to current geometry...")
        dm0 = scf.addons.project_dm_nr2nr(prev_mol, prev_dm, mol)

    mf, mf_info = run_stable_rhf(
        mol,
        dm0=dm0,
        stability_cycles=stability_cycles,
        check_external=check_external,
        conv_tol=1e-10,
        conv_tol_grad=1e-7,
        stability_tol=1e-7,
        use_newton_if_needed=False,
        level_shift=0.2,
        damping=0.2,
        raise_on_unstable=False,
    )

    mycc = None
    noons = None
    natorbs = None
    cc_info = {
        "ccsd_converged": None,
        "ccsd_energy": None,
        "ccsd_corr_energy": None,
        "noons": None,
    }

    if use_ccsd_natural_orbitals:
        mycc, noons, natorbs, cc_info = run_ccsd_natural_orbitals_pyscf(
            mf,
            cc_conv_tol=1e-10,
            cc_conv_tol_normt=1e-8,
            cc_max_cycle=200,
            cc_diis_space=12,
            cc_level_shift=0.0,
            require_ccsd_converged=require_ccsd_converged,
        )

        orbital_label = "CCSDNO"

    else:
        noons = mf.mo_occ.copy()
        natorbs = mf.mo_coeff.copy()
        orbital_label = "RHF"

    # Full active space for H4/STO-3G: mc object to place information
    # 4 spatial orbitals, 4 electrons.
    ncore = 0
    ncas = mol.nao_nr() - ncore
    nelecas = mol.nelectron - 2 * ncore

    mc = mcscf.CASCI(mf, ncas, nelecas)

    print("NCORE IS:", mc.ncore)
    print(f"Integral orbital basis: {orbital_label}")

    s = mol.intor_symmetric("int1e_ovlp")
    orth_err = np.linalg.norm(
        mc.mo_coeff.T @ s @ mc.mo_coeff - np.eye(mc.mo_coeff.shape[1])
    )
    print(f"Integral-basis orthonormality error: {orth_err:.3e}")

    if localized and use_ccsd_natural_orbitals:
        print(
            "WARNING: localized=True will rotate the CCSD natural orbitals. "
            "Set localized=False for strict CCSD-NO integrals."
        )

    h0, h1, h2 = get_restricted_NO_integrals(
        mol,
        mf,
        mc,
        natorbs,
        localized=localized,
        include_all_noncore=False,
    )

    label = f"h{n}_{orbital_label}_{model}_R_{r_tag}"

    save_tensors(
        label,
        h0,
        h1,
        h2,
        folder=tensor_dir,
    )

    final_dm = mf.make_rdm1()

    return h0, h1, h2, mol, mf, mc, final_dm, mf_info, mycc, noons, natorbs, cc_info


def plot_hf_energies(results, savepath=None, show=True):
    """
    Plot RHF and CCSD energies as a function of the H4 separation parameter.

    Parameters
    ----------
    results : dict
        Nested dictionary of the form:

            results[model][r_frag]["hf_energy"]
            results[model][r_frag]["ccsd_energy"]

        The "ccsd_energy" entry is optional. If missing or None, that point
        is skipped in the CCSD curve.

    savepath : str or Path, optional
        If provided, saves the plot to this path.

    show : bool
        If True, displays the plot.
    """
    plt.figure(figsize=(7.5, 5.0))

    plot_data = {}

    for model, model_results in results.items():
        r_values = np.array(sorted(model_results.keys()), dtype=float)

        hf_energies = np.array(
            [model_results[r].get("hf_energy", np.nan) for r in r_values],
            dtype=float,
        )

        ccsd_energies = np.array(
            [
                np.nan
                if model_results[r].get("ccsd_energy", None) is None
                else model_results[r]["ccsd_energy"]
                for r in r_values
            ],
            dtype=float,
        )

        # RHF curve
        plt.plot(
            r_values,
            hf_energies,
            marker="o",
            linestyle="-",
            label=f"{model} RHF",
        )

        # CCSD curve, only if at least one finite CCSD energy exists
        if np.any(np.isfinite(ccsd_energies)):
            plt.plot(
                r_values,
                ccsd_energies,
                marker="s",
                linestyle="--",
                label=f"{model} CCSD",
            )

        plot_data[model] = {
            "r_values": r_values,
            "hf_energies": hf_energies,
            "ccsd_energies": ccsd_energies,
        }

    plt.xlabel("H4 separation parameter R / Angstrom")
    plt.ylabel("Energy / Hartree")
    plt.title("H4 RHF and CCSD Dissociation Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"Energy plot saved to {savepath}")

    if show:
        plt.show()

    return plot_data

# Separation values.
#
# For LIN4:
#     R is the nearest-neighbor H-H distance.
#
# For D4:
#     R is the central H-H distance between the two H2 fragments.
#     R = d gives the original equally spaced H4 chain.
#
# For S4:
#     R is the square side length.
r_fragment_values = np.array([
    d,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
])


models_to_run = [
    "D4",
    "LIN4",
    "S4",
]

outdir = Path.cwd()
results = {model: {} for model in models_to_run}

for model in models_to_run:
    prev_mol = None
    prev_dm = None

    for r_frag in r_fragment_values:
        (
            h0,
            h1,
            h2,
            mol,
            mf,
            mc,
            final_dm,
            mf_info,
            mycc,
            noons,
            natorbs,
            cc_info,
        ) = compute_integrals_for_separation(
            r_frag=r_frag,
            model=model,
            d_hh=d,
            basis="sto-3g",
            localized=False,  # important: preserve strict CCSD NO basis
            outdir=outdir,
            stability_cycles=20,
            prev_mol=prev_mol,
            prev_dm=prev_dm,
            check_external=True,
            use_ccsd_natural_orbitals=True,
            require_ccsd_converged=False,
        )

        results[model][r_frag] = {
            "h0": h0,
            "h1": h1,
            "h2": h2,
            "hf_energy": mf.e_tot,
            "ccsd_energy": cc_info["ccsd_energy"],
            "ccsd_corr_energy": cc_info["ccsd_corr_energy"],
            "ccsd_converged": cc_info["ccsd_converged"],
            "natural_occupations": noons,
            "mf_converged": mf_info["converged"],
            "internal_stable": mf_info["internal_stable"],
            "external_stable": mf_info["external_stable"],
            "stability_cycles": mf_info["stability_cycles"],
        }

        prev_mol = mol
        prev_dm = final_dm

print("\nSCF stability summary")
print("=" * 80)

print("\nSCF / CCSD / natural-orbital summary")
print("=" * 110)

for model, model_results in results.items():
    print(f"\nModel: {model}")

    for r_frag in sorted(model_results.keys()):
        row = model_results[r_frag]

        print(
            f"R = {r_frag:8.4f}  "
            f"E_RHF = {row['hf_energy']: .12f}  "
            f"E_CCSD = {row['ccsd_energy']: .12f}  "
            f"SCF conv = {row['mf_converged']}  "
            f"CCSD conv = {row['ccsd_converged']}  "
            f"int stable = {row['internal_stable']}  "
            f"ext stable = {row['external_stable']}"
        )

        print("  Natural occupations:", row["natural_occupations"])

plot_data = plot_hf_energies(
    results,
    savepath=outdir / "energy_vs_H4_separation_RHF_CCSD_all_models.png",
)