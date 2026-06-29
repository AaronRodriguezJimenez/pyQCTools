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
from scipy.linalg import eigh
from pyscf import gto, scf, fci, cc
from pyscf import lo, ao2mo
from pyscf.lib import logger

from src.pyqctools.int_fcns import get_restricted_NO_integrals


n = 4
d = 0.7414  # fixed H-H distance inside each H2 fragment, Angstrom

BASIS = "sto-3g"

SCF_CONV_TOL = 1e-12
FCI_CONV_TOL = 1e-12

CC_CONV_TOL = 1e-8
CC_CONV_TOL_NORMT = 1e-6
CC_MAX_CYCLE = 999
CC_DIIS_SPACE = 12

STABILITY_TOL = 1e-8
STABILITY_CYCLES = 20

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

# ===============================================
# RUNNERS
# ===============================================
def make_broken_symmetry_guess(mf, eps=0.05):
    dm = mf.get_init_guess(key="atom")
    dm = np.asarray(dm)

    if dm.ndim == 2:
        dm = np.array([0.5 * dm, 0.5 * dm])

    dm[0, 0, 0] += eps
    dm[0, 1, 1] -= eps
    dm[1, 0, 0] -= eps
    dm[1, 1, 1] += eps

    return dm

def stable_opt_internal(mf, stability_cycles=STABILITY_CYCLES):
    log = logger.new_logger(mf)

    mo1, _, stable, _ = mf.stability(
        internal=True,
        external=False,
        return_status=True,
        tol=STABILITY_TOL,
    )

    cyc = 0
    print("STABLE?", stable)

    while (not stable) and cyc < stability_cycles:
        log.note("Try to optimize orbitals until stable, attempt %d" % cyc)

        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)

        mo1, _, stable, _ = mf.stability(
            internal=True,
            external=False,
            return_status=True,
            tol=STABILITY_TOL,
        )

        print("STABLE?", stable)
        cyc += 1

    if not stable:
        log.note("Stability Opt failed after %d attempts" % cyc)

    return mf

def run_uhf(mol, stability_cycles=10):
    mf = scf.UHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = 200
    mf.verbose = 0

    dm0 = make_broken_symmetry_guess(mf)
    mf.kernel(dm0)

    if not mf.converged:
        print("WARNING: Initial UHF did not converge. Retrying with max_cycle=500.")
        mf.max_cycle = 500
        mf.kernel(dm0)

    mf_stable = stable_opt_internal(mf, stability_cycles=stability_cycles)

    if not mf_stable.converged:
        print("WARNING: UHF is not converged after stability optimization.")

    s2, mult = mf_stable.spin_square()
    print(f"UHF spin square <S^2>: {s2}")
    print(f"UHF multiplicity estimate: {mult}")

    return mf_stable

def compute_fci_energy(mol):
    """
    Compute FCI energy using RHF orbitals as the one-particle basis.
    """

    rhf = scf.RHF(mol)
    rhf.conv_tol = SCF_CONV_TOL
    rhf.max_cycle = 300
    rhf.verbose = 0
    rhf.kernel()

    if not rhf.converged:
        print("WARNING: RHF for FCI orbital basis did not converge. Retrying.")
        rhf.max_cycle = 500
        rhf.kernel()

    mo = rhf.mo_coeff
    nmo = mo.shape[1]
    nelec = mol.nelec

    h1_mo = mo.T @ rhf.get_hcore() @ mo
    eri_mo = ao2mo.kernel(mol, mo)

    cisolver = fci.direct_spin1.FCI(mol)
    cisolver.conv_tol = FCI_CONV_TOL

    efci, ci_vec = cisolver.kernel(
        h1_mo,
        eri_mo,
        nmo,
        nelec,
        ecore=mol.energy_nuc(),
    )

    return efci, ci_vec, rhf

# =============================================================================
# Natural orbital helpers
# =============================================================================
def get_uccsd_natural_orbitals(mycc, mf, debug=True):
    """
    Build spin-dependent UCCSD natural orbitals.

    mycc.make_rdm1() gives alpha/beta 1-RDMs in the UHF MO basis.
    We diagonalize each spin block and rotate the UHF MO coefficients.
    """

    if not mycc.converged:
        print("WARNING: UCCSD did not converge. UCCSD natural orbitals may be unreliable.")

    mycc.solve_lambda()

    rdm1 = mycc.make_rdm1()

    no_coeffs = []
    no_occs = []

    for spin_idx, (dm_mo, mo_c_full) in enumerate(zip(rdm1, mf.mo_coeff)):
        spin_label = "Alpha" if spin_idx == 0 else "Beta"

        dm_mo = 0.5 * (dm_mo + dm_mo.conj().T)

        occ_vals, vecs = np.linalg.eigh(dm_mo)

        idx = occ_vals.argsort()[::-1]
        occ_vals = occ_vals[idx]
        vecs = vecs[:, idx]

        no_c = mo_c_full @ vecs

        no_coeffs.append(no_c)
        no_occs.append(occ_vals)

        if debug:
            print(f"{spin_label} UCCSD natural occupancies:", occ_vals)

    no_coeffs = np.array(no_coeffs)
    no_occs = np.array(no_occs)

    return no_coeffs, no_occs

# =============================================================================
# Tensor saving
# =============================================================================

def save_tensors(b, h0, h1a, h1b, eri_aa, eri_bb, eri_ab, folder="tensors"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = folder / f"{b}_tensors.npz"

    np.savez_compressed(
        filename,
        hc=h0,
        h1_a=h1a,
        h1_b=h1b,
        h2aa=eri_aa,
        h2bb=eri_bb,
        h2ab=eri_ab,
    )

    print(f"Tensors saved to {filename}")

# =============================================================================
# Plotting
# =============================================================================
def plot_energies(results, savepath=None, show=True):
    """
    Plot energies as a function of the H4 separation parameter.
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
                if model_results[r].get("uccsd", None) is None
                else model_results[r]["uccsd"]
                for r in r_values
            ],
            dtype=float,
        )

        fci_energies = np.array(
            [
                np.nan
                if model_results[r].get("efci", None) is None
                else model_results[r]["efci"]
                for r in r_values
            ],
            dtype=float,
        )

        # UHF curve
        plt.plot(
            r_values,
            hf_energies,
            marker="x",
            linestyle="-",
            label=f"{model} UHF",
        )

        # CCSD curve, only if at least one finite CCSD energy exists
        if np.any(np.isfinite(ccsd_energies)):
            plt.plot(
                r_values,
                ccsd_energies,
                marker="s",
                linestyle="--",
                label=f"{model} UCCSD",
            )

            # FCI curve
            if np.any(np.isfinite(fci_energies)):
                plt.plot(
                    r_values,
                    fci_energies,
                    marker="o",
                    linestyle="--",
                    label=f"{model} FCI",
                )

        plot_data[model] = {
            "r_values": r_values,
            "hf_energies": hf_energies,
            "ccsd_energies": ccsd_energies,
            "fci_energies": fci_energies,
        }

    plt.xlabel("H4 separation parameter R / Angstrom")
    plt.ylabel("Energy / Hartree")
    plt.title("H4 Dissociation Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"Energy plot saved to {savepath}")

    if show:
        plt.show()

    return plot_data

def plot_uccsd_natural_occupations(results, savepath=None, show=True):
    """
    Plot selected UCCSD natural orbital occupation numbers.

    results[model][r_frag] should contain:
        "uccsd_no_occ_alpha"
        "uccsd_no_occ_beta"
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    plot_data = {}

    for model, model_results in results.items():
        # Sort by numeric R value, but keep original dictionary keys
        sorted_items = sorted(model_results.items(), key=lambda item: float(item[0]))

        r_values = np.array([float(r) for r, _ in sorted_items])

        occ_a_0 = np.array([diag["uccsd_no_occ_alpha"][0] for _, diag in sorted_items])
        occ_a_1 = np.array([diag["uccsd_no_occ_alpha"][1] for _, diag in sorted_items])
        occ_a_2 = np.array([diag["uccsd_no_occ_alpha"][2] for _, diag in sorted_items])
        occ_a_3 = np.array([diag["uccsd_no_occ_alpha"][3] for _, diag in sorted_items])
        occ_b_0 = np.array([diag["uccsd_no_occ_beta"][0] for _, diag in sorted_items])
        occ_b_1 = np.array([diag["uccsd_no_occ_beta"][1] for _, diag in sorted_items])
        occ_b_2 = np.array([diag["uccsd_no_occ_beta"][2] for _, diag in sorted_items])
        occ_b_3 = np.array([diag["uccsd_no_occ_beta"][3] for _, diag in sorted_items])

        ax.plot(r_values, occ_a_0, marker="o", linestyle="-",  label=f"{model} alpha NO 1")
        ax.plot(r_values, occ_a_1, marker="o", linestyle="--", label=f"{model} alpha NO 2")
        ax.plot(r_values, occ_a_2, marker="o", linestyle="--", label=f"{model} alpha NO 3")
        ax.plot(r_values, occ_a_3, marker="o", linestyle="--", label=f"{model} alpha NO 4")
        ax.plot(r_values, occ_b_0, marker="s", linestyle="-",  label=f"{model} beta NO 1")
        ax.plot(r_values, occ_b_1, marker="s", linestyle="--", label=f"{model} beta NO 2")
        ax.plot(r_values, occ_b_2, marker="s", linestyle="--", label=f"{model} beta NO 3")
        ax.plot(r_values, occ_b_3, marker="s", linestyle="--", label=f"{model} beta NO 4")

        plot_data[model] = {
            "r_values": r_values,
            "occ_a_0": occ_a_0,
            "occ_a_1": occ_a_1,
            "occ_a_2": occ_a_2,
            "occ_a_3": occ_a_3,
            "occ_b_0": occ_b_0,
            "occ_b_1": occ_b_1,
            "occ_b_2": occ_b_2,
            "occ_b_3": occ_b_3,
        }

    ax.set_xlabel("H4 separation parameter R / Angstrom")
    ax.set_ylabel("UCCSD natural orbital occupations")
    ax.set_title("H4 Models UCCSD NOONs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"NOONs plot saved to {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_data

def plot_uccsd_natural_occupations_by_model(results, savepath=None, show=True):
    """
    Create one figure with separate subplots for each H4 model.

    results[model][r_frag] should contain:
        "uccsd_no_occ_alpha"
        "uccsd_no_occ_beta"
    """
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(7.5, 4.0 * n_models),
        sharex=True,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    plot_data = {}

    for ax, model in zip(axes, models):
        model_results = results[model]

        sorted_items = sorted(model_results.items(), key=lambda item: float(item[0]))
        r_values = np.array([float(r) for r, _ in sorted_items])

        occ_a_0 = np.array([diag["uccsd_no_occ_alpha"][0] for _, diag in sorted_items])
        occ_a_1 = np.array([diag["uccsd_no_occ_alpha"][1] for _, diag in sorted_items])
        occ_a_2 = np.array([diag["uccsd_no_occ_alpha"][2] for _, diag in sorted_items])
        occ_a_3 = np.array([diag["uccsd_no_occ_alpha"][3] for _, diag in sorted_items])

        occ_b_0 = np.array([diag["uccsd_no_occ_beta"][0] for _, diag in sorted_items])
        occ_b_1 = np.array([diag["uccsd_no_occ_beta"][1] for _, diag in sorted_items])
        occ_b_2 = np.array([diag["uccsd_no_occ_beta"][2] for _, diag in sorted_items])
        occ_b_3 = np.array([diag["uccsd_no_occ_beta"][3] for _, diag in sorted_items])

        ax.plot(r_values, occ_a_0, marker="o", linestyle="-",  label="alpha NO 1")
        ax.plot(r_values, occ_a_1, marker="o", linestyle="--", label="alpha NO 2")
        ax.plot(r_values, occ_a_2, marker="o", linestyle="--", label="alpha NO 3")
        ax.plot(r_values, occ_a_3, marker="o", linestyle="--", label="alpha NO 4")

        ax.plot(r_values, occ_b_0, marker="s", linestyle="-",  label="beta NO 1")
        ax.plot(r_values, occ_b_1, marker="s", linestyle="--", label="beta NO 2")
        ax.plot(r_values, occ_b_2, marker="s", linestyle="--", label="beta NO 3")
        ax.plot(r_values, occ_b_3, marker="s", linestyle="--", label="beta NO 4")

        ax.set_title(f"{model} UCCSD NOONs")
        ax.set_ylabel("NOONs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plot_data[model] = {
            "r_values": r_values,
            "occ_a_0": occ_a_0,
            "occ_a_1": occ_a_1,
            "occ_a_2": occ_a_2,
            "occ_a_3": occ_a_3,
            "occ_b_0": occ_b_0,
            "occ_b_1": occ_b_1,
            "occ_b_2": occ_b_2,
            "occ_b_3": occ_b_3,
        }

    axes[-1].set_xlabel("H4 separation parameter R / Angstrom")

    fig.suptitle("H4 Models UCCSD Natural Orbital Occupations", y=1.02)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"NOONs subplot figure saved to {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_data

# =============================================================================
# Integral transformation
# =============================================================================

def get_active_space_tensors(
    mol,
    nat_ccsd_coeff,
    mf,
    localized=False,
    ccsd=False,
    natural=False,
    debug=True,
):
    """
    Extract spin-dependent one- and two-electron tensors.

    Cases:
      ccsd=False, natural=False  -> UHF orbitals
      ccsd=False, natural=True   -> Natural UHF orbitals
      ccsd=True,  natural=True   -> Provided Natural UCCSD orbitals
      ccsd=True,  natural=False  -> Provided external orbitals

    Returns:
      h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab
    """

    ao_kin = mol.intor("int1e_kin", hermi=1)
    ao_nuc = mol.intor("int1e_nuc", hermi=1)
    ao_obi = ao_kin + ao_nuc

    n_ao = ao_obi.shape[0]

    if ccsd == False:
        mo_coeff = getattr(mf, "mo_coeff", None)

        if mo_coeff is None:
            raise ValueError("mf.mo_coeff is None")

        Ca, Cb = mo_coeff

    else:
        if nat_ccsd_coeff is None:
            raise ValueError("ccsd=True requires nat_ccsd_coeff=(Ca, Cb).")

        Ca, Cb = nat_ccsd_coeff

    nmo_a = Ca.shape[1]
    nmo_b = Cb.shape[1]

    a_nocc = int((mf.mo_occ[0] > 0).sum())
    b_nocc = int((mf.mo_occ[1] > 0).sum())

    if debug:
        print("DEBUG shapes:")
        print("  ao_obi.shape:", ao_obi.shape)
        print("  Ca.shape:", Ca.shape)
        print("  Cb.shape:", Cb.shape)
        print("  a_nocc:", a_nocc)
        print("  b_nocc:", b_nocc)
        print("  ccsd:", ccsd)
        print("  natural:", natural)
        print("  localized:", localized)

    if localized:
        a_occ_idx = np.arange(0, a_nocc)
        a_vir_idx = np.arange(a_nocc, nmo_a)
        b_occ_idx = np.arange(0, b_nocc)
        b_vir_idx = np.arange(b_nocc, nmo_b)

        Ca_occ = Ca[:, a_occ_idx] if a_occ_idx.size > 0 else Ca[:, :0]
        Ca_vir = Ca[:, a_vir_idx] if a_vir_idx.size > 0 else Ca[:, :0]
        Cb_occ = Cb[:, b_occ_idx] if b_occ_idx.size > 0 else Cb[:, :0]
        Cb_vir = Cb[:, b_vir_idx] if b_vir_idx.size > 0 else Cb[:, :0]

        a_loc_occ = lo.Boys(mol, Ca_occ).kernel() if Ca_occ.shape[1] >= 2 else Ca_occ.copy()
        a_loc_vir = lo.Boys(mol, Ca_vir).kernel() if Ca_vir.shape[1] >= 2 else Ca_vir.copy()
        b_loc_occ = lo.Boys(mol, Cb_occ).kernel() if Cb_occ.shape[1] >= 2 else Cb_occ.copy()
        b_loc_vir = lo.Boys(mol, Cb_vir).kernel() if Cb_vir.shape[1] >= 2 else Cb_vir.copy()

        Ca_final = np.column_stack((a_loc_occ, a_loc_vir))
        Cb_final = np.column_stack((b_loc_occ, b_loc_vir))

    elif natural and ccsd == False:
        # Natural UHF orbitals from AO density matrices.
        #
        # PySCF AO density D satisfies:
        #
        #     N = Tr[D S]
        #
        # Natural orbitals C are obtained from:
        #
        #     S D S C = S C n
        #
        # where C.T @ S @ C = I.

        dma, dmb = mf.make_rdm1()
        S = mf.get_ovlp()

        A_a = S @ dma @ S
        A_b = S @ dmb @ S

        A_a = 0.5 * (A_a + A_a.conj().T)
        A_b = 0.5 * (A_b + A_b.conj().T)

        occ_a, NOs_a = eigh(A_a, S)
        occ_b, NOs_b = eigh(A_b, S)

        #Sort descending
        idx_a = occ_a.argsort()[::-1]
        idx_b = occ_b.argsort()[::-1]

        occ_a = occ_a[idx_a]
        occ_b = occ_b[idx_b]

        Ca_final = NOs_a[:, idx_a]
        Cb_final = NOs_b[:, idx_b]

        if debug:
            print("  Natural UHF occ alpha:", occ_a)
            print("  Natural UHF occ beta :", occ_b)
            print("  Tr[D_alpha S]:", np.trace(dma @ S))
            print("  Tr[D_beta  S]:", np.trace(dmb @ S))

    else:
        # Includes:
        #
        #   UHF:
        #       ccsd=False, natural=False
        #
        #   Natural UCCSD:
        #       ccsd=True, natural=True
        #
        # In the Natural UCCSD case, Ca/Cb are already the natural orbitals.
        Ca_final = Ca.copy()
        Cb_final = Cb.copy()

    if Ca_final.shape[0] != n_ao or Cb_final.shape[0] != n_ao:
        raise ValueError(
            f"Final MO coeff matrices must have {n_ao} AO rows. "
            f"Got {Ca_final.shape}, {Cb_final.shape}"
        )

    S = mf.get_ovlp()

    if debug:
        print("  Ca_final.shape:", Ca_final.shape)
        print("  Cb_final.shape:", Cb_final.shape)
        print(
            "  alpha orth err:",
            np.linalg.norm(Ca_final.conj().T @ S @ Ca_final - np.eye(Ca_final.shape[1])),
        )
        print(
            "  beta  orth err:",
            np.linalg.norm(Cb_final.conj().T @ S @ Cb_final - np.eye(Cb_final.shape[1])),
        )

    h0 = mol.energy_nuc()

    h1_alpha = Ca_final.conj().T @ ao_obi @ Ca_final
    h1_beta = Cb_final.conj().T @ ao_obi @ Cb_final

    norb_a = Ca_final.shape[1]
    norb_b = Cb_final.shape[1]

    # Chemist's notation: (ij|kl).
    h2_aa = ao2mo.kernel(
        mol,
        (Ca_final, Ca_final, Ca_final, Ca_final),
        compact=False,
    ).reshape(norb_a, norb_a, norb_a, norb_a)

    h2_bb = ao2mo.kernel(
        mol,
        (Cb_final, Cb_final, Cb_final, Cb_final),
        compact=False,
    ).reshape(norb_b, norb_b, norb_b, norb_b)

    h2_ab = ao2mo.kernel(
        mol,
        (Ca_final, Ca_final, Cb_final, Cb_final),
        compact=False,
    ).reshape(norb_a, norb_a, norb_b, norb_b)

    return h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab

# =============================================================================
# Main calculation for one separation
# =============================================================================

def compute_integrals_for_separation(
        r_frag,
        model="D4",
        d_hh=0.7414,
        basis="sto-3g",
        localized=False,
        outdir=".",
        stability_cycles=20,
):
    model = model.upper()
    outdir = Path(outdir)

    uhf_tensor_dir = outdir / "UHF_tensors"
    nuhf_tensor_dir = outdir / "NO-UHF_tensors"
    nuccsd_tensor_dir = outdir / "NO-UCCSD_tensors"
    log_dir = outdir / "unrestricted_logs"

    uhf_tensor_dir.mkdir(parents=True, exist_ok=True)
    nuhf_tensor_dir.mkdir(parents=True, exist_ok=True)
    nuccsd_tensor_dir.mkdir(parents=True, exist_ok=True)
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
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.output = str(log_dir / f"H4_{model}_R_{r_tag}.out")
    mol.build()

    #dm0 = None
    #if prev_mol is not None and prev_dm is not None:
    #    print("Projecting previous density matrix to current geometry...")
    #    dm0 = scf.addons.project_dm_nr2nr(prev_mol, prev_dm, mol)

    # -------------------------------------------------------------------------
    # UHF reference
    # -------------------------------------------------------------------------

    mf_stable = run_uhf(mol, stability_cycles=stability_cycles)

    # -------------------------------------------------------------------------
    # FCI benchmark energy
    # -------------------------------------------------------------------------

    efci, ci_vec, rhf_for_fci = compute_fci_energy(mol)

    print(f"UHF energy: {mf_stable.e_tot: .12f}")
    print(f"FCI energy: {efci: .12f}")
    print(f"UHF - FCI : {mf_stable.e_tot - efci: .12e}")

    label = f"H4_{model}_R_{r_frag}".replace(".", "p")

    # -------------------------------------------------------------------------
    # Case 1: UHF orbital tensors
    # -------------------------------------------------------------------------

    h0_uhf, h1a_uhf, h1b_uhf, eri_aa_uhf, eri_bb_uhf, eri_ab_uhf = get_active_space_tensors(
        mol,
        None,
        mf_stable,
        localized=localized,
        ccsd=False,
        natural=False,
        debug=True,
    )

    save_tensors(
        f"UHF_{label}",
        h0_uhf,
        h1a_uhf,
        h1b_uhf,
        eri_aa_uhf,
        eri_bb_uhf,
        eri_ab_uhf,
        folder=uhf_tensor_dir,
    )

    # -------------------------------------------------------------------------
    # Case 2: Natural UHF orbital tensors
    # -------------------------------------------------------------------------
    h0_nuhf, h1a_nuhf, h1b_nuhf, eri_aa_nuhf, eri_bb_nuhf, eri_ab_nuhf = get_active_space_tensors(
        mol,
        None,
        mf_stable,
        localized=False,
        ccsd=False,
        natural=True,
        debug=True,
    )

    save_tensors(
        f"Natural_UHF_{label}",
        h0_nuhf,
        h1a_nuhf,
        h1b_nuhf,
        eri_aa_nuhf,
        eri_bb_nuhf,
        eri_ab_nuhf,
        folder=nuhf_tensor_dir,
    )

    # -------------------------------------------------------------------------
    # Case 3: Natural UCCSD orbital tensors
    # -------------------------------------------------------------------------

    mycc = cc.UCCSD(mf_stable)
    mycc.conv_tol = CC_CONV_TOL
    mycc.conv_tol_normt = CC_CONV_TOL_NORMT
    mycc.max_cycle = CC_MAX_CYCLE
    mycc.diis_space = CC_DIIS_SPACE
    mycc.verbose = 4

    mycc.kernel()

    if not mycc.converged:
        print("WARNING: UCCSD did not fully converge.")

    print(f"UCCSD correlation energy: {mycc.e_corr: .12f}")
    print(f"UCCSD total energy      : {mycc.e_tot: .12f}")
    print(f"UCCSD - FCI            : {mycc.e_tot - efci: .12e}")

    no_coeff_uccsd, no_occ_uccsd = get_uccsd_natural_orbitals(
        mycc,
        mf_stable,
        debug=True,
    )

    h0_nucc, h1a_nucc, h1b_nucc, eri_aa_nucc, eri_bb_nucc, eri_ab_nucc = get_active_space_tensors(
        mol,
        no_coeff_uccsd,
        mf_stable,
        localized=False,
        ccsd=True,
        natural=True,
        debug=True,
    )

    save_tensors(
        f"Natural_UCCSD_{label}",
        h0_nucc,
        h1a_nucc,
        h1b_nucc,
        eri_aa_nucc,
        eri_bb_nucc,
        eri_ab_nucc,
        folder=nuccsd_tensor_dir,
    )

    diagnostics = {
        "r_frag": r_frag,
        "hf_energy": mf_stable.e_tot,
        "efci": efci,
        "uccsd": mycc.e_tot,
        "uccsd_corr": mycc.e_corr,
        "uhf_minus_fci": mf_stable.e_tot - efci,
        "uccsd_minus_fci": mycc.e_tot - efci,
        "uhf_s2": mf_stable.spin_square()[0],
        "uhf_converged": mf_stable.converged,
        "uccsd_converged": mycc.converged,
        "uccsd_no_occ_alpha": no_occ_uccsd[0],
        "uccsd_no_occ_beta": no_occ_uccsd[1],
    }

    return (
        h0_nucc,
        h1a_nucc,
        h1b_nucc,
        eri_aa_nucc,
        eri_bb_nucc,
        eri_ab_nucc,
        mol,
        mf_stable,
        mycc,
        efci,
        diagnostics,
    )


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
            h1a,
            h1b,
            eri_aa,
            eri_bb,
            eri_ab,
            mol,
            mf,
            mcc,
            efci,
            diagnostics,
        ) = compute_integrals_for_separation(
            r_frag=r_frag,
            model=model,
            d_hh=d,
            basis="sto-3g",
            localized=False,  # important: preserve strict CCSD NO basis
            outdir=outdir,
            stability_cycles=20,
        )

        results[model][r_frag] = diagnostics

plot_data = plot_energies(
    results,
    savepath=outdir / "energy_vs_H4_separation_UHF_CCSD_FCI_all_models.png",
)

plot_noons = plot_uccsd_natural_occupations(
    results,
savepath=outdir / "UCCSD_NOONS_all_models.png",
)

plot_noons = plot_uccsd_natural_occupations_by_model(
    results,
    savepath=outdir / "UCCSD_NOONS_subplots_by_model.png",
)