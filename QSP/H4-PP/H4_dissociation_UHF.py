"""
Integrals for H2-H2 / H4 Chains

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

from pyscf import gto, scf
from pyscf import lo, ao2mo
from scipy.linalg import eigh
from pyscf.lib.exceptions import LinearDependencyError
from pyscf.lib import logger

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

def safe_uhf_stability(
    mf,
    internal=True,
    external=False,
    return_status=True,
    nroots=1,
    tol=1e-7,
):
    """
    Safe wrapper around mf.stability.

    External UHF stability can fail for near-degenerate stretched H systems
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


def run_stable_uhf(
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
    Run UHF and optimize until internally stable.

    External stability is treated as a diagnostic only. For stretched H chains,
    UHF is often externally unstable because a spin-broken UHF solution exists.
    """

    mf = scf.UHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_cycle
    mf.diis_space = 12

    # These help avoid oscillations near degeneracies.
    mf.level_shift = level_shift
    mf.damp = damping

    print("Running initial UHF...")
    mf.kernel(dm0=dm0)

    # If it converged with level shift/damping, remove them and polish.
    if mf.converged:
        print("Polishing UHF without level shift/damping...")
        dm_polish = mf.make_rdm1()
        mf.level_shift = 0.0
        mf.damp = 0.0
        mf.kernel(dm0=dm_polish)

    if not mf.converged and use_newton_if_needed:
        print("UHF did not converge. Retrying with Newton solver...")
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
        msg = "UHF did not converge."
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
        print(f"UHF internal stability check cycle {cyc}")

        mo_i, _, stable_i, _ = safe_uhf_stability(
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
            print("  Internal stability check failed. Keeping current UHF solution.")
            break

        print("  Re-optimizing UHF from internally improved orbitals...")
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
        print("Running external UHF stability diagnostic...")

        _, mo_e, _, stable_e = safe_uhf_stability(
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
                "WARNING: UHF is externally unstable. "
                "For stretched H systems this usually means a lower-energy "
                "spin-broken UHF solution exists."
            )

        if stable_e is None:
            print(
                "WARNING: External stability diagnostic failed. "
                "This does not necessarily mean the UHF calculation failed."
            )

    if info["internal_stable"] is not True:
        msg = "UHF solution was not confirmed internally stable."

        if raise_on_unstable:
            raise RuntimeError(msg)

        print("WARNING:", msg)

    return mf, info

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

def compute_integrals_for_separation(
    r_frag,
    model="D4",
    d_hh=0.7414,
    basis="sto-3g",
    localized=True,
    outdir=".",
    stability_cycles=20,
    prev_mol=None,
    prev_dm=None,
    check_external=True,
):
    model = model.upper()

    outdir = Path(outdir)
    tensor_dir = outdir / "UHF-tensors"
    log_dir = outdir / "UHF-logs"

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

    # Use the previous geometry's density as the next initial guess.
    # Projection is better than directly reusing prev_dm because the AO basis
    # changes when the atoms move.
    dm0 = None
    if prev_mol is not None and prev_dm is not None:
        print("Projecting previous density matrix to current geometry...")
        dm0 = scf.addons.project_dm_nr2nr(prev_mol, prev_dm, mol)

    mf, mf_info = run_stable_uhf(
        mol,
        dm0=dm0,
        stability_cycles=stability_cycles,
        check_external=check_external,
        conv_tol=1e-10,
        conv_tol_grad=1e-7,
        stability_tol=1e-7,
        use_newton_if_needed=False,
        level_shift=0.5,
        damping=0.2,
        raise_on_unstable=False,
    )

    h0_uhf, h1a_uhf, h1b_uhf, eri_aa_uhf, eri_bb_uhf, eri_ab_uhf = get_active_space_tensors(
        mol,
        None,
        mf,
        localized=localized,
        ccsd=False,
        natural=False,
        debug=True,
    )

    label = f"h{n}_UHF_{model}_R_{r_tag}"
    save_tensors(
        label,
        h0_uhf,
        h1a_uhf,
        h1b_uhf,
        eri_aa_uhf,
        eri_bb_uhf,
        eri_ab_uhf,
        folder=tensor_dir,
    )

    final_dm = mf.make_rdm1()

    return mol, mf, final_dm, mf_info

def plot_hf_energies(results, savepath=None, show=True):
    """
    Plot RHF energy as a function of the H4 separation parameter.

    Parameters
    ----------
    results : dict
        Nested dictionary of the form:

            results[model][r_frag]["hf_energy"]

    savepath : str or Path, optional
        If provided, saves the plot to this path.

    show : bool
        If True, displays the plot.
    """
    plt.figure(figsize=(7, 4.5))

    plot_data = {}

    for model, model_results in results.items():
        r_values = np.array(sorted(model_results.keys()), dtype=float)
        hf_energies = np.array(
            [model_results[r]["hf_energy"] for r in r_values]
        )

        plt.plot(r_values, hf_energies, marker="o", label=model)

        plot_data[model] = {
            "r_values": r_values,
            "hf_energies": hf_energies,
        }

    plt.xlabel("H4 separation parameter R / Angstrom")
    plt.ylabel("UHF energy / Hartree")
    plt.title("H4 UHF Energy for LIN4, D4, and S4 Geometries")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"HF energy plot saved to {savepath}")

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
        mol, mf, final_dm, mf_info = compute_integrals_for_separation(
            r_frag=r_frag,
            model=model,
            d_hh=d,
            basis="sto-3g",
            localized=False,
            outdir=outdir,
            stability_cycles=20,
            prev_mol=prev_mol,
            prev_dm=prev_dm,
            check_external=True,)

        results[model][r_frag] = {
            "hf_energy": mf.e_tot,
            "mf_converged": mf_info["converged"],
            "internal_stable": mf_info["internal_stable"],
            "external_stable": mf_info["external_stable"],
            "stability_cycles": mf_info["stability_cycles"],
        }

        prev_mol = mol
        prev_dm = final_dm

print("\nSCF stability summary")
print("=" * 80)

for model, model_results in results.items():
    print(f"\nModel: {model}")
    for r_frag in sorted(model_results.keys()):
        row = model_results[r_frag]
        print(
            f"R = {r_frag:8.4f}  "
            f"E_UHF = {row['hf_energy']: .12f}  "
            f"conv = {row['mf_converged']}  "
            f"int_stable = {row['internal_stable']}  "
            f"ext_stable = {row['external_stable']}  "
            f"stab_cycles = {row['stability_cycles']}"
        )

plot_data = plot_hf_energies(
    results,
    savepath=outdir / "uhf_energy_vs_H4_separation_all_models.png",
)