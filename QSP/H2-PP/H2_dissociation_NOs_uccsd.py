# Full scrip that produces molecular integrals for H2 PES
# at different levels.
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pyscf import gto, scf, fci, cc
from pyscf import lo, ao2mo
from pyscf.lib import logger

# =============================================================================
# User settings
# =============================================================================

BASIS = "sto-3g"

SCF_CONV_TOL = 1e-12
FCI_CONV_TOL = 1e-12

CC_CONV_TOL = 1e-10
CC_CONV_TOL_NORMT = 1e-8
CC_MAX_CYCLE = 300
CC_DIIS_SPACE = 12

STABILITY_TOL = 1e-8
STABILITY_CYCLES = 10

SHOW_PLOTS = True

# Fixed H-H distance label from your original script.
d = 0.7414

# Selected H-H separations.
R_FRAGMENT_VALUES = np.array(
    [
        0.50,
        0.60,
        d,
        0.90,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
    ]
)

# =============================================================================
# Geometry and SCF helpers
# =============================================================================

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


def make_h2_pes(r):
    """
    Linear two-H system:

        H -- r -- H

    r is the H-H separation in Angstrom.
    """
    if r <= 0:
        raise ValueError("r_frag must be positive.")

    z = np.array(
        [
            0.0,
            r,
        ]
    )

    return "\n".join(f"H 0.0 0.0 {zi:.10f}" for zi in z)


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


def run_uhf(mol, stability_cycles=STABILITY_CYCLES):
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


# =============================================================================
# Main calculation for one separation
# =============================================================================

def compute_integrals_for_separation(
    r_frag,
    basis=BASIS,
    localized=False,
    outdir=".",
    stability_cycles=STABILITY_CYCLES,
):
    outdir = Path(outdir)

    uhf_tensor_dir = outdir / "uhf_tensors"
    nuhf_tensor_dir = outdir / "nuhf_tensors"
    nuccsd_tensor_dir = outdir / "nuccsd_tensors"
    log_dir = outdir / "logs"

    uhf_tensor_dir.mkdir(parents=True, exist_ok=True)
    nuhf_tensor_dir.mkdir(parents=True, exist_ok=True)
    nuccsd_tensor_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    atom = make_h2_pes(r_frag)

    print("=" * 80)
    print(f"2-H separation R = {r_frag} Angstrom")
    print(atom)
    print("=" * 80)

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.output = str(log_dir / f"H_H_R_{r_frag:.6f}.out".replace(".", "p"))
    mol.build()

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

    label = f"H2_R_{r_frag}".replace(".", "p")

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


# =============================================================================
# Plotting
# =============================================================================

def plot_energies(results, savepath=None, show=True):
    r_values = np.array(sorted(results.keys()), dtype=float)

    hf_energies = np.array([results[r]["hf_energy"] for r in r_values])
    fci_energies = np.array([results[r]["efci"] for r in r_values])
    uccsd_energies = np.array([results[r]["uccsd"] for r in r_values])

    plt.figure(figsize=(6, 4))
    plt.plot(r_values, hf_energies, marker="o", label="UHF")
    plt.plot(r_values, fci_energies, marker="o", label="FCI")
    plt.plot(r_values, uccsd_energies, marker="o", label="UCCSD")
    plt.xlabel("H-H separation R / Angstrom")
    plt.ylabel("Energy / Hartree")
    plt.title("H2/STO-3G Energies")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"Energy plot saved to {savepath}")

    if show:
        plt.show()

    return r_values, hf_energies


def plot_energy_differences(results, savepath=None, show=True):
    r_values = np.array(sorted(results.keys()), dtype=float)

    uhf_minus_fci = np.array([results[r]["uhf_minus_fci"] for r in r_values])
    uccsd_minus_fci = np.array([results[r]["uccsd_minus_fci"] for r in r_values])

    plt.figure(figsize=(6, 4))
    plt.plot(r_values, uhf_minus_fci, marker="o", label="UHF - FCI")
    plt.plot(r_values, uccsd_minus_fci, marker="o", label="UCCSD - FCI")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("H-H separation R / Angstrom")
    plt.ylabel("Energy difference / Hartree")
    plt.title("Energy Errors Relative to FCI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"Energy-difference plot saved to {savepath}")

    if show:
        plt.show()

    return r_values, uhf_minus_fci, uccsd_minus_fci


def plot_uccsd_natural_occupations(results, savepath=None, show=True):
    r_values = np.array(sorted(results.keys()), dtype=float)

    occ_a_0 = np.array([results[r]["uccsd_no_occ_alpha"][0] for r in r_values])
    occ_a_1 = np.array([results[r]["uccsd_no_occ_alpha"][1] for r in r_values])
    occ_b_0 = np.array([results[r]["uccsd_no_occ_beta"][0] for r in r_values])
    occ_b_1 = np.array([results[r]["uccsd_no_occ_beta"][1] for r in r_values])

    plt.figure(figsize=(6, 4))
    plt.plot(r_values, occ_a_0, marker="o", label="alpha NO occ 1")
    plt.plot(r_values, occ_a_1, marker="o", label="alpha NO occ 2")
    plt.plot(r_values, occ_b_0, marker="s", label="beta NO occ 1")
    plt.plot(r_values, occ_b_1, marker="s", label="beta NO occ 2")
    plt.xlabel("H-H separation R / Angstrom")
    plt.ylabel("Natural occupation")
    plt.title("UCCSD Natural Occupations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"Natural-occupation plot saved to {savepath}")

    if show:
        plt.show()

    return r_values


# =============================================================================
# Driver
# =============================================================================

def main():
    outdir = Path.cwd()

    results = {}

    for r_frag in R_FRAGMENT_VALUES:
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
            basis=BASIS,
            localized=False,
            outdir=outdir,
            stability_cycles=STABILITY_CYCLES,
        )

        results[float(r_frag)] = diagnostics

    plot_energies(
        results,
        savepath=outdir / "energy_vs_H2_separation.png",
        show=SHOW_PLOTS,
    )

    plot_energy_differences(
        results,
        savepath=outdir / "energy_errors_vs_H2_separation.png",
        show=SHOW_PLOTS,
    )

    plot_uccsd_natural_occupations(
        results,
        savepath=outdir / "uccsd_natural_occupations_vs_H2_separation.png",
        show=SHOW_PLOTS,
    )

if __name__ == "__main__":
    main()