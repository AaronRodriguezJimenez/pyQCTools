# qc_strong_correlation_diagnostics.py

import numpy as np
from pyscf.fci import cistring


def _as_nelec_tuple(nelec):
    """
    Convert PySCF nelec format to (nalpha, nbeta).
    """
    if isinstance(nelec, tuple):
        return nelec
    nalpha = nelec // 2
    nbeta = nelec - nalpha
    return nalpha, nbeta


def binary_entropy(x, eps=1e-15):
    """
    Binary entropy for values x in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1.0 - eps)
    return -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))


def fci_one_rdm_spin_summed(fcivec, norb, nelec):
    """
    Compute the spin-summed 1-RDM from a PySCF FCI vector.

    Returns
    -------
    dm1 : ndarray, shape (norb, norb)
        Spin-summed 1-RDM in the orbital basis used for FCI.
    dm1a : ndarray
        Alpha 1-RDM.
    dm1b : ndarray
        Beta 1-RDM.
    """
    from pyscf import fci

    dm1a, dm1b = fci.direct_spin1.make_rdm1s(fcivec, norb, nelec)
    dm1 = dm1a + dm1b

    dm1 = 0.5 * (dm1 + dm1.T)
    dm1a = 0.5 * (dm1a + dm1a.T)
    dm1b = 0.5 * (dm1b + dm1b.T)

    return dm1, dm1a, dm1b


def natural_occupations_from_dm1(dm1):
    """
    Natural occupations from a spin-summed spatial-orbital 1-RDM.

    For a closed-shell-like spatial 1-RDM, occupations lie between 0 and 2.
    """
    evals, evecs = np.linalg.eigh(0.5 * (dm1 + dm1.T))
    idx = np.argsort(evals)[::-1]

    occ = evals[idx]
    coeff = evecs[:, idx]

    return occ, coeff


def natural_occupation_diagnostics(occ, nelec):
    """
    Diagnostics based on spatial natural orbital occupations.

    Parameters
    ----------
    occ : array
        Spatial natural occupations, usually between 0 and 2.
    nelec : int or tuple
        Number of electrons.

    Returns
    -------
    dict
    """
    occ = np.asarray(occ, dtype=float)
    nalpha, nbeta = _as_nelec_tuple(nelec)
    ne = nalpha + nbeta

    # For spatial orbitals, the nominal number of occupied orbitals
    # in a closed-shell reference is nelec / 2.
    nocc = min(nalpha, nbeta)
    nocc = int(nocc)

    # Fractional occupation / effectively unpaired electron index.
    # YAMAGUCHI INDEX:
    # This is small for occupations close to 0 or 2.
    # It is maximal for occupations close to 1.
    n_unpaired = float(np.sum(occ * (2.0 - occ)))

    # A normalized version can be useful for comparing different sizes.
    n_unpaired_per_electron = n_unpaired / ne if ne > 0 else np.nan

    # Frontier natural occupation gap.
    if 0 < nocc < len(occ):
        hono = float(occ[nocc - 1])
        luno = float(occ[nocc])
        hono_luno_gap = hono - luno
    else:
        hono = np.nan
        luno = np.nan
        hono_luno_gap = np.nan

    # Occupation entropy using f_i = n_i / 2.
    # This is not the same as orbital entanglement entropy,
    # but it is a useful scalar measure of fractional occupations.
    f = occ / 2.0
    no_entropy = float(np.sum(binary_entropy(f)))

    # Count orbitals with significant fractional occupation.
    # You can tune the thresholds.
    frac_mask = (occ > 0.02) & (occ < 1.98)
    active_fractional_orbitals = int(np.sum(frac_mask))

    radical_mask = (occ > 0.5) & (occ < 1.5)
    radical_like_orbitals = int(np.sum(radical_mask))

    return {
        "n_unpaired": n_unpaired,
        "n_unpaired_per_electron": n_unpaired_per_electron,
        "hono": hono,
        "luno": luno,
        "hono_luno_gap": hono_luno_gap,
        "no_entropy": no_entropy,
        "active_fractional_orbitals": active_fractional_orbitals,
        "radical_like_orbitals": radical_like_orbitals,
    }


def fci_ci_vector_diagnostics(fcivec, norb, nelec, eps=1e-15):
    """
    Diagnostics from the FCI coefficient vector.

    Computes:
    - dominant determinant weight
    - RHF-like reference determinant weight
    - CI entropy
    - effective number of determinants
    """
    nalpha, nbeta = _as_nelec_tuple(nelec)

    nstr_a = cistring.num_strings(norb, nalpha)
    nstr_b = cistring.num_strings(norb, nbeta)

    ci = np.asarray(fcivec).reshape(nstr_a, nstr_b)

    norm = float(np.vdot(ci, ci).real)

    if norm <= 0:
        raise ValueError("FCI vector has zero norm.")

    p = np.abs(ci.ravel()) ** 2
    p = p / np.sum(p)

    p_nonzero = p[p > eps]

    ci_entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    ci_neff = float(np.exp(ci_entropy))
    dominant_weight = float(np.max(p))

    # RHF-like closed-shell reference determinant:
    # alpha and beta occupy the lowest nalpha/nbeta spatial orbitals.
    ref_str_a = sum(1 << i for i in range(nalpha))
    ref_str_b = sum(1 << i for i in range(nbeta))

    ref_addr_a = cistring.str2addr(norb, nalpha, ref_str_a)
    ref_addr_b = cistring.str2addr(norb, nbeta, ref_str_b)

    reference_weight = float(abs(ci[ref_addr_a, ref_addr_b]) ** 2 / norm)

    return {
        "ci_norm": norm,
        "ci_reference_weight": reference_weight,
        "ci_multiconfig_index": 1.0 - reference_weight,
        "ci_dominant_weight": dominant_weight,
        "ci_entropy": ci_entropy,
        "ci_neff": ci_neff,
    }


def uhf_spin_diagnostics(umf):
    """
    UHF spin contamination diagnostics.

    Returns <S^2> and spin multiplicity-like value.
    """
    if umf is None:
        return {
            "uhf_s2": np.nan,
            "uhf_multiplicity": np.nan,
        }

    try:
        s2, mult = umf.spin_square()
        return {
            "uhf_s2": float(s2),
            "uhf_multiplicity": float(mult),
        }
    except Exception:
        return {
            "uhf_s2": np.nan,
            "uhf_multiplicity": np.nan,
        }


def energy_diagnostics(e_rhf=None, e_uhf=None, e_fci=None):
    """
    Energy-based correlation diagnostics.
    """
    out = {}

    if e_rhf is not None and e_fci is not None:
        out["e_corr_rhf"] = float(e_fci - e_rhf)
        out["abs_e_corr_rhf"] = float(e_rhf - e_fci)
    else:
        out["e_corr_rhf"] = np.nan
        out["abs_e_corr_rhf"] = np.nan

    if e_uhf is not None and e_fci is not None:
        out["e_corr_uhf"] = float(e_fci - e_uhf)
        out["abs_e_corr_uhf"] = float(e_uhf - e_fci)
    else:
        out["e_corr_uhf"] = np.nan
        out["abs_e_corr_uhf"] = np.nan

    return out


def collect_fci_strong_correlation_diagnostics(
    fcivec,
    norb,
    nelec,
    e_fci,
    e_rhf=None,
    e_uhf=None,
    umf=None,
    prefix="fci",
):
    """
    Collect all strong-correlation diagnostics from an FCI calculation.

    Returns
    -------
    row : dict
        Flat dictionary suitable for appending to a pandas DataFrame.
    """
    row = {}

    row.update(energy_diagnostics(e_rhf=e_rhf, e_uhf=e_uhf, e_fci=e_fci))
    row.update(uhf_spin_diagnostics(umf))

    dm1, dm1a, dm1b = fci_one_rdm_spin_summed(fcivec, norb, nelec)
    occ, no_coeff = natural_occupations_from_dm1(dm1)

    # Reorder orbitals descending
    idx = occ.argsort()[::-1]
    occ = occ[idx]
    no_coeff = no_coeff[:, idx]


    no_diag = natural_occupation_diagnostics(occ, nelec)
    ci_diag = fci_ci_vector_diagnostics(fcivec, norb, nelec)

    for key, value in no_diag.items():
        row[f"{prefix}_{key}"] = value

    for key, value in ci_diag.items():
        row[f"{prefix}_{key}"] = value

    # Store natural occupations as a compact string for CSV output.
    row[f"{prefix}_natural_occupations"] = " ".join(f"{x:.10f}" for x in occ)

    return row