from pathlib import Path
import numpy as np

from pyscf import gto, scf, cc
from pyscf.mcscf import addons


d = 0.7414


def tag_float(r):
    return f"{r:.6g}".replace(".", "p").replace("-", "m")


def make_S4_paldus_geometry(r_frag):
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


def load_tensors(path):
    data = np.load(path, allow_pickle=True)

    return {
        "path": Path(path),
        "keys": list(data.keys()),
        "h0": float(np.asarray(data["hc"])),
        "h1": np.asarray(data["h1e"]),
        "h2": np.asarray(data["h2e"]),
    }


def rel_norm_diff(a, b, eps=1e-14):
    return np.linalg.norm((a - b).ravel()) / max(
        np.linalg.norm(a.ravel()),
        np.linalg.norm(b.ravel()),
        eps,
    )


def max_abs_diff(a, b):
    return np.max(np.abs(a - b))


def rotate_h1(h1, U):
    """
    h'_ab = sum_pq U_pa h_pq U_qb
    """
    return U.T @ h1 @ U


def rotate_h2_chemists(h2, U):
    """
    Rotate chemists' notation ERIs:

        g'_{abcd} = sum_pqrs U_pa U_qb U_rc U_sd g_{pqrs}

    Assumes h2 is PySCF/chemist notation (pq|rs).
    """
    return np.einsum(
        "pa,qb,rc,sd,pqrs->abcd",
        U,
        U,
        U,
        U,
        h2,
        optimize=True,
    )


def build_rhf_ccsd_no_rotation_for_s4(
    r_frag,
    basis="sto-3g",
    conv_tol=1e-10,
    cc_conv_tol=1e-10,
):
    """
    Recompute only RHF, CCSD, and the CCSD natural-orbital rotation U.

    This does not rebuild or save integral tensors.

    Returns
    -------
    mol, mf, mycc, noons, natorbs, U
    """
    mol = gto.Mole()
    mol.atom = make_S4_paldus_geometry(r_frag)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.kernel()

    if not mf.converged:
        print(f"WARNING: RHF did not converge at R={r_frag}")

    mycc = cc.CCSD(mf)
    mycc.conv_tol = cc_conv_tol
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 200
    mycc.kernel()

    if not mycc.converged:
        print(f"WARNING: CCSD did not converge at R={r_frag}")

    noons, natorbs = addons.make_natural_orbitals(mycc)

    S = mol.intor_symmetric("int1e_ovlp")

    # U transforms RHF canonical MOs to CCSD natural orbitals:
    #
    #     C_NO = C_RHF @ U
    #
    U = mf.mo_coeff.T @ S @ natorbs

    return mol, mf, mycc, np.asarray(noons), np.asarray(natorbs), U


def check_ccsd_no_rotation(mf, mycc, noons, U):
    """
    Verify that U diagonalizes the CCSD 1-RDM.
    """
    dm1_mo = mycc.make_rdm1(ao_repr=False)
    dm1_mo = 0.5 * (dm1_mo + dm1_mo.T)

    dm1_no = U.T @ dm1_mo @ U

    offdiag = dm1_no - np.diag(np.diag(dm1_no))

    return {
        "U_unitarity_error": np.linalg.norm(U.T @ U - np.eye(U.shape[1])),
        "dm1_no_offdiag_norm": np.linalg.norm(offdiag),
        "dm1_no_offdiag_max": np.max(np.abs(offdiag)),
        "noon_diag_error": np.linalg.norm(np.diag(dm1_no) - noons),
        "noon_diag_max_error": np.max(np.abs(np.diag(dm1_no) - noons)),
        "noons": noons,
        "dm1_no": dm1_no,
    }


def compare_saved_ccsdno_to_recomputed_no_rotation(
    r_frag,
    rhf_path,
    ccsdno_path,
    basis="sto-3g",
):
    """
    Main diagnostic.

    It checks whether the saved CCSDNO tensors equal the saved RHF tensors
    rotated by the recomputed CCSD natural-orbital rotation.
    """
    rhf = load_tensors(rhf_path)
    saved_no = load_tensors(ccsdno_path)

    mol, mf, mycc, noons, natorbs, U = build_rhf_ccsd_no_rotation_for_s4(
        r_frag,
        basis=basis,
    )

    no_diag = check_ccsd_no_rotation(mf, mycc, noons, U)

    h1_pred = rotate_h1(rhf["h1"], U)
    h2_pred = rotate_h2_chemists(rhf["h2"], U)

    print("\n" + "=" * 100)
    print(f"R = {r_frag:.6f}")
    print("=" * 100)
    print(f"RHF tensor file:     {rhf['path']}")
    print(f"CCSDNO tensor file:  {saved_no['path']}")
    print(f"RHF keys:            {rhf['keys']}")
    print(f"CCSDNO keys:         {saved_no['keys']}")

    print("\nCCSD natural-orbital diagnostic")
    print(f"  CCSD converged:              {mycc.converged}")
    print(f"  CCSD total energy:           {mycc.e_tot: .12f}")
    print(f"  Natural occupations:         {noons}")
    print(f"  sum(NOONs):                  {np.sum(noons):.12f}")
    print(f"  U unitarity error:           {no_diag['U_unitarity_error']:.3e}")
    print(f"  dm1 NO offdiag norm:         {no_diag['dm1_no_offdiag_norm']:.3e}")
    print(f"  dm1 NO offdiag max:          {no_diag['dm1_no_offdiag_max']:.3e}")
    print(f"  NOON diag norm error:        {no_diag['noon_diag_error']:.3e}")
    print(f"  NOON diag max error:         {no_diag['noon_diag_max_error']:.3e}")

    print("\nSaved CCSDNO vs RHF tensors rotated by recomputed CCSD-NO U")
    print(f"  |h0_saved - h0_pred|:        {abs(saved_no['h0'] - rhf['h0']):.12e}")
    print(f"  max |h1_saved - h1_pred|:    {max_abs_diff(saved_no['h1'], h1_pred):.12e}")
    print(f"  rel ||h1_saved - h1_pred||:  {rel_norm_diff(saved_no['h1'], h1_pred):.12e}")
    print(f"  max |h2_saved - h2_pred|:    {max_abs_diff(saved_no['h2'], h2_pred):.12e}")
    print(f"  rel ||h2_saved - h2_pred||:  {rel_norm_diff(saved_no['h2'], h2_pred):.12e}")

    return {
        "R": r_frag,
        "noons": noons,
        "ccsd_converged": mycc.converged,
        "ccsd_energy": mycc.e_tot,
        "U_unitarity_error": no_diag["U_unitarity_error"],
        "dm1_no_offdiag_norm": no_diag["dm1_no_offdiag_norm"],
        "dm1_no_offdiag_max": no_diag["dm1_no_offdiag_max"],
        "noon_diag_error": no_diag["noon_diag_error"],
        "noon_diag_max_error": no_diag["noon_diag_max_error"],
        "h1_max_abs_diff_pred": max_abs_diff(saved_no["h1"], h1_pred),
        "h1_rel_diff_pred": rel_norm_diff(saved_no["h1"], h1_pred),
        "h2_max_abs_diff_pred": max_abs_diff(saved_no["h2"], h2_pred),
        "h2_rel_diff_pred": rel_norm_diff(saved_no["h2"], h2_pred),
    }


def run_no_rotation_comparison_for_cases(r_values, base_rhf, base_ccsdno):
    base_rhf = Path(base_rhf)
    base_ccsdno = Path(base_ccsdno)

    rows = []

    for r in r_values:
        tag = tag_float(r)

        rhf_path = base_rhf / f"h4_RHF_S4_R_{tag}_tensors.npz"
        ccsdno_path = base_ccsdno / f"h4_CCSDNO_S4_R_{tag}_tensors.npz"

        row = compare_saved_ccsdno_to_recomputed_no_rotation(
            r,
            rhf_path,
            ccsdno_path,
            basis="sto-3g",
        )

        rows.append(row)

    print("\n" + "=" * 120)
    print("Compact CCSD-NO rotation verification")
    print("=" * 120)
    print(
        f"{'R':>8s}  "
        f"{'NOON sum':>12s}  "
        f"{'dm1 offdiag':>12s}  "
        f"{'h1 rel':>12s}  "
        f"{'h2 rel':>12s}  "
        f"{'NOONs':>40s}"
    )

    for row in rows:
        print(
            f"{row['R']:8.4f}  "
            f"{np.sum(row['noons']):12.8f}  "
            f"{row['dm1_no_offdiag_norm']:12.4e}  "
            f"{row['h1_rel_diff_pred']:12.4e}  "
            f"{row['h2_rel_diff_pred']:12.4e}  "
            f"{str(np.round(row['noons'], 8)):>40s}"
        )

    return rows

BASE_RHF = "/Users/admin/PycharmProjects/pyQCTools/QSP/H4-PP/tensors"
BASE_CCSDNO = "/Users/admin/PycharmProjects/pyQCTools/QSP/H4-PP/NO-CCSD_tensors"

R_VALUES = [0.50, 0.7414, 1.00, 1.50, 2.00, 2.50, 3.00]

rows = run_no_rotation_comparison_for_cases(
    R_VALUES,
    BASE_RHF,
    BASE_CCSDNO,
)

def build_reference_tensors_from_same_mf(
    r_frag,
    basis="sto-3g",
):
    from pyscf import gto, scf, mcscf, cc
    from pyscf.mcscf import addons
    from src.pyqctools.int_fcns import get_restricted_active_space_integrals

    mol = gto.Mole()
    mol.atom = make_S4_paldus_geometry(r_frag)
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 200
    mycc.kernel()

    noons, natorbs = addons.make_natural_orbitals(mycc)

    ncore = 0
    ncas = mol.nao_nr()
    nelecas = mol.nelectron

    # Canonical RHF tensors from this exact mf
    mc_rhf = mcscf.CASCI(mf, ncas, nelecas)
    mc_rhf.mo_coeff = mf.mo_coeff.copy()

    h0_rhf, h1_rhf, h2_rhf = get_restricted_active_space_integrals(
        mol,
        mf,
        mc_rhf,
        localized=False,
        include_all_noncore=False,
    )

    # CCSD NO tensors from this exact mf/mycc
    mc_no = mcscf.CASCI(mf, ncas, nelecas)
    mc_no.mo_coeff = natorbs.copy()

    h0_no, h1_no, h2_no = get_restricted_active_space_integrals(
        mol,
        mf,
        mc_no,
        localized=False,
        include_all_noncore=False,
    )

    return {
        "mol": mol,
        "mf": mf,
        "mycc": mycc,
        "noons": noons,
        "h0_rhf": h0_rhf,
        "h1_rhf": h1_rhf,
        "h2_rhf": h2_rhf,
        "h0_no": h0_no,
        "h1_no": h1_no,
        "h2_no": h2_no,
    }

def compare_saved_ccsdno_to_direct_recomputed(r_frag, saved_ccsdno_path):
    saved = load_tensors(saved_ccsdno_path)

    ref = build_reference_tensors_from_same_mf(r_frag)

    print("\n" + "=" * 100)
    print(f"Direct recomputed CCSDNO comparison, R = {r_frag}")
    print("=" * 100)

    print(f"NOONs: {ref['noons']}")
    print(f"sum(NOONs): {np.sum(ref['noons']):.12f}")
    print(f"CCSD converged: {ref['mycc'].converged}")
    print(f"CCSD energy: {ref['mycc'].e_tot:.12f}")

    print("\nSaved CCSDNO vs directly recomputed CCSDNO")
    print(f"  |h0_saved - h0_ref|:       {abs(saved['h0'] - ref['h0_no']):.12e}")
    print(f"  h1 rel diff:               {rel_norm_diff(saved['h1'], ref['h1_no']):.12e}")
    print(f"  h2 rel diff:               {rel_norm_diff(saved['h2'], ref['h2_no']):.12e}")
    print(f"  h1 max abs diff:           {max_abs_diff(saved['h1'], ref['h1_no']):.12e}")
    print(f"  h2 max abs diff:           {max_abs_diff(saved['h2'], ref['h2_no']):.12e}")

BASE_CCSDNO = "/Users/admin/PycharmProjects/pyQCTools/QSP/H4-PP/NO-CCSD_tensors"

for r in [0.50, 0.7414, 1.00, 1.50, 2.00, 2.50, 3.00]:
    tag = tag_float(r)
    path = Path(BASE_CCSDNO) / f"h4_CCSDNO_S4_R_{tag}_tensors.npz"
    compare_saved_ccsdno_to_direct_recomputed(r, path)