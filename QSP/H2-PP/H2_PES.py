import csv
import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo, fci,cc
from pyscf.lib import logger


BASIS = "sto-3g"

R_MIN = 0.50
R_MAX = 3.00
R_STEP = 0.1

SCF_CONV_TOL = 1e-12
FCI_CONV_TOL = 1e-12
STABILITY_CYCLES = 10
STABILITY_TOL = 1e-8

CSV_FILE = "h2_sto3g_uhf_fci_pes.csv"
ENERGY_PLOT = "h2_sto3g_uhf_fci_pes.png"
DIFF_PLOT = "h2_sto3g_uhf_minus_fci.png"

SHOW_PLOTS = True


def build_h2(r_angstrom):
    mol = gto.M(
        atom=[
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, r_angstrom)],
        ],
        basis=BASIS,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    return mol


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


def stable_opt_internal(mf, stability_cycles=STABILITY_CYCLES, stability_tol=STABILITY_TOL):
    log = logger.new_logger(mf)

    mo1, _, stable, _ = mf.stability(
        internal=True,
        external=False,
        return_status=True,
        tol=stability_tol,
    )

    cyc = 0
    print("STABLE?", stable)

    while (not stable) and cyc < stability_cycles:
        log.note("Try to optimize orbitals until stable, attempt %d" % cyc)

        dm1 = mf.make_rdm1(mo1, mf.mo_occ)

        mf = mf.run(dm1)

        mo1, _, stable, _ = mf.stability(
            internal=True,
            external=True,
            return_status=True,
            tol=stability_tol,
        )

        print("STABLE?", stable)
        cyc += 1

    if not stable:
        log.note("Stability optimization failed after %d attempts" % cyc)

    return mf, stable, cyc


def run_uhf(mol):
    mf = scf.UHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = 200
    mf.verbose = 0

    dm0 = make_broken_symmetry_guess(mf)
    mf.kernel(dm0)

    if not mf.converged:
        mf.max_cycle = 500
        mf.kernel(dm0)

    mf, stable, stability_attempts = stable_opt_internal(mf)

    s2, mult = mf.spin_square()

    return mf , {
        "energy": float(mf.e_tot),
        "s2": float(s2),
        "multiplicity": float(mult),
        "converged": bool(mf.converged),
        "stable": bool(stable),
        "stability_attempts": int(stability_attempts),
    }

def run_uccsd(mf):
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 200
    mycc.diis_space = 12
    mycc.verbose = 4

    ecc, t1, t2 = mycc.kernel()

    e_tot = mf.e_tot + ecc

    print("UCCSD converged:", mycc.converged)
    print("UCCSD corr energy:", ecc)
    print("UCCSD total energy:", e_tot)
    print("T1 alpha max:", abs(t1[0]).max())
    print("T1 beta  max:", abs(t1[1]).max())
    print("T2 aa max:", abs(t2[0]).max() if t2[0].size else 0.0)
    print("T2 ab max:", abs(t2[1]).max() if t2[1].size else 0.0)
    print("T2 bb max:", abs(t2[2]).max() if t2[2].size else 0.0)

    return e_tot, mycc

def run_fci(mol):
    rhf = scf.RHF(mol)
    rhf.conv_tol = SCF_CONV_TOL
    rhf.max_cycle = 200
    rhf.verbose = 0
    rhf.kernel()

    if not rhf.converged:
        rhf.max_cycle = 500
        rhf.kernel()

    mo = rhf.mo_coeff
    nmo = mo.shape[1]
    nelec = mol.nelec

    h1_mo = mo.T @ rhf.get_hcore() @ mo
    eri_mo = ao2mo.kernel(mol, mo)

    cisolver = fci.direct_spin1.FCI(mol)
    cisolver.conv_tol = FCI_CONV_TOL

    e_fci, ci_vec = cisolver.kernel(
        h1_mo,
        eri_mo,
        nmo,
        nelec,
        ecore=mol.energy_nuc(),
    )

    return {
        "energy": float(e_fci),
        "rhf_energy": float(rhf.e_tot),
        "rhf_converged": bool(rhf.converged),
    }


def calculate_pes():
    distances = np.arange(R_MIN, R_MAX + 0.5 * R_STEP, R_STEP)
    N_POINTS = len(distances)
    results = []

    for i, r in enumerate(distances, start=1):
        print()
        print(f"Point {i}/{N_POINTS}: R = {r:.6f} Angstrom")

        mol = build_h2(r)

        mf, uhf = run_uhf(mol)
        e_uccsd, _ = run_uccsd(mf)
        fci_result = run_fci(mol)

        e_uhf = uhf["energy"]
        e_fci = fci_result["energy"]
        e_rhf = fci_result["rhf_energy"]
        diff = e_uhf - e_fci

        row = {
            "R_angstrom": r,
            "E_RHF": e_rhf,
            "E_UHF": e_uhf,
            "E_UCCSD": e_uccsd,
            "E_FCI": e_fci,
            "UHF_minus_FCI": diff,
            "S2_UHF": uhf["s2"],
            "multiplicity_UHF": uhf["multiplicity"],
            "UHF_converged": uhf["converged"],
            "UHF_stable": uhf["stable"],
            "UHF_stability_attempts": uhf["stability_attempts"],
            "RHF_converged": fci_result["rhf_converged"],
        }

        results.append(row)

        print(
            f"E_RHF = {e_rhf: .12f}  "
            f"E_UHF = {e_uhf: .12f}  "
            f"E_UCCSD = {e_uccsd: .12f}  "
            f"E_FCI = {e_fci: .12f}"
        )
        print(
            f"UHF-FCI = {diff: .8e}  "
            f"<S^2> = {uhf['s2']: .8f}  "
            f"UHF stable = {uhf['stable']}"
        )

    return results


def save_csv(results):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved data to {CSV_FILE}")


def plot_pes(results):
    r = np.array([row["R_angstrom"] for row in results])
    e_rhf = np.array([row["E_RHF"] for row in results])
    e_uhf = np.array([row["E_UHF"] for row in results])
    #e_uccsd = np.array([row["E_UCCSD"] for row in results])
    e_fci = np.array([row["E_FCI"] for row in results])

    plt.figure()
    plt.plot(r, e_rhf, marker="^", markersize=3, label="RHF")
    plt.plot(r, e_uhf, marker="o", markersize=3, label="UHF, stability optimized")
    #plt.plot(r, e_uccsd, marker="x", markersize=3, label="UCCSD, stability optimized")
    plt.plot(r, e_fci, marker="s", markersize=3, label="FCI")
    plt.xlabel("H-H distance / Angstrom")
    plt.ylabel("Total energy / Hartree")
    plt.title("H2/STO-3G Potential Energy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ENERGY_PLOT, dpi=300)

    print(f"Saved PES plot to {ENERGY_PLOT}")

    if SHOW_PLOTS:
        plt.show()


def plot_energy_difference(results):
    r = np.array([row["R_angstrom"] for row in results])
    diff = np.array([row["UHF_minus_FCI"] for row in results])

    plt.figure()
    plt.plot(r, diff, marker="o", markersize=3)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("H-H distance / Angstrom")
    plt.ylabel("E_UHF - E_FCI / Hartree")
    plt.title("H2/STO-3G: UHF-FCI Energy Difference")
    plt.tight_layout()
    plt.savefig(DIFF_PLOT, dpi=300)

    print(f"Saved difference plot to {DIFF_PLOT}")

    if SHOW_PLOTS:
        plt.show()


def check_variational_bound(results, tol=1e-8):
    failures = []

    for row in results:
        if row["E_UHF"] + tol < row["E_FCI"]:
            failures.append(row)

    print()

    if failures:
        print("Warning: UHF energy below FCI beyond tolerance.")
        for row in failures:
            print(
                f"R = {row['R_angstrom']:.6f} Angstrom, "
                f"E_UHF - E_FCI = {row['UHF_minus_FCI']:.12e}, "
                f"UHF stable = {row['UHF_stable']}"
            )
    else:
        print("All points satisfy E_UHF >= E_FCI within tolerance.")


def check_stability(results):
    unstable = [row for row in results if not row["UHF_stable"]]

    print()

    if unstable:
        print("Warning: Some UHF solutions are still internally unstable.")
        for row in unstable:
            print(
                f"R = {row['R_angstrom']:.6f} Angstrom, "
                f"attempts = {row['UHF_stability_attempts']}"
            )
    else:
        print("All UHF solutions reported internally stable.")


def main():
    results = calculate_pes()

    save_csv(results)
    plot_pes(results)
    plot_energy_difference(results)

    check_variational_bound(results)
    check_stability(results)


if __name__ == "__main__":
    main()