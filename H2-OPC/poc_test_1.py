import numpy as np
import pandas as pd
from scipy.special import eval_legendre
from numpy.linalg import lstsq
from pyscf import gto, scf, mp
import matplotlib.pyplot as plt
#
# Initial test for para-ortho H2 conversion
# idea generated to test possible QC usage
# MINIMAL WORKFLOW
# - Build geometries for H2 near a small site model
# - Use PySCF to compute classical reference energies
# - Use Qiskit Nature to convert the electronic Hamiltonian to qubit Hamiltonian
# - Run a quantum algorithm on a simulator for selected grid points.
# - Build V(z,Theta)
# - Fit V0, V2, V4
# - Compare catalyst/site models using E_d, anisotropy, t_R and a sticking proxy

BOHR_TO_ANG = 0.529177
H2_BOND = 0.7414  # Angstrom

def h2_atoms_near_site(z=3.0, theta_deg=0.0):
    """
    H2 center of mass is placed at (0, 0, z).
    theta is angle between H-H axis and surface normal z.
    """
    theta = np.deg2rad(theta_deg)

    # H2 axis in x-z plane
    axis = np.array([np.sin(theta), 0.0, np.cos(theta)])
    r1 = np.array([0.0, 0.0, z]) + 0.5 * H2_BOND * axis
    r2 = np.array([0.0, 0.0, z]) - 0.5 * H2_BOND * axis

    return [
        ("H", tuple(r1)),
        ("H", tuple(r2)),
    ]

def water_site_atoms():
    """
    Very simple fixed water geometry, centered near origin.
    This is not a real surface, only a POC adsorption site.
    """
    return [
        ("O", (0.0000, 0.0000, 0.0000)),
        ("H", (0.9572, 0.0000, 0.0000)),
        ("H", (-0.2390, 0.9270, 0.0000)),
    ]

def atom_string(atoms):
    return "; ".join(
        f"{sym} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}"
        for sym, xyz in atoms
    )

def complex_geometry(z, theta):
    return atom_string(water_site_atoms() + h2_atoms_near_site(z, theta))

#
# WRITE XYZ files
#
def atoms_to_xyz(atoms, comment="Generated geometry"):
    """
    Convert a geometry into XYZ format.

    Accepts either:
      1. List format:
         [("O", (0.0, 0.0, 0.0)), ("H", (0.9, 0.0, 0.0))]
      2. PySCF-style string:
         "O 0.0 0.0 0.0; H 0.9 0.0 0.0"

    Returns:
      XYZ-format string.
    """

    if isinstance(atoms, str):
        parsed_atoms = []
        for item in atoms.replace("\n", ";").split(";"):
            item = item.strip()
            if not item:
                continue

            parts = item.split()
            if len(parts) != 4:
                raise ValueError(f"Could not parse atom entry: {item}")

            symbol = parts[0]
            x, y, z = map(float, parts[1:])
            parsed_atoms.append((symbol, (x, y, z)))

        atoms = parsed_atoms

    lines = [str(len(atoms)), comment]

    for symbol, coords in atoms:
        x, y, z = coords
        lines.append(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}")

    return "\n".join(lines)


def write_xyz(filename, atoms, comment="Generated geometry"):
    """
    Write an XYZ file to disk.
    """
    xyz_text = atoms_to_xyz(atoms, comment=comment)

    with open(filename, "w") as f:
        f.write(xyz_text + "\n")

    return filename

#- - - - PYSCF
# For physisorption, HF is not enough because dispersion/correlation matters.
# For the POC, we use MP2 (or CCSD if small enough).
def pyscf_mp2_energy(atom_spec, basis="sto-3g", charge=0, spin=0):
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)
    mp2_calc = mp.MP2(mf).run(verbose=0)

    return mf.e_tot + mp2_calc.e_corr

def interaction_energy_pyscf(z, theta, basis="sto-3g"):
    e_complex = pyscf_mp2_energy(complex_geometry(z, theta), basis=basis)
    e_site = pyscf_mp2_energy(atom_string(water_site_atoms()), basis=basis)
    e_h2 = pyscf_mp2_energy(atom_string(h2_atoms_near_site(20.0, theta)), basis=basis)
    return e_complex - e_site - e_h2

#
# Fit V0, V2, V4
#
def fit_legendre_for_each_z(df, l_values=(0, 2, 4)):
    out = []

    for z, sub in df.groupby("z_A"):
        theta = np.deg2rad(sub["theta_deg"].values)
        x = np.cos(theta)
        y = sub["E_int_meV"].values

        A = np.column_stack([eval_legendre(l, x) for l in l_values])
        coeffs, *_ = lstsq(A, y, rcond=None)

        row = {"z_A": z}
        for l, c in zip(l_values, coeffs):
            row[f"V{l}_meV"] = c
        out.append(row)

    return pd.DataFrame(out)

def plot_potential_results(
    df,
    fit_df=None,
    energy_col="E_int_meV",
    z_col="z_A",
    theta_col="theta_deg",
    save_prefix=None,
    show=True,
):
    """
    Plot physisorption-grid results.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns such as:
        z_A, theta_deg, E_int_meV

    fit_df : pandas.DataFrame or None
        Optional DataFrame from fit_legendre_for_each_z().
        Expected columns:
        z_A, V0_meV, V2_meV, V4_meV, ...

    energy_col : str
        Column containing interaction energy.

    save_prefix : str or None
        If provided, saves figures as:
        {save_prefix}_energy_vs_z.png
        {save_prefix}_legendre_coefficients.png

    show : bool
        Whether to call plt.show().

    Returns
    -------
    figs : list
        List of matplotlib Figure objects.
    """

    figs = []

    required = {z_col, theta_col, energy_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # Plot 1: interaction energy vs z for each orientation
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for theta, sub in df.groupby(theta_col):
        sub = sub.sort_values(z_col)
        ax1.plot(
            sub[z_col],
            sub[energy_col],
            marker="o",
            label=fr"$\Theta={theta}^\circ$",
        )

    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.set_xlabel(r"$z$ / Å")
    ax1.set_ylabel("Interaction energy / meV")
    ax1.set_title(r"Computed physisorption potential $V(z,\Theta)$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    figs.append(fig1)

    if save_prefix is not None:
        fig1.savefig(f"{save_prefix}_energy_vs_z.png", dpi=300, bbox_inches="tight")

    # Plot 2: Legendre coefficients vs z
    if fit_df is not None:
        if z_col not in fit_df.columns:
            raise ValueError(f"fit_df must contain column: {z_col}")

        coeff_cols = [
            col for col in fit_df.columns
            if col.startswith("V") and col.endswith("_meV")
        ]

        if not coeff_cols:
            raise ValueError("fit_df has no coefficient columns like V0_meV, V2_meV.")

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        fit_sorted = fit_df.sort_values(z_col)

        for col in coeff_cols:
            ax2.plot(
                fit_sorted[z_col],
                fit_sorted[col],
                marker="o",
                label=col.replace("_meV", ""),
            )

        ax2.axhline(0.0, linestyle="--", linewidth=1)
        ax2.set_xlabel(r"$z$ / Å")
        ax2.set_ylabel("Legendre coefficient / meV")
        ax2.set_title(r"Legendre expansion coefficients $V_\ell(z)$")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        figs.append(fig2)

        if save_prefix is not None:
            fig2.savefig(
                f"{save_prefix}_legendre_coefficients.png",
                dpi=300,
                bbox_inches="tight",
            )

    if show:
        plt.show()

    return figs

###########################################
# CALCULATIONS
# - - - Tiny Grid
z_values = np.arange(2.5, 5.1, 0.1) #[2.5, 3.0, 3.5, 4.0, 5.0]
theta_values = [0, 45, 90]

rows = []

for z in z_values:
    for theta in theta_values:
        e_int = interaction_energy_pyscf(z, theta, basis="sto-3g")
        rows.append({
            "z_A": z,
            "theta_deg": theta,
            "E_int_Hartree": e_int,
            "E_int_meV": e_int * 27.211386 * 1000,
        })
        geom = complex_geometry(z, theta)
        print(f"Geometry: {z}, {theta}")
        write_xyz(f"geom_{z}_{theta}.xyz", geom, comment="Generated geometry")

df = pd.DataFrame(rows)
print("Results data frame (df)")
print(df)

# - Fit for V0, V2, V4
fit_df = fit_legendre_for_each_z(df)
print("Results legendre-fitted data (fit_df)")
print(fit_df)


figs = plot_potential_results(
    df,
    fit_df=fit_df,
    save_prefix="h2_water_poc",
)

#
# Compute residence-time proxy
kB_meV_per_K = 0.08617333262
t0 = 1e-11  # seconds
def residence_time(E_d_meV, T_K):
    return t0 * np.exp(E_d_meV / (kB_meV_per_K * T_K))

E_d_meV = -fit_df["V0_meV"].min()

print("Residence Time (s):")
for T in [10, 20, 30]:
    print(T, residence_time(E_d_meV, T))


St = 0.1          # dimensionless sticking coefficient
tau = 1000.0      # s, microscopic o-p conversion time on ASW
tau_inv = 1.0 / tau

temperature = 30.0  # K

# Residence time from your PES-derived adsorption energy
t_R = residence_time(E_d_meV, temperature)
t_R_inv = 1.0 / t_R

# Probability that a physisorbed molecule converts before desorbing
p_convert_given_adsorbed = tau_inv / (t_R_inv + tau_inv)

# Probability per collision, including sticking
p_convert_per_collision = St * p_convert_given_adsorbed

print(f"T = {temperature} K")
print(f"E_d = {E_d_meV:.4f} meV")
print(f"t_R = {t_R:.4e} s")
print(f"tau = {tau:.4e} s")
print(f"P(convert | adsorbed) = {p_convert_given_adsorbed:.4e}")
print(f"P(convert per collision) = {p_convert_per_collision:.4e}")

# Paper-like gamma range; gamma is a collision rate in s^-1
gammas = np.logspace(-7, 0, 300)

# Overall conversion rate, units s^-1
conversion_rates = St * gammas * p_convert_given_adsorbed

plt.figure(figsize=(7, 5))
plt.plot(gammas, conversion_rates, label="MP2-derived residence time")
plt.xscale("log")
plt.yscale("log")
plt.title(f"Overall o-p conversion rate, T={temperature} K")
plt.xlabel(r"Collision rate $\gamma$ / s$^{-1}$")
plt.ylabel(r"Overall conversion rate $P$ / s$^{-1}$")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("h2_water_poc_conversion_rate.png",dpi=300, bbox_inches="tight")
plt.show()
