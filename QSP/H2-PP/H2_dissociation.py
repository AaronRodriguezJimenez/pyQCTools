"""
 Integrals for H2 dissociation
"""
import os
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.lib import logger
from src.pyqctools.int_fcns import get_restricted_active_space_integrals
import matplotlib.pyplot as plt

d = 0.7414  # fixed H-H distance inside each H2 fragment, Angstrom
def make_h2_pes(r):
    """
    Linear H2 split into two H atoms:

        H -- r -- H

    r is the H-H separation between the two H atoms.
    """
    if r <= 0:
        raise ValueError("r_frag must be positive.")

    z = np.array([
        0.0,
        r,
    ])

    return "\n".join(f"H 0.0 0.0 {zi:.10f}" for zi in z)


def stable_opt_internal(mf, stability_cycles=10):
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)

    cyc = 0
    print("STABLE?", stable)

    while (not stable and cyc < stability_cycles):
        log.note("Try to optimize orbitals until stable, attempt %d" % cyc)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf.stability(return_status=True)
        print("STABLE?", stable)
        cyc += 1

    if not stable:
        log.note("Stability Opt failed after %d attempts" % cyc)

    return mf


def save_tensors(label, h0, h1, h2, folder="tensors"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = folder / f"{label}_tensors.npz"
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Tensors saved to {filename}")


def compute_integrals_for_separation(
    r_frag,
    basis="sto-3g",
    localized=False,
    outdir=".",
    stability_cycles=10,
):
    outdir = Path(outdir)
    tensor_dir = outdir / "tensors"
    log_dir = outdir / "logs"

    tensor_dir.mkdir(parents=True, exist_ok=True)
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
    mol.verbose = 4
    mol.output = str(log_dir / f"H_H_R_{r_frag}.out")
    mol.build()

    # Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # Stability optimization
    mf = stable_opt_internal(mf, stability_cycles=stability_cycles)

    # FCI energy
    cisolver = fci.FCI(mf)
    # cisolver.spin = 0
    cisolver.nroots = 1
    efci = cisolver.kernel()[0]

    # Full active space for H4/STO-3G:
    # 4 spatial orbitals, 4 electrons.
    ncore = 0
    ncas = mol.nao_nr() - ncore
    nelecas = mol.nelectron - 2 * ncore

    mc = mcscf.CASCI(mf, ncas, nelecas)
    print("NCORE IS:", mc.ncore)

    h0, h1, h2 = get_restricted_active_space_integrals(
        mol,
        mf,
        mc,
        localized=localized,
        include_all_noncore=False,
    )

    label = f"RHF_H2_R_{r_frag}".replace(".", "p")
    save_tensors(label, h0, h1, h2, folder=tensor_dir)

    return h0, h1, h2, mol, mf, mc, efci

def plot_energies(results, savepath=None, show=True):
    """
    Plot energy

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
    hf_energies = np.array([results[r]["hf_energy"] for r in r_values])
    fci_energies = np.array([results[r]["efci"] for r in r_values])

    plt.figure(figsize=(6, 4))
    plt.plot(r_values, hf_energies, marker="o", label="RHF")
    plt.plot(r_values, fci_energies, marker="o", label="FCI")
    plt.xlabel("H-H separation R / Angstrom")
    plt.ylabel("RHF energy / Hartree")
    plt.title("H-H2 Separation")
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
])

outdir = Path.cwd()
results = {}
for r_frag in r_fragment_values:
    h0, h1, h2, mol, mf, mc, efci = compute_integrals_for_separation(
        r_frag=r_frag,
        basis="sto-3g",
        localized=False,
        outdir=outdir,
        stability_cycles=10,
    )

    results[r_frag] = {
        "h0": h0,
        "h1": h1,
        "h2": h2,
        "hf_energy": mf.e_tot,
        "efci": efci,
    }

r_values, hf_energies = plot_energies(
    results,
    savepath=outdir / "hf_energy_vs_H2_separation.png",
)