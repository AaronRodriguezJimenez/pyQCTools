# ploting correlation diagnostics results for H4 models
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

TARGET_METRICS = [
    "fci_ci_entropy",
    "fci_n_unpaired", #Yamaguchi index
    "abs_e_corr_rhf",
]

METRIC_LABELS = {
    "fci_ci_entropy": r"FCI CI Entropy, $S_{\mathrm{CI}}$",
    "fci_n_unpaired": r"Effectively unpaired electrons (Yamaguchi index)",
    "abs_e_corr_rhf": r"RHF Correlation energy"
}

models = ["D4", "LIN4", "S4"]

def read_diagnostic_csv(path, model_name, r_column="R"):
    """
    Read one diagnostic output csv for a given H4 model
    :param path: str or path
    :param model_name: str: D4, LIN4, S4
    :param r_column: str, Name of the separation column
    :return: df: pandas.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path)

    if r_column not in df.columns:
        raise ValueError(
            f"File {path} does not contain R column '{r_column}'."
            f"Available columns are: {list(df.columns)}"
        )
    # If abs_e_corr_rhf was not stored, compute if from e_rhf and efci
    if "abs_e_corr_rhf" not in df.columns:
        if {"e_rhf", "e_fci"}.issubset(df.columns):
            df["abs_e_corr_rhf"] = df["e_rhf"] - df["e_fci"]
        else:
            raise ValueError(
                f"File {path} does not contain 'abs_e_corr_rhf',"
                "amd it also lacks e_rhf/e_fci columns."
            )

    missing =  [col for col in TARGET_METRICS if col not in df.columns]
    if missing:
        raise ValueError(
            f"File {path} is missing required columns: {missing}."
            f"Available columns are: {list(df.columns)}"
        )

    df = df.copy()
    df["model"] = model_name
    df = df.sort_values(by=[r_column])

    return df

def load_all_diagnostics(d4_csv, lin4_csv, s4_csv, r_column="R"):
    dfs = [
        read_diagnostic_csv(d4_csv, "D4", r_column=r_column),
        read_diagnostic_csv(lin4_csv, "LIN4", r_column=r_column),
        read_diagnostic_csv(s4_csv, "S4", r_column=r_column),
    ]

    return pd.concat(dfs, ignore_index=True)

def plot_metric_comparison(
        df,
        metric,
        r_column="R",
        savepath=None,
        show=False,
):
    """
    Plot one metric for a H4 model
    """
    fig, ax = plt.subplots(figsize=(7.0, 5))

    for model in models:
        sub = df[df["model"] == model].sort_values(r_column)

        if sub.empty:
            continue

        ax.plot(
            sub[r_column],
            sub[metric],
            marker="o",
            linewidth=2,
            label=f"{model}",
        )

    ax.set_xlabel(f"Separation coordinate, $R$ / Angstrom")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

def plot_combined_panel(
        df,
        metrics=TARGET_METRICS,
        r_column="R",
        savepath=None,
        show=False,
):
    """
    Make a three-panel comparative plot
    """
    n = len(metrics)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n,
        figsize=(6.0 * n, 4.8),
        sharex=False,
    )

    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model in ["D4", "LIN4", "S4"]:
            sub = df[df["model"] == model].sort_values(r_column)

            if sub.empty:
                continue

            ax.plot(
                sub[r_column],
                sub[metric],
                marker="o",
                linewidth=2,
                label=model,
            )

        ax.set_xlabel(r"Separation coordinate, $R$ / Å")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    axes[0].legend(frameon=False)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read H4 strong-correlation diagnostic CSV files and compare "
            "D4, LIN4, and S4 for FCI CI entropy, effectively unpaired "
            "electrons, and RHF correlation energy."
        )
    )

    parser.add_argument(
        "--d4",
        required=True,
        help="Path to D4 diagnostic CSV.",
    )

    parser.add_argument(
        "--lin4",
        required=True,
        help="Path to LIN4 diagnostic CSV.",
    )

    parser.add_argument(
        "--s4",
        required=True,
        help="Path to S4 diagnostic CSV.",
    )

    parser.add_argument(
        "--r-column",
        default="R",
        help="Name of the separation column. Default: R.",
    )

    parser.add_argument(
        "--outdir",
        default="h4_diagnostic_plots",
        help="Output directory for plots and combined CSV.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_all_diagnostics(
        d4_csv=args.d4,
        lin4_csv=args.lin4,
        s4_csv=args.s4,
        r_column=args.r_column,
    )

    combined_csv = outdir / "h4_D4_LIN4_S4_combined_qc_diagnostics.csv"
    df.to_csv(combined_csv, index=False)
    print(f"Combined diagnostics saved to: {combined_csv}")

    # Three-panel summary plot
    combined_plot = outdir / "h4_D4_LIN4_S4_qc_diagnostics_panel.png"
    plot_combined_panel(
        df,
        metrics=TARGET_METRICS,
        r_column=args.r_column,
        savepath=combined_plot,
        show=args.show,
    )
    print(f"Combined panel saved to: {combined_plot}")

    # Individual plots
    for metric in TARGET_METRICS:
        savepath = outdir / f"h4_D4_LIN4_S4_{metric}.png"
        plot_metric_comparison(
            df,
            metric,
            r_column=args.r_column,
            savepath=savepath,
            show=args.show,
        )
        print(f"{metric} plot saved to: {savepath}")


if __name__ == "__main__":
    main()
    #
    # Run with command line as:
    #  python plot_correlation_diagnostics.py \
    #   --d4 h4_D4_strong_correlation_diagnostics.csv \
    #   --lin4 h4_LIN4_strong_correlation_diagnostics.csv \
    #   --s4 h4_S4_strong_correlation_diagnostics.csv
