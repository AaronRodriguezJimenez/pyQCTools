# plot_endo_vs_surf.py
# Recreate grouped bar plot comparing ENDO vs SURF ΔEbin values for X12Y12 clusters
# Requires: matplotlib, numpy, pandas (pandas only used if you want to inspect the dataframe)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# --- Data (ordered as requested) ---
clusters = [
    r'$\mathrm{B}_{12}\mathrm{N}_{12}$',
    r'$\mathrm{B}_{12}\mathrm{P}_{12}$',
    r'$\mathrm{B}_{12}\mathrm{As}_{12}$',
    r'$\mathrm{Ga}_{12}\mathrm{N}_{12}$',
    r'$\mathrm{Ga}_{12}\mathrm{As}_{12}$',
    r'$\mathrm{Zn}_{12}\mathrm{O}_{12}$',
    r'$\mathrm{Zn}_{12}\mathrm{S}_{12}$'
]

# HOMO-LUMO GAP data (pristine, endo) in eV
pristine_values = [11.0959, 7.9571, 7.3509, 8.0599, 6.0001, 9.0364, 8.8506]
endo_values = [6.9918, 4.4115, 4.8586, 3.8789, 6.1481, 4.0579, 6.4038]

# Optional: create a dataframe to inspect
df = pd.DataFrame({
    'Cluster': clusters,
    'gap (ENDO) [eV]': endo_values,
    'gap (PRIST) [eV]': pristine_values
})
print(df)

# --- Plot configuration ---
x = np.arange(len(clusters))
width = 0.38  # bar width

fig, ax = plt.subplots(figsize=(10, 6.0))

bars_endo = ax.bar(x - width/2, endo_values, width,
                   color='0.35', label='ENDO', edgecolor='black')
bars_surf = ax.bar(x + width/2, pristine_values, width,
                   color='0.75',label='PRIST', edgecolor='black')

# Annotate bars with values (formatting)
def annotate_bars(bars, values, fmt="{:.3f}"):
    for bar,level in zip(bars, values):
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(-1, np.sign(level)*15) if level > -1.570 else (-5, np.sign(level)*15),  # vertical offset
                    textcoords="offset points",
                    verticalalignment="bottom" if level < 0 else "top",
                    ha='center', fontsize=12)

annotate_bars(bars_endo, endo_values)
annotate_bars(bars_surf, pristine_values)

# Axes and labels
ax.set_xticks(x)
ax.set_xticklabels(clusters, fontsize=14)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
ax.tick_params(axis='both',labelsize=14)
ax.set_ylabel(r'Gap (eV)', fontsize=14)
ax.legend(frameon=True)
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)

# Improve layout
plt.tight_layout()

# Save files (change filenames if you like)
png_out = "HOMO-LUMO_endo_vs_surf.png"
svg_out = "HOMO-LUMO_endo_vs_surf.svg"
pdf_out = "HOMO-LUMO_endo_vs_surf.pdf"

#plt.savefig(png_out, dpi=300)
#plt.savefig(svg_out)
plt.savefig(pdf_out)
plt.show()

print(f"Saved: {png_out}, {svg_out}, {pdf_out}")

