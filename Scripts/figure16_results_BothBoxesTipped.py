"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot ice results for both boxes tipped case.

Run from `Figures` directory: uv run Scripts/figure16_results_BothBoxesTipped.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure16_results_BothBoxesTipped"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" 
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

# Data file
nc_file = data_dir / "results_BothBoxesMYthenTipped.nc"

# Plot only years 1-20.
plot_years = 20

# Figure settings
EArc_color = "darkblue"
WArc_color = "darkorange"

figsize = (5, 4.2)
linewidth = 1.5
fontsize = 10
title_fontsize = fontsize
legend_fontsize = 8


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def format_time_axis(
    ax,
    xticks: np.ndarray,
    xlabels: list[str],
    plot_end: int,
    show_xlabel: bool = False,
) -> None:
    """Apply common time-axis formatting."""
    ax.grid(False)
    ax.set_xlim([0, plot_end - 1])
    ax.set_xticks(xticks)

    if show_xlabel:
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Year", fontsize=fontsize)
    else:
        ax.set_xticklabels([])


# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not nc_file.exists():
    raise FileNotFoundError(f"Could not find NetCDF file: {nc_file}")

with xr.open_dataset(nc_file) as ds:
    h_ice_EArc = np.asarray(ds["h_ice_EArc"].values).squeeze().astype(float)
    h_ice_WArc = np.asarray(ds["h_ice_WArc"].values).squeeze().astype(float)
    SIC_EArc = np.asarray(ds["SIC_EArc"].values).squeeze().astype(float)
    SIC_WArc = np.asarray(ds["SIC_WArc"].values).squeeze().astype(float)
    ice_type_EArc = np.asarray(ds["ice_type_EArc"].values).squeeze().astype(float)
    ice_type_WArc = np.asarray(ds["ice_type_WArc"].values).squeeze().astype(float)

    days_per_year = (
        int(np.asarray(ds["timestep_days_per_year"].values).squeeze())
        if "timestep_days_per_year" in ds
        else 365
    )
    n_years = (
        int(np.asarray(ds["timestep_n_years"].values).squeeze())
        if "timestep_n_years" in ds
        else len(h_ice_EArc) // days_per_year
    )

n_time = len(h_ice_EArc)
plot_years = min(plot_years, n_years)
plot_end = plot_years * days_per_year

xticks = np.arange(0, plot_end, days_per_year)
xlabels = [
    str(year) if i % 4 == 0 else ""
    for i, year in enumerate(range(1, plot_years + 1))
]


# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=figsize)
gs = GridSpec(
    nrows=3,
    ncols=1,
    figure=fig,
    height_ratios=[1.05, 1.05, 1.05],
    hspace=0.45,
)

fig.suptitle(
    "Both Boxes Tipped",
    fontsize=title_fontsize,
    fontweight="bold",
)

ax_h = fig.add_subplot(gs[0, 0])
ax_sic = fig.add_subplot(gs[1, 0])
ax_type = fig.add_subplot(gs[2, 0])

ax_h.plot(
    h_ice_EArc,
    label="East Arctic",
    color=EArc_color,
    linewidth=linewidth,
)
ax_h.plot(
    h_ice_WArc,
    label="West Arctic",
    color=WArc_color,
    linewidth=linewidth,
)
ax_h.set_title("(a) Ice Thickness [m]", fontsize=title_fontsize)
ax_h.set_ylim([-0.1, 4])
ax_h.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
format_time_axis(ax_h, xticks, xlabels, plot_end, show_xlabel=False)

ax_sic.plot(
    SIC_EArc,
    label="East Arctic",
    color=EArc_color,
    linewidth=linewidth,
)
ax_sic.plot(
    SIC_WArc,
    label="West Arctic",
    color=WArc_color,
    linewidth=linewidth,
)
ax_sic.set_title("(b) SIC", fontsize=title_fontsize)
ax_sic.set_ylim([-0.1, 1.1])
format_time_axis(ax_sic, xticks, xlabels, plot_end, show_xlabel=False)

ax_type.plot(
    ice_type_EArc,
    label="East Arctic",
    color=EArc_color,
    linewidth=linewidth,
)
ax_type.plot(
    ice_type_WArc,
    label="West Arctic",
    color=WArc_color,
    linewidth=linewidth,
)
ax_type.set_title("(c) Ice Type", fontsize=title_fontsize)
ax_type.set_ylim([0.9, 2.1])
ax_type.set_yticks([1, 2])
ax_type.set_yticklabels(["FY", "MY"])
format_time_axis(ax_type, xticks, xlabels, plot_end, show_xlabel=True)

for ax, label in zip([ax_h, ax_sic, ax_type], ["m", "SIC", "Ice type"]):
    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(label, fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.1, 0.5)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
