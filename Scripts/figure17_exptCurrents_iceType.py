"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot ice type for the the currents experiment.

Run from `Figures` directory: uv run Scripts/figure17_exptCurrents_iceType.py
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

from arctic_ice.arctic_ice_model_args import ArcticIceModelParameters


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure17_exptCurrents_iceType"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_Currents"
default_ts_dir = data_dir / "Default_TS"
counterfactual_ts_dir = data_dir / "Counterfactual_TS"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_expt_currents"

days_per_year = 365
num_years = 11
num_timesteps = num_years * days_per_year

# Figure settings
figsize = (7.5, 5.3)
fontsize = 10
title_fontsize = fontsize + 2
default_shape = "*"
default_color = [0.5, 0.5, 0.5]
default_markersize = 10
default_markeredgecolor = "k"

# Model parameters
box_params = ArcticIceModelParameters()

# Axis labels
xlabel = "North Atlantic Current [Sv]"
ylabel = "North Pacific Current [Sv]"


# -----------------------------------------------------------------------------
# Read experiment data and compute ice-type state
# -----------------------------------------------------------------------------
pattern = re.compile(
    rf"{nc_filename_base}_"
    r"NAtl(?P<NAtl>[-+]?\d+(?:\.\d+)?)Sv_(?P<NAtl_TS>[^_]+)_"
    r"NPac(?P<NPac>[-+]?\d+(?:\.\d+)?)Sv_(?P<NPac_TS>[^.]+)\.nc$"
)


def compute_ice_state(ice_type: np.ndarray) -> float:
    """Classify ice type over the final 11 years: FY=1, Mixed=2, MY=3."""
    ice_type = np.asarray(ice_type, dtype=float).squeeze()
    ice_type = ice_type[-num_timesteps:]

    sum_multi_year = np.sum(ice_type == 2)

    if sum_multi_year == num_timesteps:
        return 3
    if sum_multi_year == 0:
        return 1
    return 2


def read_case(case_dir: Path) -> dict[str, np.ndarray]:
    """Read one currents-experiment case directory."""
    if not case_dir.exists():
        raise FileNotFoundError(f"Could not find experiment data directory: {case_dir}")

    records = []
    for nc_file in sorted(case_dir.glob(f"{nc_filename_base}_*.nc")):
        match = pattern.match(nc_file.name)
        if match is None:
            continue

        record = {
            "NAtl": float(match.group("NAtl")),
            "NPac": float(match.group("NPac")),
            "NAtl_TS": match.group("NAtl_TS"),
            "NPac_TS": match.group("NPac_TS"),
            "file": nc_file,
        }
        records.append(record)

    if not records:
        raise FileNotFoundError(
            "No matching NetCDF files found in "
            f"{case_dir}. Expected files like "
            f"{nc_filename_base}_NAtl3.5Sv_defaultTS_NPac3.0Sv_defaultTS.nc"
        )

    current_NAtl_vals = np.array(sorted({record["NAtl"] for record in records}))
    current_NPac_vals = np.array(sorted({record["NPac"] for record in records}))

    run_state_EArc = np.full((current_NAtl_vals.size, current_NPac_vals.size), np.nan)
    run_state_WArc = np.full_like(run_state_EArc, np.nan)

    NAtl_to_ind = {value: i for i, value in enumerate(current_NAtl_vals)}
    NPac_to_ind = {value: i for i, value in enumerate(current_NPac_vals)}

    for record in records:
        ind_NAtl = NAtl_to_ind[record["NAtl"]]
        ind_NPac = NPac_to_ind[record["NPac"]]

        with xr.open_dataset(record["file"]) as ds:
            ice_type_EArc = np.asarray(ds["ice_type_EArc"].values).squeeze().astype(float)
            ice_type_WArc = np.asarray(ds["ice_type_WArc"].values).squeeze().astype(float)

        run_state_EArc[ind_NAtl, ind_NPac] = compute_ice_state(ice_type_EArc)
        run_state_WArc[ind_NAtl, ind_NPac] = compute_ice_state(ice_type_WArc)

    dx = np.nanmedian(np.diff(current_NAtl_vals)) if current_NAtl_vals.size > 1 else 0.5
    dy = np.nanmedian(np.diff(current_NPac_vals)) if current_NPac_vals.size > 1 else 0.5

    extent = [
        current_NAtl_vals[0] - 0.5 * dx,
        current_NAtl_vals[-1] + 0.5 * dx,
        current_NPac_vals[0] - 0.5 * dy,
        current_NPac_vals[-1] + 0.5 * dy,
    ]

    return {
        "current_NAtl_vals": current_NAtl_vals,
        "current_NPac_vals": current_NPac_vals,
        "extent": extent,
        "run_state_EArc": run_state_EArc,
        "run_state_WArc": run_state_WArc,
    }


default_ts = read_case(default_ts_dir)
counterfactual_ts = read_case(counterfactual_ts_dir)


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
xticks_default = default_ts["current_NAtl_vals"]
yticks_default = default_ts["current_NPac_vals"]

xticks_counter = counterfactual_ts["current_NAtl_vals"]
yticks_counter = counterfactual_ts["current_NPac_vals"]

xticklabels_default = [
    str(round(x, 1)) if i % 2 == 0 else ""
    for i, x in enumerate(xticks_default)
]
yticklabels_default = [
    str(round(y, 1)) if i % 2 == 0 else ""
    for i, y in enumerate(yticks_default)
]

xticklabels_counter = [
    str(round(x, 1)) if i % 2 == 0 else ""
    for i, x in enumerate(xticks_counter)
]
yticklabels_counter = [
    str(round(y, 1)) if i % 2 == 0 else ""
    for i, y in enumerate(yticks_counter)
]

boundaries = [0.5, 1.5, 2.5, 3.5]
viridis = plt.get_cmap("viridis", 3)
cmap = ListedColormap([viridis(i) for i in range(3)])
norm = BoundaryNorm(boundaries, ncolors=cmap.N)


# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    hspace=0.55,
    wspace=0.25,
)

ax_E_default = fig.add_subplot(gs[0, 0])
ax_W_default = fig.add_subplot(gs[0, 1])
ax_E_counter = fig.add_subplot(gs[1, 0])
ax_W_counter = fig.add_subplot(gs[1, 1])

im_E_default = ax_E_default.imshow(
    default_ts["run_state_EArc"].T,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=default_ts["extent"],
)
im_W_default = ax_W_default.imshow(
    default_ts["run_state_WArc"].T,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=default_ts["extent"],
)
im_E_counter = ax_E_counter.imshow(
    counterfactual_ts["run_state_EArc"].T,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=counterfactual_ts["extent"],
)
im_W_counter = ax_W_counter.imshow(
    counterfactual_ts["run_state_WArc"].T,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=counterfactual_ts["extent"],
)

for ax in [ax_E_default, ax_W_default, ax_E_counter, ax_W_counter]:
    ax.plot(
        box_params.current_NAtl / 1e6,
        box_params.current_NPac / 1e6,
        default_shape,
        c=default_color,
        markersize=default_markersize,
        markeredgecolor=default_markeredgecolor,
    )

for ax in [ax_E_default, ax_W_default]:
    ax.set_xticks(xticks_default)
    ax.set_xticklabels(xticklabels_default)
    ax.set_yticks(yticks_default)
    ax.set_yticklabels(yticklabels_default)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)

for ax in [ax_E_counter, ax_W_counter]:
    ax.set_xticks(xticks_counter)
    ax.set_xticklabels(xticklabels_counter)
    ax.set_yticks(yticks_counter)
    ax.set_yticklabels(yticklabels_counter)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)

for ax in [ax_E_default, ax_E_counter]:
    ax.set_ylabel(ylabel, fontsize=fontsize)

ax_E_counter.set_xlabel(xlabel, fontsize=fontsize)
ax_W_counter.set_xlabel(xlabel, fontsize=fontsize)

ax_E_default.set_title("(a) East Arctic", fontsize=title_fontsize)
ax_W_default.set_title("(b) West Arctic", fontsize=title_fontsize)
ax_E_counter.set_title("(c) East Arctic", fontsize=title_fontsize)
ax_W_counter.set_title("(d) West Arctic", fontsize=title_fontsize)

fig.text(
    0.5,
    0.96,
    "Ice Type for default T/S",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight="bold",
)
fig.text(
    0.5,
    0.49,
    "Ice Type for counterfactual T/S",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight="bold",
)

fig.subplots_adjust(right=0.90)
pos_top = ax_W_default.get_position()
pos_bottom = ax_W_counter.get_position()
cbar_ax = fig.add_axes([0.93, pos_bottom.y0 + (pos_top.y1 - pos_bottom.y0)/4, 0.015, (pos_top.y1 - pos_bottom.y0)/2])
cbar = fig.colorbar(im_W_default, cax=cbar_ax, ticks=[1, 2, 3])
cbar.ax.set_yticklabels(["First-year", "Mixed", "Multi-year"])
cbar.ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
