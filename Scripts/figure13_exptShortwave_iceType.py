"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot ice type for the shortwave flux experiment.

Run from `Figures` directory: uv run Scripts/figure13_exptShortwave_iceType.py
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

from arctic_ice.arctic_ice_model_args import (
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic,
)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure13_exptShortwave_iceType"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_ShortwaveFlux"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_experiment_shortwave"

days_per_year = 365
num_years = 11
num_timesteps = num_years * days_per_year

# Figure settings
figsize = (8.5, 2.5)
fontsize = 10
title_fontsize = fontsize + 2
default_shape = "*"
default_color = [0.5, 0.5, 0.5]
default_markersize = 10
default_markeredgecolor = "k"

# Model parameters
box_paramsEArc = ArcticIceModelParametersEastArctic()
box_paramsWArc = ArcticIceModelParametersWestArctic()


# -----------------------------------------------------------------------------
# Experiment data files
# -----------------------------------------------------------------------------
if not data_dir.exists():
    raise FileNotFoundError(f"Could not find experiment data directory: {data_dir}")

pattern = re.compile(
    rf"{nc_filename_base}_"
    r"mean(?P<mean>[-+]?\d+)"
    r"_amp(?P<amp>[-+]?\d+)\.nc$"
)

records = []
for nc_file in sorted(data_dir.glob(f"{nc_filename_base}_*.nc")):
    match = pattern.match(nc_file.name)
    if match is None:
        continue

    record = {key: int(value) for key, value in match.groupdict().items()}
    record["file"] = nc_file
    records.append(record)

if not records:
    raise FileNotFoundError(f"No matching NetCDF files found in {data_dir}")

shortwave_means = np.array(sorted({record["mean"] for record in records}))
shortwave_amps = np.array(sorted({record["amp"] for record in records}))

run_state_EArc = np.full((shortwave_means.size, shortwave_amps.size), np.nan)
run_state_WArc = np.full_like(run_state_EArc, np.nan)

mean_to_ind = {value: i for i, value in enumerate(shortwave_means)}
amp_to_ind = {value: i for i, value in enumerate(shortwave_amps)}


# -----------------------------------------------------------------------------
# Read NetCDF data and compute ice-type state
# -----------------------------------------------------------------------------
for record in records:
    ind_mean = mean_to_ind[record["mean"]]
    ind_amp = amp_to_ind[record["amp"]]

    with xr.open_dataset(record["file"]) as ds:
        ice_type_EArc = np.asarray(ds["ice_type_EArc"].values).squeeze().astype(float)
        ice_type_WArc = np.asarray(ds["ice_type_WArc"].values).squeeze().astype(float)

    ice_type_EArc = ice_type_EArc[-num_timesteps:]
    ice_type_WArc = ice_type_WArc[-num_timesteps:]

    sum_ice_type_EArc = np.sum(ice_type_EArc == 2)
    sum_ice_type_WArc = np.sum(ice_type_WArc == 2)

    if sum_ice_type_EArc == num_timesteps:
        run_state_EArc[ind_mean, ind_amp] = 3
    elif sum_ice_type_EArc == 0:
        run_state_EArc[ind_mean, ind_amp] = 1
    elif 0 < sum_ice_type_EArc < num_timesteps:
        run_state_EArc[ind_mean, ind_amp] = 2

    if sum_ice_type_WArc == num_timesteps:
        run_state_WArc[ind_mean, ind_amp] = 3
    elif sum_ice_type_WArc == 0:
        run_state_WArc[ind_mean, ind_amp] = 1
    elif 0 < sum_ice_type_WArc < num_timesteps:
        run_state_WArc[ind_mean, ind_amp] = 2


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
xticks = shortwave_amps
yticks = shortwave_means
xticklabels = [str(x) if i % 10 == 0 else " " for i, x in enumerate(xticks)]
yticklabels = [str(y) if i % 10 == 0 else " " for i, y in enumerate(yticks)]

fig_extent = [
    shortwave_amps[0] - 0.5 * (shortwave_amps[1] - shortwave_amps[0]),
    shortwave_amps[-1] + 0.5 * (shortwave_amps[1] - shortwave_amps[0]),
    shortwave_means[0] - 0.5 * (shortwave_means[1] - shortwave_means[0]),
    shortwave_means[-1] + 0.5 * (shortwave_means[1] - shortwave_means[0]),
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
    nrows=1,
    ncols=2,
    hspace=0.55,
    wspace=0.25,
)

ax_E_state = fig.add_subplot(gs[0, 0])
ax_W_state = fig.add_subplot(gs[0, 1])

im_E_state = ax_E_state.imshow(
    run_state_EArc,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=fig_extent,
)
im_W_state = ax_W_state.imshow(
    run_state_WArc,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=fig_extent,
)

ax_E_state.plot(
    box_paramsEArc.shortwave_amp,
    box_paramsEArc.shortwave_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)
ax_W_state.plot(
    box_paramsWArc.shortwave_amp,
    box_paramsWArc.shortwave_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)

ax_E_state.set_xticks(xticks)
ax_E_state.set_xticklabels(xticklabels)
ax_E_state.set_yticks(yticks)
ax_E_state.set_yticklabels(yticklabels)
ax_E_state.set_xlabel("Shortwave Amplitude [W m$^{-2}$]", fontsize=fontsize)
ax_E_state.set_ylabel("Shortwave Mean [W m$^{-2}$]", fontsize=fontsize)
ax_E_state.set_title("(a) East Arctic", fontsize=title_fontsize)

ax_W_state.set_xticks(xticks)
ax_W_state.set_xticklabels(xticklabels)
ax_W_state.set_yticks(yticks)
ax_W_state.set_yticklabels(yticklabels)
ax_W_state.set_xlabel("Shortwave Amplitude [W m$^{-2}$]", fontsize=fontsize)
ax_W_state.set_title("(b) West Arctic", fontsize=title_fontsize)

for ax in [ax_E_state, ax_W_state]:
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)

fig.text(
    0.5,
    1.03,
    "Ice Type",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight="bold",
)

# Add colorbar.
fig.subplots_adjust(right=0.90)

pos_state = ax_W_state.get_position()
cbar_state_ax = fig.add_axes([0.92, pos_state.y0, 0.015, pos_state.height])
cbar_state = fig.colorbar(im_W_state, cax=cbar_state_ax, ticks=[1, 2, 3])
cbar_state.ax.set_yticklabels(["First-year", "Mixed", "Multi-year"])
cbar_state.ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
