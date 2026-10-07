"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot ice type for the air temperature experiment.

Run from `Figures` directory: uv run Scripts/figure10_exptTatm_iceType.py
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
fig_name = "figure10_exptTatm_iceType"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_AirTemperature"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_expt_airTemp"

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
    r"East_mean(?P<E_mean>[-+]?\d+\.\d+)_amp(?P<E_amp>[-+]?\d+\.\d+)_"
    r"West_mean(?P<W_mean>[-+]?\d+\.\d+)_amp(?P<W_amp>[-+]?\d+\.\d+)\.nc$"
)

records = []
for nc_file in sorted(data_dir.glob(f"{nc_filename_base}_*.nc")):
    match = pattern.match(nc_file.name)
    if match is None:
        continue

    record = {key: float(value) for key, value in match.groupdict().items()}
    record["file"] = nc_file

    mean_offset_EArc = record["E_mean"] - box_paramsEArc.T_atm_mean
    mean_offset_WArc = record["W_mean"] - box_paramsWArc.T_atm_mean
    amp_offset_EArc = record["E_amp"] - box_paramsEArc.T_atm_amp
    amp_offset_WArc = record["W_amp"] - box_paramsWArc.T_atm_amp

    if (
        np.isclose(mean_offset_EArc, mean_offset_WArc)
        and np.isclose(amp_offset_EArc, amp_offset_WArc)
    ):
        records.append(record)

EArc_T_atm_means = np.array(sorted({record["E_mean"] for record in records}))
EArc_T_atm_amps = np.array(sorted({record["E_amp"] for record in records}))
WArc_T_atm_means = np.array(sorted({record["W_mean"] for record in records}))
WArc_T_atm_amps = np.array(sorted({record["W_amp"] for record in records}))

run_state_EArc = np.full((EArc_T_atm_means.size, EArc_T_atm_amps.size), np.nan)
run_state_WArc = np.full((WArc_T_atm_means.size, WArc_T_atm_amps.size), np.nan)

E_mean_to_ind = {value: i for i, value in enumerate(EArc_T_atm_means)}
E_amp_to_ind = {value: i for i, value in enumerate(EArc_T_atm_amps)}
W_mean_to_ind = {value: i for i, value in enumerate(WArc_T_atm_means)}
W_amp_to_ind = {value: i for i, value in enumerate(WArc_T_atm_amps)}

# -----------------------------------------------------------------------------
# Read NetCDF data and compute ice-type state
# -----------------------------------------------------------------------------
for record in records:
    ind_mean_EArc = E_mean_to_ind[record["E_mean"]]
    ind_amp_EArc = E_amp_to_ind[record["E_amp"]]
    ind_mean_WArc = W_mean_to_ind[record["W_mean"]]
    ind_amp_WArc = W_amp_to_ind[record["W_amp"]]

    with xr.open_dataset(record["file"]) as ds:
        ice_type_EArc = np.asarray(ds["ice_type_EArc"].values).squeeze().astype(float)
        ice_type_WArc = np.asarray(ds["ice_type_WArc"].values).squeeze().astype(float)

    ice_type_EArc = ice_type_EArc[-num_timesteps:]
    ice_type_WArc = ice_type_WArc[-num_timesteps:]

    sum_ice_type_EArc = np.sum(ice_type_EArc == 2)
    sum_ice_type_WArc = np.sum(ice_type_WArc == 2)

    if sum_ice_type_EArc == num_timesteps:
        run_state_EArc[ind_mean_EArc, ind_amp_EArc] = 3
    elif sum_ice_type_EArc == 0:
        run_state_EArc[ind_mean_EArc, ind_amp_EArc] = 1
    elif 0 < sum_ice_type_EArc < num_timesteps:
        run_state_EArc[ind_mean_EArc, ind_amp_EArc] = 2

    if sum_ice_type_WArc == num_timesteps:
        run_state_WArc[ind_mean_WArc, ind_amp_WArc] = 3
    elif sum_ice_type_WArc == 0:
        run_state_WArc[ind_mean_WArc, ind_amp_WArc] = 1
    elif 0 < sum_ice_type_WArc < num_timesteps:
        run_state_WArc[ind_mean_WArc, ind_amp_WArc] = 2

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
xticks_EArc = EArc_T_atm_amps
yticks_EArc = EArc_T_atm_means
xticks_WArc = WArc_T_atm_amps
yticks_WArc = WArc_T_atm_means

xticklabels_EArc = [
    str(round(x, 1)) if i % 2 == 0 else "" for i, x in enumerate(xticks_EArc)
]
yticklabels_EArc = [
    str(round(y, 1)) if i % 2 == 0 else "" for i, y in enumerate(yticks_EArc)
]
xticklabels_WArc = [
    str(round(x, 1)) if i % 2 == 0 else "" for i, x in enumerate(xticks_WArc)
]
yticklabels_WArc = [
    str(round(y, 1)) if i % 2 == 0 else "" for i, y in enumerate(yticks_WArc)
]

extent_EArc = [
    round(EArc_T_atm_amps[0] - 0.5 * (EArc_T_atm_amps[1] - EArc_T_atm_amps[0]), 1),
    round(EArc_T_atm_amps[-1] + 0.5 * (EArc_T_atm_amps[1] - EArc_T_atm_amps[0]), 1),
    round(EArc_T_atm_means[0] - 0.5 * (EArc_T_atm_means[1] - EArc_T_atm_means[0]), 1),
    round(EArc_T_atm_means[-1] + 0.5 * (EArc_T_atm_means[1] - EArc_T_atm_means[0]), 1),
]
extent_WArc = [
    round(WArc_T_atm_amps[0] - 0.5 * (WArc_T_atm_amps[1] - WArc_T_atm_amps[0]), 1),
    round(WArc_T_atm_amps[-1] + 0.5 * (WArc_T_atm_amps[1] - WArc_T_atm_amps[0]), 1),
    round(WArc_T_atm_means[0] - 0.5 * (WArc_T_atm_means[1] - WArc_T_atm_means[0]), 1),
    round(WArc_T_atm_means[-1] + 0.5 * (WArc_T_atm_means[1] - WArc_T_atm_means[0]), 1),
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
    extent=extent_EArc,
)
im_W_state = ax_W_state.imshow(
    run_state_WArc,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
)

ax_E_state.plot(
    box_paramsEArc.T_atm_amp,
    box_paramsEArc.T_atm_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)
ax_W_state.plot(
    box_paramsWArc.T_atm_amp,
    box_paramsWArc.T_atm_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)

ax_E_state.set_xticks(xticks_EArc)
ax_E_state.set_xticklabels(xticklabels_EArc)
ax_E_state.set_yticks(yticks_EArc)
ax_E_state.set_yticklabels(yticklabels_EArc)
ax_E_state.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)
ax_E_state.set_ylabel("Air Temperature Mean [°C]", fontsize=fontsize)
ax_E_state.set_title("(a) East Arctic", fontsize=title_fontsize)

ax_W_state.set_xticks(xticks_WArc)
ax_W_state.set_xticklabels(xticklabels_WArc)
ax_W_state.set_yticks(yticks_WArc)
ax_W_state.set_yticklabels(yticklabels_WArc)
ax_W_state.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)
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
    fontweight="bold"
)

fig.subplots_adjust(right=0.90)
pos_state = ax_W_state.get_position()
cbar_state_ax = fig.add_axes([0.92, pos_state.y0, 0.015, pos_state.height])
cbar_state = fig.colorbar(im_W_state, cax=cbar_state_ax, ticks=[1, 2, 3])
cbar_state.ax.set_yticklabels(["First-year", "Mixed", "Multi-year"])
cbar_state.ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
