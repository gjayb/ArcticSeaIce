"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot mean ice thickness, SIC, and ice volume for last 10 years 
of model run for the air temperature experiment.

Run from `Figures` directory: uv run Scripts/figure09_exptTatm_ice_thickness_SIC_volume.py
"""

from pathlib import Path
import re
import sys

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arctic_ice.arctic_ice_model_args import (  # noqa: E402
    ArcticIceModelBoxDimensions,
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic,
)

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure09_exptTatm_ice_thickness_SIC_volume"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_AirTemperature"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_expt_airTemp"

n_years = 50
days_per_year = 365
n_days = n_years * days_per_year
last_10_years = 365 * 10

# Figure settings
figsize = (8.5, 8.0)
fontsize = 10
title_fontsize = fontsize+2
default_shape = "*"
default_color = [0.5, 0.5, 0.5]
default_markersize = 10
default_markeredgecolor = "k"

# Model parameters
box_dims = ArcticIceModelBoxDimensions()
box_paramsEArc = ArcticIceModelParametersEastArctic()
box_paramsWArc = ArcticIceModelParametersWestArctic()

# -----------------------------------------------------------------------------
# Get "Both Boxes" experiment files
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

mean_h_ice_EArc = np.full((EArc_T_atm_means.size, EArc_T_atm_amps.size), np.nan)
mean_h_ice_WArc = np.full((WArc_T_atm_means.size, WArc_T_atm_amps.size), np.nan)
mean_SIC_EArc = np.full_like(mean_h_ice_EArc, np.nan)
mean_SIC_WArc = np.full_like(mean_h_ice_WArc, np.nan)
mean_ice_volume_EArc = np.full_like(mean_h_ice_EArc, np.nan)
mean_ice_volume_WArc = np.full_like(mean_h_ice_WArc, np.nan)

E_mean_to_ind = {value: i for i, value in enumerate(EArc_T_atm_means)}
E_amp_to_ind = {value: i for i, value in enumerate(EArc_T_atm_amps)}
W_mean_to_ind = {value: i for i, value in enumerate(WArc_T_atm_means)}
W_amp_to_ind = {value: i for i, value in enumerate(WArc_T_atm_amps)}

# -----------------------------------------------------------------------------
# Read NetCDF data and compute metrics
# -----------------------------------------------------------------------------
for record in records:
    ind_mean_EArc = E_mean_to_ind[record["E_mean"]]
    ind_amp_EArc = E_amp_to_ind[record["E_amp"]]
    ind_mean_WArc = W_mean_to_ind[record["W_mean"]]
    ind_amp_WArc = W_amp_to_ind[record["W_amp"]]

    with xr.open_dataset(record["file"]) as ds:
        h_ice_EArc = np.asarray(ds["h_ice_EArc"].values).squeeze().astype(float)
        h_ice_WArc = np.asarray(ds["h_ice_WArc"].values).squeeze().astype(float)
        SIC_EArc = np.asarray(ds["SIC_EArc"].values).squeeze().astype(float)
        SIC_WArc = np.asarray(ds["SIC_WArc"].values).squeeze().astype(float)

    h_ice_EArc = h_ice_EArc[:n_days]
    h_ice_WArc = h_ice_WArc[:n_days]
    SIC_EArc = SIC_EArc[:n_days]
    SIC_WArc = SIC_WArc[:n_days]

    ice_volume_EArc = SIC_EArc * h_ice_EArc * box_dims.area_EArc
    ice_volume_WArc = SIC_WArc * h_ice_WArc * box_dims.area_WArc

    mean_h_ice_EArc[ind_mean_EArc, ind_amp_EArc] = np.nanmean(
        h_ice_EArc[-last_10_years:]
    )
    mean_h_ice_WArc[ind_mean_WArc, ind_amp_WArc] = np.nanmean(
        h_ice_WArc[-last_10_years:]
    )
    mean_SIC_EArc[ind_mean_EArc, ind_amp_EArc] = np.nanmean(
        SIC_EArc[-last_10_years:]
    )
    mean_SIC_WArc[ind_mean_WArc, ind_amp_WArc] = np.nanmean(
        SIC_WArc[-last_10_years:]
    )
    mean_ice_volume_EArc[ind_mean_EArc, ind_amp_EArc] = np.nanmean(
        ice_volume_EArc[-last_10_years:]
    )
    mean_ice_volume_WArc[ind_mean_WArc, ind_amp_WArc] = np.nanmean(
        ice_volume_WArc[-last_10_years:]
    )

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

hice_cmin = 0
hice_cmax = 6
hice_ticks = np.linspace(hice_cmin, hice_cmax, 7)

sic_cmin = 0.6
sic_cmax = 1.0
sic_ticks = np.linspace(sic_cmin, sic_cmax, 5)

volume_cmin = 0e13
volume_cmax = 6e13
volume_ticks = np.linspace(volume_cmin, volume_cmax, 7)

# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    hspace=0.55,
    wspace=0.25,
)

ax_E_hice = fig.add_subplot(gs[0, 0])
ax_W_hice = fig.add_subplot(gs[0, 1])
ax_E_sic = fig.add_subplot(gs[1, 0])
ax_W_sic = fig.add_subplot(gs[1, 1])
ax_E_volume = fig.add_subplot(gs[2, 0])
ax_W_volume = fig.add_subplot(gs[2, 1])

im_E_hice = ax_E_hice.imshow(
    mean_h_ice_EArc,
    cmap=cmocean.cm.dense,
    aspect="auto",
    origin="lower",
    extent=extent_EArc,
    vmin=hice_cmin,
    vmax=hice_cmax,
)
im_W_hice = ax_W_hice.imshow(
    mean_h_ice_WArc,
    cmap=cmocean.cm.dense,
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
    vmin=hice_cmin,
    vmax=hice_cmax,
)

im_E_sic = ax_E_sic.imshow(
    mean_SIC_EArc,
    cmap=cmocean.cm.amp_r,
    aspect="auto",
    origin="lower",
    extent=extent_EArc,
    vmin=sic_cmin,
    vmax=sic_cmax,
)
im_W_sic = ax_W_sic.imshow(
    mean_SIC_WArc,
    cmap=cmocean.cm.amp_r,
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
    vmin=sic_cmin,
    vmax=sic_cmax,
)

im_E_volume = ax_E_volume.imshow(
    mean_ice_volume_EArc,
    cmap=cmocean.cm.tempo,
    aspect="auto",
    origin="lower",
    extent=extent_EArc,
    vmin=volume_cmin,
    vmax=volume_cmax,
)
im_W_volume = ax_W_volume.imshow(
    mean_ice_volume_WArc,
    cmap=cmocean.cm.tempo,
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
    vmin=volume_cmin,
    vmax=volume_cmax,
)

for ax in [ax_E_hice, ax_E_sic, ax_E_volume]:
    ax.plot(
        box_paramsEArc.T_atm_amp,
        box_paramsEArc.T_atm_mean,
        default_shape,
        c=default_color,
        markersize=default_markersize,
        markeredgecolor=default_markeredgecolor,
    )

for ax in [ax_W_hice, ax_W_sic, ax_W_volume]:
    ax.plot(
        box_paramsWArc.T_atm_amp,
        box_paramsWArc.T_atm_mean,
        default_shape,
        c=default_color,
        markersize=default_markersize,
        markeredgecolor=default_markeredgecolor,
    )

for ax in [ax_E_hice, ax_E_sic, ax_E_volume]:
    ax.set_xticks(xticks_EArc)
    ax.set_xticklabels(xticklabels_EArc)
    ax.set_yticks(yticks_EArc)
    ax.set_yticklabels(yticklabels_EArc)
    ax.set_ylabel("Air Temperature Mean [°C]", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)
    if ax == ax_E_volume:
        ax.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)

for ax in [ax_W_hice, ax_W_sic, ax_W_volume]:
    ax.set_xticks(xticks_WArc)
    ax.set_xticklabels(xticklabels_WArc)
    ax.set_yticks(yticks_WArc)
    ax.set_yticklabels(yticklabels_WArc)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)
    if ax == ax_W_volume:
        ax.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)

ax_E_hice.set_title("(a) East Arctic", fontsize=title_fontsize)
ax_W_hice.set_title("(b) West Arctic", fontsize=title_fontsize)
ax_E_sic.set_title("(c) East Arctic", fontsize=title_fontsize)
ax_W_sic.set_title("(d) West Arctic", fontsize=title_fontsize)
ax_E_volume.set_title("(e) East Arctic", fontsize=title_fontsize)
ax_W_volume.set_title("(f) West Arctic", fontsize=title_fontsize)

fig.text(
    0.5,
    0.93,
    "Ice Thickness [m]",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight='bold'
)
fig.text(
    0.5,
    0.64,
    "Sea Ice Concentration",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight='bold'
)
fig.text(
    0.5,
    0.34,
    "Ice Volume [m³]",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight='bold'
)

# Add one colorbar for each row.
fig.subplots_adjust(right=0.90)

pos_hice = ax_W_hice.get_position()
cbar_hice_ax = fig.add_axes([0.92, pos_hice.y0, 0.015, pos_hice.height])
cbar_hice = fig.colorbar(im_W_hice, cax=cbar_hice_ax)
cbar_hice.set_ticks(hice_ticks)
cbar_hice.ax.tick_params(labelsize=fontsize)

pos_sic = ax_W_sic.get_position()
cbar_sic_ax = fig.add_axes([0.92, pos_sic.y0, 0.015, pos_sic.height])
cbar_sic = fig.colorbar(im_W_sic, cax=cbar_sic_ax)
cbar_sic.set_ticks(sic_ticks)
cbar_sic.ax.tick_params(labelsize=fontsize)

pos_volume = ax_W_volume.get_position()
cbar_volume_ax = fig.add_axes([0.92, pos_volume.y0, 0.015, pos_volume.height])
cbar_volume = fig.colorbar(im_W_volume, cax=cbar_volume_ax)
cbar_volume.set_ticks(volume_ticks)
cbar_volume.ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
