"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot number of days below freezing and days between ice volume
min and max for the air temperature experiment.

Run from `Figures` directory: uv run Scripts/figure08_exptTatm_freezingDays_iceDays.py
"""

from pathlib import Path
import re
import sys

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from arctic_ice.arctic_ice_model_args import (
    ArcticIceModelBoxDimensions,
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic,
)

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure08_exptTatm_freezingDays_iceDays"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_AirTemperature"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_expt_airTemp"

n_years = 50
days_per_year = 365
n_days = n_years * days_per_year
num_years_for_mean = 10
T_freezing = -1.85

# Figure settings
figsize = (8.5, 5.3)
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
# Read experiment files
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

if not records:
    raise FileNotFoundError(f"No 'Both Boxes' NetCDF files found in {data_dir}")

EArc_T_atm_means = np.array(sorted({record["E_mean"] for record in records}))
EArc_T_atm_amps = np.array(sorted({record["E_amp"] for record in records}))
WArc_T_atm_means = np.array(sorted({record["W_mean"] for record in records}))
WArc_T_atm_amps = np.array(sorted({record["W_amp"] for record in records}))

num_days_below_freezing_EArc = np.full(
    (EArc_T_atm_means.size, EArc_T_atm_amps.size), np.nan
)
num_days_below_freezing_WArc = np.full(
    (WArc_T_atm_means.size, WArc_T_atm_amps.size), np.nan
)
mean_days_min_to_max_EArc = np.full_like(num_days_below_freezing_EArc, np.nan)
mean_days_min_to_max_WArc = np.full_like(num_days_below_freezing_WArc, np.nan)

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
        T_atm_EArc = np.asarray(ds["T_atm_EArc"].values).squeeze().astype(float)
        T_atm_WArc = np.asarray(ds["T_atm_WArc"].values).squeeze().astype(float)

    T_atm_EArc = T_atm_EArc[:n_days]
    T_atm_WArc = T_atm_WArc[:n_days]
    h_ice_EArc = h_ice_EArc[:n_days]
    h_ice_WArc = h_ice_WArc[:n_days]
    SIC_EArc = SIC_EArc[:n_days]
    SIC_WArc = SIC_WArc[:n_days]

    num_days_below_freezing_EArc[ind_mean_EArc, ind_amp_EArc] = np.sum(
        T_atm_EArc[:days_per_year] < T_freezing
    )
    num_days_below_freezing_WArc[ind_mean_WArc, ind_amp_WArc] = np.sum(
        T_atm_WArc[:days_per_year] < T_freezing
    )

    ice_volume_EArc = SIC_EArc * h_ice_EArc * box_dims.area_EArc
    ice_volume_WArc = SIC_WArc * h_ice_WArc * box_dims.area_WArc

    ice_volume_EArc = ice_volume_EArc.reshape(n_years, days_per_year)
    ice_volume_WArc = ice_volume_WArc.reshape(n_years, days_per_year)

    min_ice_day_EArc = (np.argmin(ice_volume_EArc, axis=1) + 1) % days_per_year
    min_ice_day_WArc = (np.argmin(ice_volume_WArc, axis=1) + 1) % days_per_year

    max_ice_day_EArc = (np.argmax(ice_volume_EArc, axis=1) + 1) % days_per_year
    max_ice_day_WArc = (np.argmax(ice_volume_WArc, axis=1) + 1) % days_per_year

    days_min_to_max_EArc = (max_ice_day_EArc[1:] + days_per_year) - min_ice_day_EArc[:-1]
    days_min_to_max_WArc = (max_ice_day_WArc[1:] + days_per_year) - min_ice_day_WArc[:-1]

    mean_days_min_to_max_EArc[ind_mean_EArc, ind_amp_EArc] = np.nanmean(
        days_min_to_max_EArc[-num_years_for_mean:]
    )
    mean_days_min_to_max_WArc[ind_mean_WArc, ind_amp_WArc] = np.nanmean(
        days_min_to_max_WArc[-num_years_for_mean:]
    )

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
xticks_EArc = EArc_T_atm_amps
yticks_EArc = EArc_T_atm_means
xticks_WArc = WArc_T_atm_amps
yticks_WArc = WArc_T_atm_means

xticklabels_EArc = [str(round(x, 1)) if i % 2 == 0 else "" for i, x in enumerate(xticks_EArc)]
yticklabels_EArc = [str(round(y, 1)) if i % 2 == 0 else "" for i, y in enumerate(yticks_EArc)]
xticklabels_WArc = [str(round(x, 1)) if i % 2 == 0 else "" for i, x in enumerate(xticks_WArc)]
yticklabels_WArc = [ str(round(y, 1)) if i % 2 == 0 else "" for i, y in enumerate(yticks_WArc)]

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

top_cmin = np.floor(
    min(np.nanmin(num_days_below_freezing_EArc), np.nanmin(num_days_below_freezing_WArc))
    / 25
) * 25
top_cmax = np.ceil(
    max(np.nanmax(num_days_below_freezing_EArc), np.nanmax(num_days_below_freezing_WArc))
    / 25
) * 25
top_ticks = np.arange(top_cmin, top_cmax + 1, 25)

bottom_cmin = 250
bottom_cmax = 365
bottom_ticks = np.linspace(bottom_cmin, bottom_cmax, 6)

# -----------------------------------------------------------------------------
# Create figure
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    hspace=0.55,
    wspace=0.25,
)

ax_E_top = fig.add_subplot(gs[0, 0])
ax_W_top = fig.add_subplot(gs[0, 1])

ax_E_bottom = fig.add_subplot(gs[1, 0])
ax_W_bottom = fig.add_subplot(gs[1, 1])

im_E_top = ax_E_top.imshow(
    num_days_below_freezing_EArc,
    cmap=cmocean.cm.deep,
    aspect="auto",
    origin="lower",
    extent=extent_EArc,
    vmin=top_cmin,
    vmax=top_cmax,
)
im_W_top = ax_W_top.imshow(
    num_days_below_freezing_WArc,
    cmap=cmocean.cm.deep,
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
    vmin=top_cmin,
    vmax=top_cmax,
)

im_E_bottom = ax_E_bottom.imshow(
    mean_days_min_to_max_EArc,
    cmap="viridis_r",
    aspect="auto",
    origin="lower",
    extent=extent_EArc,
    vmin=bottom_cmin,
    vmax=bottom_cmax,
)
im_W_bottom = ax_W_bottom.imshow(
    mean_days_min_to_max_WArc,
    cmap="viridis_r",
    aspect="auto",
    origin="lower",
    extent=extent_WArc,
    vmin=bottom_cmin,
    vmax=bottom_cmax,
)

for ax in [ax_E_top, ax_E_bottom]:
    ax.plot(
        box_paramsEArc.T_atm_amp,
        box_paramsEArc.T_atm_mean,
        default_shape,
        c=default_color,
        markersize=default_markersize,
        markeredgecolor=default_markeredgecolor,
    )

for ax in [ax_W_top, ax_W_bottom]:
    ax.plot(
        box_paramsWArc.T_atm_amp,
        box_paramsWArc.T_atm_mean,
        default_shape,
        c=default_color,
        markersize=default_markersize,
        markeredgecolor=default_markeredgecolor,
    )

for ax in [ax_E_top, ax_E_bottom]:
    ax.set_xticks(xticks_EArc)
    ax.set_xticklabels(xticklabels_EArc)
    ax.set_yticks(yticks_EArc)
    ax.set_yticklabels(yticklabels_EArc)
    ax.set_ylabel("Air Temperature Mean [°C]", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)
    if ax == ax_E_bottom:
        ax.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)

for ax in [ax_W_top, ax_W_bottom]:
    ax.set_xticks(xticks_WArc)
    ax.set_xticklabels(xticklabels_WArc)
    ax.set_yticks(yticks_WArc)
    ax.set_yticklabels(yticklabels_WArc)
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)
    if ax == ax_W_bottom:
        ax.set_xlabel("Air Temperature Amplitude [°C]", fontsize=fontsize)

ax_E_top.set_title("(a) East Arctic", fontsize=title_fontsize)
ax_W_top.set_title("(b) West Arctic", fontsize=title_fontsize)
ax_E_bottom.set_title("(c) East Arctic", fontsize=title_fontsize)
ax_W_bottom.set_title("(d) West Arctic", fontsize=title_fontsize)

# Make each plot panel square.
# for ax in [ax_E_top, ax_W_top, ax_E_bottom, ax_W_bottom]:
#     ax.set_box_aspect(1)

fig.text(
    0.5,
    0.96,
    f"Number of days below {T_freezing:.2f}°C",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight='bold'
)
fig.text(
    0.5,
    0.49,
    "Number of days between ice volume minima and maxima",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight='bold'
)

fig.subplots_adjust(right=0.90)

pos_top = ax_W_top.get_position()
cbar_top_ax = fig.add_axes([0.93, pos_top.y0, 0.015, pos_top.height])
cbar_top = fig.colorbar(im_W_top, cax=cbar_top_ax)

pos_bottom = ax_W_bottom.get_position()
cbar_bottom_ax = fig.add_axes([0.93, pos_bottom.y0, 0.015, pos_bottom.height])
cbar_bottom = fig.colorbar(im_W_bottom, cax=cbar_bottom_ax)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
