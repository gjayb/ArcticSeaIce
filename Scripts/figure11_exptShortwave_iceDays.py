"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot the number of days between ice-volume minima and maxima for the
shortwave flux experiment.

Run from `Figures` directory: uv run Scripts/figure11_exptShortwave_iceDays.py
"""

from pathlib import Path
import re
import sys

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
fig_name = "figure11_exptShortwave_iceDays"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data" / "Experiment_ShortwaveFlux"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

nc_filename_base = "results_experiment_shortwave"

n_years = 50
days_per_year = 365
n_days = n_years * days_per_year
num_years_for_mean = 10

# Figure settings
figsize = (8.5, 2.5)
fontsize = 10
title_fontsize = fontsize + 2
default_shape = "*"
default_color = [0.5, 0.5, 0.5]
default_markersize = 10
default_markeredgecolor = "k"

# Model parameters
box_dims = ArcticIceModelBoxDimensions()
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

mean_num_days_EArc = np.full((shortwave_means.size, shortwave_amps.size), np.nan)
mean_num_days_WArc = np.full_like(mean_num_days_EArc, np.nan)

mean_to_ind = {value: i for i, value in enumerate(shortwave_means)}
amp_to_ind = {value: i for i, value in enumerate(shortwave_amps)}

# -----------------------------------------------------------------------------
# Read NetCDF data and compute metric
# -----------------------------------------------------------------------------
for record in records:
    ind_mean = mean_to_ind[record["mean"]]
    ind_amp = amp_to_ind[record["amp"]]

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

    new_shape = (n_years, days_per_year)
    ice_volume_EArc = ice_volume_EArc.reshape(new_shape)
    ice_volume_WArc = ice_volume_WArc.reshape(new_shape)

    min_ice_day_EArc = (np.argmin(ice_volume_EArc, axis=1) + 1) % days_per_year
    min_ice_day_WArc = (np.argmin(ice_volume_WArc, axis=1) + 1) % days_per_year

    max_ice_day_EArc = (np.argmax(ice_volume_EArc, axis=1) + 1) % days_per_year
    max_ice_day_WArc = (np.argmax(ice_volume_WArc, axis=1) + 1) % days_per_year

    num_days_EArc = (max_ice_day_EArc[1:] + days_per_year) - min_ice_day_EArc[:-1]
    num_days_WArc = (max_ice_day_WArc[1:] + days_per_year) - min_ice_day_WArc[:-1]

    mean_num_days_EArc[ind_mean, ind_amp] = np.nanmean(
        num_days_EArc[-num_years_for_mean:]
    )
    mean_num_days_WArc[ind_mean, ind_amp] = np.nanmean(
        num_days_WArc[-num_years_for_mean:]
    )


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

cmin = 260
cmax = 330
cticks = np.linspace(cmin, cmax, 6)


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

ax_E = fig.add_subplot(gs[0, 0])
ax_W = fig.add_subplot(gs[0, 1])

im_E = ax_E.imshow(
    mean_num_days_EArc,
    cmap="viridis_r",
    aspect="auto",
    origin="lower",
    extent=fig_extent,
    vmin=cmin,
    vmax=cmax,
)
im_W = ax_W.imshow(
    mean_num_days_WArc,
    cmap="viridis_r",
    aspect="auto",
    origin="lower",
    extent=fig_extent,
    vmin=cmin,
    vmax=cmax,
)

ax_E.plot(
    box_paramsEArc.shortwave_amp,
    box_paramsEArc.shortwave_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)
ax_W.plot(
    box_paramsWArc.shortwave_amp,
    box_paramsWArc.shortwave_mean,
    default_shape,
    c=default_color,
    markersize=default_markersize,
    markeredgecolor=default_markeredgecolor,
)

ax_E.set_xticks(xticks)
ax_E.set_xticklabels(xticklabels)
ax_E.set_yticks(yticks)
ax_E.set_yticklabels(yticklabels)
ax_E.set_xlabel("Shortwave Amplitude [W m$^{-2}$]", fontsize=fontsize)
ax_E.set_ylabel("Shortwave Mean [W m$^{-2}$]", fontsize=fontsize)
ax_E.set_title("(a) East Arctic", fontsize=title_fontsize)

ax_W.set_xticks(xticks)
ax_W.set_xticklabels(xticklabels)
ax_W.set_yticks(yticks)
ax_W.set_yticklabels(yticklabels)
ax_W.set_xlabel("Shortwave Amplitude [W m$^{-2}$]", fontsize=fontsize)
ax_W.set_title("(b) West Arctic", fontsize=title_fontsize)

for ax in [ax_E, ax_W]:
    ax.tick_params(labelsize=fontsize)
    ax.grid(False)

fig.text(
    0.5,
    1.05,
    "Number of days between ice volume minima and maxima",
    ha="center",
    va="top",
    fontsize=title_fontsize,
    fontweight="bold",
)

fig.subplots_adjust(right=0.90)
pos = ax_W.get_position()
cbar_ax = fig.add_axes([0.92, pos.y0, 0.015, pos.height])
cbar = fig.colorbar(im_W, cax=cbar_ax)
cbar.set_ticks(cticks)
cbar.ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
