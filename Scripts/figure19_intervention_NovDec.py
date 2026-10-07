"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot the difference in ice thickness between the Nov-Dec 10 cm/month
ice-addition intervention and the no-intervention case.

Run from `Figures` directory: uv run Scripts/figure19_intervention_NovDec.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure19_intervention_NovDec"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

# No-intervention and intervention model results.
nc_file_no_intervention = (
    data_dir / "results_expt_addIce_East_addhice000_West_addhice000_monthsNov-Dec.nc"
)
nc_file_intervention = (
    data_dir / "results_expt_addIce_East_addhice010_West_addhice010_monthsNov-Dec.nc"
)

# Figure settings
figsize = (8.5, 2.3)
linewidth = 1.5
fontsize = 10
title_fontsize = fontsize
legend_fontsize = 8

EArc_color = "darkblue"
WArc_color = "darkorange"


# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not nc_file_no_intervention.exists():
    raise FileNotFoundError(
        f"Could not find no-intervention NetCDF file: {nc_file_no_intervention}"
    )

if not nc_file_intervention.exists():
    raise FileNotFoundError(
        f"Could not find intervention NetCDF file: {nc_file_intervention}"
    )

with xr.open_dataset(nc_file_no_intervention) as ds:
    h_ice_EArc_no = np.asarray(ds["h_ice_EArc"].values).squeeze().astype(float)
    h_ice_WArc_no = np.asarray(ds["h_ice_WArc"].values).squeeze().astype(float)

    days_per_year = (
        int(np.asarray(ds["timestep_days_per_year"].values).squeeze())
        if "timestep_days_per_year" in ds
        else 365
    )

with xr.open_dataset(nc_file_intervention) as ds:
    h_ice_EArc_intervention = (
        np.asarray(ds["h_ice_EArc"].values).squeeze().astype(float)
    )
    h_ice_WArc_intervention = (
        np.asarray(ds["h_ice_WArc"].values).squeeze().astype(float)
    )

if len(h_ice_EArc_intervention) != len(h_ice_EArc_no):
    raise ValueError("East Arctic model runs have different lengths.")

if len(h_ice_WArc_intervention) != len(h_ice_WArc_no):
    raise ValueError("West Arctic model runs have different lengths.")

n_time = len(h_ice_EArc_no)
n_years = n_time // days_per_year

# Intervention minus no intervention.
h_ice_diff_EArc = h_ice_EArc_intervention - h_ice_EArc_no
h_ice_diff_WArc = h_ice_WArc_intervention - h_ice_WArc_no


# -----------------------------------------------------------------------------
# Time axis
# -----------------------------------------------------------------------------
timesteps = np.arange(1, n_time + 1)

# Put a tick at the beginning of every year and label every other year.
xticks = timesteps[::days_per_year]
years = np.arange(1, len(xticks) + 1)
xlabels = [str(year) if year % 2 == 1 else "" for year in years]


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=figsize)

ax.plot(
    timesteps,
    h_ice_diff_EArc,
    color=EArc_color,
    linewidth=linewidth,
    label="East Arctic",
)
ax.plot(
    timesteps,
    h_ice_diff_WArc,
    color=WArc_color,
    linewidth=linewidth,
    label="West Arctic",
)

ax.set_title(
    "Difference in ice thickness [m] (with intervention - without intervention)",
    fontsize=title_fontsize,
)
ax.set_xlabel("Year", fontsize=fontsize)
ax.set_ylabel("Ice Thickness Difference [m]", fontsize=fontsize)

ax.set_xlim([1, n_time])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.tick_params(labelsize=fontsize)

ax.grid(True, alpha=0.5)
ax.legend(loc="upper left", fontsize=legend_fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
