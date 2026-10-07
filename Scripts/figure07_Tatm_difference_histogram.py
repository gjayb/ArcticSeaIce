"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot the distribution of modeled minus satellite air temperature
for the Arctic ice box model.

Run from `Figures` directory: uv run Scripts/figure07_Tatm_difference_histogram.py
"""

from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import arctic_ice.arctic_ice_model_validation as val_funcs


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure07_Tatm_difference_histogram"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

# Data files
nc_file = data_dir / "results_T_atm_modeled.nc"
tatm_mat_file = data_dir / "AirTemp2m_Greenland_ArcticOcean_Boxes.mat"

# Date range used for the validation data
date1 = datetime(1990, 1, 1)
date2 = datetime(2023, 12, 31)

# Box indices used in the model scripts
EArc_idx = 0
WArc_idx = 1
NUM_BOXES = 2

# Figure settings
EArc_color = "darkblue"
WArc_color = "darkorange"

figsize = (5.0, 3.0)
fontsize = 10
title_fontsize = fontsize
legend_fontsize = 10 

# Histogram settings
bins = np.arange(-10, 11, 1)
xticks = np.arange(-10, 11, 1)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def read_satellite_tatm(tatm_file: Path) -> tuple[list[datetime], np.ndarray]:
    """Read, filter, and format satellite air temperature data."""
    valdata = val_funcs.ValidationData()

    valdata.T_atm_dates, valdata.T_atm_EArc, valdata.T_atm_WArc, _, _ = (
        val_funcs.read_merra_from_mat(tatm_file)
    )

    # Convert temperature from Kelvin to Celsius.
    valdata.T_atm_EArc = valdata.T_atm_EArc - 273.15
    valdata.T_atm_WArc = valdata.T_atm_WArc - 273.15

    # Remove leap days because the model uses 365 days per year.
    leap_day_ind_tatm = [
        i for i, d in enumerate(valdata.T_atm_dates) if d.month == 2 and d.day == 29
    ]

    valdata.T_atm_dates = [
        d for i, d in enumerate(valdata.T_atm_dates) if i not in leap_day_ind_tatm
    ]

    for attr in ["T_atm_EArc", "T_atm_WArc"]:
        setattr(
            valdata,
            attr,
            np.delete(getattr(valdata, attr), leap_day_ind_tatm, axis=0),
        )

    # Filter to the validation date range.
    tatm_dates = np.array(valdata.T_atm_dates)
    tatm_idx = np.where((tatm_dates >= date1) & (tatm_dates <= date2))[0]

    valdata.T_atm_dates = [valdata.T_atm_dates[i] for i in tatm_idx]
    valdata.T_atm_EArc = valdata.T_atm_EArc[tatm_idx]
    valdata.T_atm_WArc = valdata.T_atm_WArc[tatm_idx]

    valdata_T_atm = np.full((NUM_BOXES, len(valdata.T_atm_dates)), np.nan)
    valdata_T_atm[EArc_idx] = valdata.T_atm_EArc.reshape(1, len(valdata.T_atm_dates))
    valdata_T_atm[WArc_idx] = valdata.T_atm_WArc.reshape(1, len(valdata.T_atm_dates))

    return valdata.T_atm_dates, valdata_T_atm


# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not nc_file.exists():
    raise FileNotFoundError(f"Could not find NetCDF file: {nc_file}")

if not tatm_mat_file.exists():
    raise FileNotFoundError(f"Could not find satellite air-temperature MAT file: {tatm_mat_file}")

with xr.open_dataset(nc_file) as ds:
    T_atm_EArc = np.asarray(ds["T_atm_EArc"].values).squeeze().astype(float)
    T_atm_WArc = np.asarray(ds["T_atm_WArc"].values).squeeze().astype(float)

n_time = len(T_atm_EArc)

_, valdata_T_atm = read_satellite_tatm(tatm_mat_file)

if valdata_T_atm.shape[1] != n_time:
    raise ValueError(
        "Satellite air-temperature data length does not match model output length. "
        f"model={n_time}, satellite={valdata_T_atm.shape[1]}"
    )

# Compute model - satellite differences
diff_EArc = T_atm_EArc - valdata_T_atm[EArc_idx]
diff_WArc = T_atm_WArc - valdata_T_atm[WArc_idx]

diff_EArc_valid = diff_EArc[np.isfinite(diff_EArc)]
diff_WArc_valid = diff_WArc[np.isfinite(diff_WArc)]

std_EArc = np.nanstd(diff_EArc)
std_WArc = np.nanstd(diff_WArc)

# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=figsize)

ax.hist(
    diff_EArc_valid,
    bins=bins,
    color=EArc_color,
    alpha=0.5,
    label="East Arctic",
)

ax.hist(
    diff_WArc_valid,
    bins=bins,
    color=WArc_color,
    alpha=0.5,
    label="West Arctic",
)

ax.set_xlim([-10, 10])
ax.set_xticks(xticks)
ax.set_xticklabels([f"{tick:d}" if i % 2 == 0 else "" for i, tick in enumerate(xticks)])
ax.set_xlabel("Model - satellite air temperature [°C]", fontsize=fontsize)
ax.set_ylabel("Count", fontsize=fontsize)
# ax.set_title("Modeled - Satellite Air Temperatures", fontsize=title_fontsize)
ax.tick_params(labelsize=fontsize)
ax.grid(False)

ymax = ax.get_ylim()[1]
ax.text(
    -9.5,
    0.91 * ymax,
    f"1 st. dev. = {std_EArc:.1f}", #f"East Arctic 1 std = {std_EArc:.1f}",
    color=EArc_color,
    fontsize=legend_fontsize,
)

ax.text(
    -9.5,
    0.83 * ymax,
    f"1 st. dev. = {std_WArc:.1f}", #f"West Arctic 1 std = {std_WArc:.1f}",
    color=WArc_color,
    fontsize=legend_fontsize,
)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
