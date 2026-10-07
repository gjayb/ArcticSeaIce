"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot air temperature and ice results for Arctic ice box model
results using satellite data as input air temperature.

Run from `Figures` directory: uv run Scripts/figure04_Tatm_satellite.py
"""

from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

import arctic_ice.arctic_ice_model_validation as val_funcs


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
fig_name = "figure04_satelliteTatm_results"

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "Data"
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

# Data files
nc_file = data_dir / "results_T_atm_satellite.nc"
sic_mat_file = data_dir / "IceCon_East_West_Boxes_1990-2024.mat"
hice_mat_file = data_dir / "cdr_sea_ice_thickness_2025_Greenland_ArcticOcean_Boxes.mat"

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
EArc_sat_color = "lightblue"
WArc_sat_color = "wheat"

figsize = (8.5, 9)
linewidth = 1.5
markersize = 1
fontsize = 10
title_fontsize = fontsize
legend_fontsize = 8
missing_data_color = "grey"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def read_satellite_data(
    sic_file: Path,
    hice_file: Path,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Read, filter, and format satellite h_ice and SIC data."""
    valdata = val_funcs.ValidationData()

    valdata.SIC_dates, valdata.SIC_EArc, valdata.SIC_WArc = val_funcs.read_sic_from_mat(
        sic_file
    )
    valdata.h_ice_dates, valdata.h_ice_EArc, valdata.h_ice_WArc, _, _ = (
        val_funcs.read_avhrr_from_mat(hice_file)
    )

    # Remove leap days because the model uses 365 days per year.
    leap_day_ind_sic = [
        i for i, d in enumerate(valdata.SIC_dates) if d.month == 2 and d.day == 29
    ]
    leap_day_ind_hice = [
        i for i, d in enumerate(valdata.h_ice_dates) if d.month == 2 and d.day == 29
    ]

    valdata.SIC_dates = [
        d for i, d in enumerate(valdata.SIC_dates) if i not in leap_day_ind_sic
    ]
    valdata.h_ice_dates = [
        d for i, d in enumerate(valdata.h_ice_dates) if i not in leap_day_ind_hice
    ]

    for attr, idx in {
        "SIC_EArc": leap_day_ind_sic,
        "SIC_WArc": leap_day_ind_sic,
        "h_ice_EArc": leap_day_ind_hice,
        "h_ice_WArc": leap_day_ind_hice,
    }.items():
        setattr(valdata, attr, np.delete(getattr(valdata, attr), idx, axis=0))

    # Filter to the validation date range.
    sic_dates = np.array(valdata.SIC_dates)
    hice_dates = np.array(valdata.h_ice_dates)
    sic_idx = np.where((sic_dates >= date1) & (sic_dates <= date2))[0]
    hice_idx = np.where((hice_dates >= date1) & (hice_dates <= date2))[0]

    valdata.SIC_dates = [valdata.SIC_dates[i] for i in sic_idx]
    valdata.SIC_EArc = valdata.SIC_EArc[sic_idx]
    valdata.SIC_WArc = valdata.SIC_WArc[sic_idx]

    valdata.h_ice_dates = [valdata.h_ice_dates[i] for i in hice_idx]
    valdata.h_ice_EArc = valdata.h_ice_EArc[hice_idx]
    valdata.h_ice_WArc = valdata.h_ice_WArc[hice_idx]

    val_dates = valdata.h_ice_dates

    valdata_SIC = np.full((NUM_BOXES, len(val_dates)), np.nan, dtype=np.float64)
    valdata_SIC[EArc_idx] = valdata.SIC_EArc.reshape(1, len(val_dates))
    valdata_SIC[WArc_idx] = valdata.SIC_WArc.reshape(1, len(val_dates))

    valdata_h_ice = np.full((NUM_BOXES, len(val_dates)), np.nan, dtype=np.float64)
    valdata_h_ice[EArc_idx] = valdata.h_ice_EArc.reshape(1, len(val_dates))
    valdata_h_ice[WArc_idx] = valdata.h_ice_WArc.reshape(1, len(val_dates))

    return val_dates, valdata_h_ice, valdata_SIC


def format_time_axis(
    ax,
    xticks: np.ndarray,
    xlabels: list[str],
    n_time: int,
    nan_indices: np.ndarray | None = None,
    show_xlabel: bool = False,
) -> None:
    """Apply common time-axis formatting and missing-data gray lines."""
    ax.grid(False)
    ax.set_xlim([0, n_time - 1])
    ax.set_xticks(xticks)

    if show_xlabel:
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Year", fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    if nan_indices is not None:
        for nan_idx in nan_indices:
            ax.axvline(
                x=nan_idx,
                color=missing_data_color,
                linestyle="solid",
                linewidth=0.7,
            )


def plot_1to1_scatter(
    ax,
    model: np.ndarray,
    satellite: np.ndarray,
    color: str,
    title: str,
    xlim: list[float] | None = None,
    ylim: list[float] | None = None,
) -> None:
    """Plot model-versus-satellite scatter with a 1:1 line."""
    model = np.asarray(model, dtype=float)
    satellite = np.asarray(satellite, dtype=float)
    valid = np.isfinite(model) & np.isfinite(satellite)

    model = model[valid]
    satellite = satellite[valid]

    if xlim is None or ylim is None:
        max_val = max(np.nanmax(model), np.nanmax(satellite)) * 1.05
        xlim = [0, max_val]
        ylim = [0, max_val]

    ax.scatter(model, satellite, s=markersize, color=color, label="_nolegend_")
    ax.plot(
        [xlim[0], xlim[1]],
        [ylim[0], ylim[1]],
        color="gray",
        linestyle="-",
        linewidth=1,
        label="1:1",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    tick_values = [xlim[0], xlim[1]]
    tick_labels = [f"{value:g}" for value in tick_values]
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Model", fontsize=fontsize)
    ax.set_ylabel("Satellite", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc="upper left", fontsize=legend_fontsize)


# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
if not nc_file.exists():
    raise FileNotFoundError(f"Could not find NetCDF file: {nc_file}")

with xr.open_dataset(nc_file) as ds:
    # Model output
    h_ice_EArc = np.asarray(ds["h_ice_EArc"].values).squeeze().astype(float)
    h_ice_WArc = np.asarray(ds["h_ice_WArc"].values).squeeze().astype(float)
    SIC_EArc = np.asarray(ds["SIC_EArc"].values).squeeze().astype(float)
    SIC_WArc = np.asarray(ds["SIC_WArc"].values).squeeze().astype(float)
    ice_type_EArc = np.asarray(ds["ice_type_EArc"].values).squeeze().astype(float)
    ice_type_WArc = np.asarray(ds["ice_type_WArc"].values).squeeze().astype(float)
    T_atm_EArc = np.asarray(ds["T_atm_EArc"].values).squeeze().astype(float)
    T_atm_WArc = np.asarray(ds["T_atm_WArc"].values).squeeze().astype(float)

    # Satellite T_atm 
    if "paramsEArc_T_atm" in ds:
        T_atm_sat_EArc = np.asarray(ds["paramsEArc_T_atm"].values).squeeze().astype(float)
    else:
        T_atm_sat_EArc = T_atm_EArc.copy()

    if "paramsWArc_T_atm" in ds:
        T_atm_sat_WArc = np.asarray(ds["paramsWArc_T_atm"].values).squeeze().astype(float)
    else:
        T_atm_sat_WArc = T_atm_WArc.copy()

    days_per_year = int(np.asarray(ds["timestep_days_per_year"].values).squeeze()) if "timestep_days_per_year" in ds else 365
    n_years = int(np.asarray(ds["timestep_n_years"].values).squeeze()) if "timestep_n_years" in ds else len(h_ice_EArc) // days_per_year

n_time = len(h_ice_EArc)

# Satellite h_ice and SIC from MAT files.
val_dates, valdata_h_ice, valdata_SIC = read_satellite_data(sic_mat_file, hice_mat_file)

if valdata_h_ice.shape[1] != n_time or valdata_SIC.shape[1] != n_time:
    raise ValueError(
        "Satellite data length does not match model output length. "
        f"model={n_time}, h_ice={valdata_h_ice.shape[1]}, SIC={valdata_SIC.shape[1]}"
    )

val_dates = val_dates[:n_time]

xticks = np.arange(0, n_years * days_per_year, days_per_year)
xlabels = [dt.strftime("%Y") for dt in val_dates[::days_per_year]]
xlabels = [label if i % 4 == 0 else "" for i, label in enumerate(xlabels)]

# Missing T_atm input indices are shown as gray vertical lines.
nan_Tatm_EArc = np.where(np.isnan(T_atm_sat_EArc))[0]
nan_Tatm_WArc = np.where(np.isnan(T_atm_sat_WArc))[0]
all_nans = np.unique(np.concatenate((nan_Tatm_EArc, nan_Tatm_WArc)))

# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=figsize)
gs = GridSpec(
    nrows=7,
    ncols=4,
    figure=fig,
    height_ratios=[1.0, 1.0, 1.05, 1.05, 1.05, 0.25, 1.65],
    hspace=0.45,
    wspace=0.45,
)

# T_atm line plots.
ax_T_E = fig.add_subplot(gs[0, :])
ax_T_W = fig.add_subplot(gs[1, :])

ax_T_E.plot(T_atm_EArc, color=EArc_color, label="Model")
ax_T_E.plot(T_atm_sat_EArc, color=EArc_sat_color, label="Satellite", alpha=0.0) # alpha=0.8
ax_T_E.set_title("(a) Air Temperature East Box", fontsize=title_fontsize)
ax_T_E.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
format_time_axis(ax_T_E, xticks, xlabels, n_time, all_nans, show_xlabel=False)

ax_T_W.plot(T_atm_WArc, color=WArc_color, label="Model")
ax_T_W.plot(T_atm_sat_WArc, color=WArc_sat_color, label="Satellite", alpha=0.0) # alpha=0.8
ax_T_W.set_title("(b) Air Temperature West Box", fontsize=title_fontsize)
ax_T_W.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
format_time_axis(ax_T_W, xticks, xlabels, n_time, all_nans, show_xlabel=False)

# Ice line plots.
ax_h = fig.add_subplot(gs[2, :])
ax_sic = fig.add_subplot(gs[3, :])
ax_type = fig.add_subplot(gs[4, :])

ax_h.plot(h_ice_EArc, label="Model, East Arctic", color=EArc_color, linewidth=linewidth)
ax_h.plot(h_ice_WArc, label="Model, West Arctic", color=WArc_color, linewidth=linewidth)
ax_h.plot(valdata_h_ice[EArc_idx], label="Satellite, East Arctic",
          color=EArc_sat_color, linewidth=linewidth)
ax_h.plot(valdata_h_ice[WArc_idx], label="Satellite, West Arctic",
          color=WArc_sat_color, linewidth=linewidth)
ax_h.set_title("(c) Ice thickness", fontsize=title_fontsize)
ax_h.set_ylim([-0.1, 4])
# ax_h.legend(loc="upper right", fontsize=legend_fontsize, ncol=4)
format_time_axis(ax_h, xticks, xlabels, n_time, all_nans, show_xlabel=False)

ax_sic.plot(SIC_EArc, label="Model, East Arctic", color=EArc_color, linewidth=linewidth)
ax_sic.plot(SIC_WArc, label="Model, West Arctic", color=WArc_color, linewidth=linewidth)
ax_sic.plot(valdata_SIC[EArc_idx], label="Satellite, East Arctic",
            color=EArc_sat_color, linewidth=linewidth)
ax_sic.plot(valdata_SIC[WArc_idx], label="Satellite, West Arctic",
            color=WArc_sat_color, linewidth=linewidth)
ax_sic.set_title("(d) SIC", fontsize=title_fontsize)
ax_sic.set_ylim([-0.1, 1.1])
format_time_axis(ax_sic, xticks, xlabels, n_time, all_nans, show_xlabel=False)

ax_type.plot(ice_type_EArc, label="Model, East Arctic", color=EArc_color, linewidth=linewidth)
ax_type.plot(ice_type_WArc, label="Model, West Arctic", color=WArc_color, linewidth=linewidth)
ax_type.set_title("(e) Ice Type", fontsize=title_fontsize)
ax_type.set_ylim([0.9, 2.1])
ax_type.set_yticks([1, 2])
ax_type.set_yticklabels(["FY", "MY"])
format_time_axis(ax_type, xticks, xlabels, n_time, all_nans, show_xlabel=True)

for ax, label in zip(
    [ax_T_E, ax_T_W, ax_h, ax_sic, ax_type],
    ["°C", "°C", "m", "SIC", "Ice type"],
):
    ax.set_ylabel(label, fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    
# Scatter plots
ax_sc_h_w = fig.add_subplot(gs[6, 0])
ax_sc_h_e = fig.add_subplot(gs[6, 1])
ax_sc_s_w = fig.add_subplot(gs[6, 2])
ax_sc_s_e = fig.add_subplot(gs[6, 3])

hice_w_max = np.ceil(
    np.nanmax(np.concatenate((h_ice_WArc, valdata_h_ice[WArc_idx])))
)
hice_e_max = np.ceil(
    np.nanmax(np.concatenate((h_ice_EArc, valdata_h_ice[EArc_idx])))
)

plot_1to1_scatter(
    ax_sc_h_w,
    h_ice_WArc,
    valdata_h_ice[WArc_idx],
    WArc_color,
    "(f) Ice Thickness [m]\nWest Box",
    xlim=[0, hice_w_max],
    ylim=[0, hice_w_max]
)
plot_1to1_scatter(
    ax_sc_h_e,
    h_ice_EArc,
    valdata_h_ice[EArc_idx],
    EArc_color,
    "(g) Ice Thickness [m]\nEast Box",
    xlim=[0, hice_e_max],
    ylim=[0, hice_e_max]
)
plot_1to1_scatter(
    ax_sc_s_w,
    SIC_WArc,
    valdata_SIC[WArc_idx],
    WArc_color,
    "(h) SIC\nWest Box",
    xlim=[0, 1],
    ylim=[0, 1],
)
plot_1to1_scatter(
    ax_sc_s_e,
    SIC_EArc,
    valdata_SIC[EArc_idx],
    EArc_color,
    "(i) SIC\nEast Box",
    xlim=[0, 1],
    ylim=[0, 1],
)

for ax in [ax_sc_h_e, ax_sc_s_w, ax_sc_s_e]:
    ax.set_ylabel("")

for ax in [ax_T_E, ax_T_W, ax_h, ax_sic, ax_type]:
    ax.tick_params(labelsize=fontsize)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
