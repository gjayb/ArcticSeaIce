"""
Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Distributed under the terms of the BSD 3-Clause License.

Plot map with Arctic ice box model boxes.

Run from `Figures` directory: uv run Scripts/figure01_Arctic_map.py
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Figure settings
# -----------------------------------------------------------------------------
fig_name = "figure01_Arctic_map"

script_dir = Path(__file__).resolve().parent
fig_dir = script_dir.parent / "Pngs"
fig_dir.mkdir(parents=True, exist_ok=True)

fontsize = 8
label_fontsize = 12
east_color = (0.0, 0.0, 0.545)
west_color = (1.0, 0.549, 0.0)
land_color = (0.7, 0.7, 0.7)

map_crs = ccrs.NorthPolarStereo(central_longitude=0)
data_crs = ccrs.PlateCarree()

MERIDIANS = np.arange(-180, 181, 30)
PARALLELS = (60, 75)
LABEL_LAT = 53
GRID_ZORDER = 10
TEXT_ZORDER = GRID_ZORDER + 1
LABEL_ZORDER = GRID_ZORDER + 2

# -----------------------------------------------------------------------------
# Box bounds
# -----------------------------------------------------------------------------
# East Arctic
east_boundary_lon = np.arange(180, -181, -1)
east_boundary_lat = np.full_like(east_boundary_lon, 65.0, dtype=float)

east_lon = np.r_[east_boundary_lon, -180, 180]
east_lat = np.r_[east_boundary_lat, 90, 90]

# West Arctic
west_boundary_lon = np.arange(10, -171, -1)
west_boundary_lat = np.full_like(west_boundary_lon, 57.0, dtype=float)

west_lon = np.r_[west_boundary_lon, -170, 10]
west_lat = np.r_[west_boundary_lat, 90, 90]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def format_lon_label(lon):
    """Format longitude labels as 30°W, 0°, 30°E, etc."""
    if lon == 0:
        return r"0$^\circ$"
    if abs(lon) == 180:
        return r"180$^\circ$"

    hemi = "E" if lon > 0 else "W"
    return rf"{abs(int(lon))}$^\circ${hemi}"


# -----------------------------------------------------------------------------
# Common plotting keyword arguments
# -----------------------------------------------------------------------------
grid_kwargs = dict(
    transform=data_crs,
    color="0.5",
    linewidth=0.5,
    zorder=GRID_ZORDER,
)

text_kwargs = dict(
    transform=data_crs,
    ha="center",
    va="center",
    fontsize=fontsize,
    zorder=TEXT_ZORDER,
)

label_kwargs = dict(
    transform=data_crs,
    color="white",
    fontsize=label_fontsize,
    fontweight="bold",
    ha="center",
    va="center",
    zorder=LABEL_ZORDER,
)

# -----------------------------------------------------------------------------
# Make plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(5, 5),
    subplot_kw={"projection": map_crs},
)

ax.set_extent([-180, 180, 48, 90], crs=data_crs)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
circle = mpath.Path(
    np.array([0.5, 0.5]) +
    0.5 * np.vstack([np.sin(theta), np.cos(theta)]).T
)
ax.set_boundary(circle, transform=ax.transAxes)

# Plot Arctic boxes
ax.fill(
    east_lon,
    east_lat,
    facecolor=east_color,
    edgecolor="k",
    linewidth=0.5,
    transform=data_crs,
    zorder=1,
)

ax.fill(
    west_lon,
    west_lat,
    facecolor=west_color,
    edgecolor="k",
    linewidth=0.5,
    transform=data_crs,
    zorder=2,
)

# Plot land
ax.add_feature(
    cfeature.LAND.with_scale("50m"),
    facecolor=land_color,
    edgecolor="k",
    linewidth=0.5,
    zorder=3,
)

# Plot meridians
for lon in MERIDIANS:
    ax.plot(
        np.full(300, lon),
        np.linspace(48, 90, 300),
        **grid_kwargs,
    )

# Plot parallels
for lat in PARALLELS:
    ax.plot(
        np.linspace(-180, 180, 500),
        np.full(500, lat),
        **grid_kwargs,
    )

# Latitude labels
for lon, lat, label in [
    (-180, 60, r"60$^\circ$N"),
    (-150, 75, r"75$^\circ$N"),
    # (0, 90, r"90$^\circ$N"),
]:
    ax.text(lon, lat, label, **text_kwargs)

# Longitude labels
for lon in MERIDIANS[1:]:
    ax.text(
        lon,
        LABEL_LAT,
        format_lon_label(lon),
        **text_kwargs,
    )

# Box labels
ax.text(130, 80, "East", **label_kwargs)
ax.text(-130, 80, "West", **label_kwargs)

fig.savefig(fig_dir / f"{fig_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {fig_dir / f'{fig_name}.png'}")
