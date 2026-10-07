# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

class Constants:
    """
    seconds_per_year: Seconds in one year
    density_freshwater: Density of freshwater [kg m-3]
    density_atm: Air density [kg m-3]
    cp_atm: Specific heat of air [J kg-1 K-1]
    pressure_atm: Air pressure [hPa]
    mixing_ratio: Mass mixing ratio [g kg-1]
    k_atm: Atmospheric heat exchange coefficient [W m-1 K-1]
    phi_up: Partitioning coefficient for flushing
    emittance: Emittance of the sea surface
    cloudcover_coefficient: Cloud cover coefficient at 80°N
    sb_constant: Stefan-Boltzmann constant [W m-2 K-4]
    latitude: Mean latitude [°N]
    CHE_ice: Turbulent exchange coefficient for ice
    CHE_water: Turbulent exchange coefficient for water
    LE: Latent heat of sublimation [J kg-1]
    LI: Volumetric latent heat of fusion of ice [J kg-1]
    LV: Latent heat of evaporation of seawater [J kg-1]
    EArc_idx: Index value for the East Arctic box
    WArc_idx: Index value for the West Arctic box
    """
    seconds_per_day = 24 * 60 * 60
    density_freshwater = 1000
    density_atm = 1.39
    cp_atm = 1006
    pressure_atm = 1013
    mixing_ratio = 1
    k_atm = 20
    phi_up = 0.3
    emittance = 0.98
    cloudcover_coefficient = 0.84
    sb_constant = 5.67e-8
    latitude = 80
    CHE_ice = 1e-3
    CHE_water = 1e-3
    LE = 575 * 4184
    LI = 332000
    LV = 2500000
    EArc_idx = 0
    WArc_idx = 1
