# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from typing import Tuple, Union

import numpy as np

from .arctic_ice_model_args import (
    ArcticIceModelParameters,
    ArcticIceModelBoxDimensions,
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic
)
from .arctic_ice_model_constants import Constants


def compute_new_ice_temperatures(step: int,
                                 box_idx: int,
                                 h_ice: np.ndarray,
                                 T_atm: np.ndarray,
                                 T_ocean_freezing: np.ndarray,
                                 k_ice: float
                                 ) -> Tuple[float, float]:
    """
    Compute the sea ice surface and bottom temperatures for an Arctic box.

    :param step: Current model time step index
    :param box_idx: Arctic box index
    :param h_ice: Sea ice thickness time series [m]
    :param T_atm: Air temperature time series [degC]
    :param T_ocean_freezing: Ocean freezing temperature time series [degC]
    :param k_ice: Sea ice thermal conductivity [W m^-1 K^-1]
    :return: Sea ice surface and bottom temperatures [degC]
    """
    if h_ice[step, box_idx] > 0:
        if T_atm[step, box_idx] < T_ocean_freezing[step, box_idx]:
            # Freezing season
            T_icebottom = T_ocean_freezing[step, box_idx]
            T_icesurface = (k_ice / (Constants.k_atm * h_ice[step, box_idx])) * (
                    T_ocean_freezing[step, box_idx] - T_atm[step, box_idx]) + T_atm[step, box_idx]
            if T_icesurface > T_ocean_freezing[step, box_idx]:
                T_icesurface = T_ocean_freezing[step, box_idx]
        else:
            # Melting season
            T_icebottom = T_ocean_freezing[step, box_idx]
            T_icesurface = T_ocean_freezing[step, box_idx]
    else:
        # No sea ice present, ice temperatures set to ocean temperature
        T_icebottom = T_ocean_freezing[step, box_idx]
        T_icesurface = T_ocean_freezing[step, box_idx]

    return T_icesurface, T_icebottom


def compute_ice_thickness_concentration(
        step: int,
        box_idx: int,
        paramsBox: Union[ArcticIceModelParametersWestArctic, ArcticIceModelParametersEastArctic],
        params: ArcticIceModelParameters,
        box_dims: ArcticIceModelBoxDimensions,
        h_ice: np.ndarray,
        add_h_ice: np.ndarray,
        SIC: np.ndarray,
        FDD: np.ndarray,
        FDD_startstep: int,
        TDD: np.ndarray,
        TDD_counter: int,
        F_heat: float,
        T_atm: np.ndarray,
        T_ocean: np.ndarray,
        T_ocean_freezing: np.ndarray,
        density_ice: np.ndarray,
        ice_type: np.ndarray,
        add_h_ice_intervention: np.ndarray,
        time_step_size_in_years: float,
        seasonal_cycle_period: float,
        cycle_step: int,
        dt: float,
        days_per_year: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Compute the updated sea ice thickness, concentration, and ice type for an Arctic box.

    :param step: Current model time step index
    :param box_idx: Arctic box index
    :param paramsBox: Box-specific model parameters
    :param params: Model parameters shared by both Arctic boxes
    :param box_dims: Arctic box dimensions
    :param h_ice: Sea ice thickness time series [m]
    :param add_h_ice: Additional sea ice thickness from excess concentration [m]
    :param SIC: Sea ice concentration time series
    :param FDD: Freezing degree day time series
    :param FDD_startstep: Starting time step of the current freezing season
    :param TDD: Thawing degree day time series
    :param TDD_counter: Number of thawing days in the current melting season
    :param F_heat: Heat flux [W m^-2]
    :param T_atm: Air temperature time series [degC]
    :param T_ocean: Ocean temperature time series [degC]
    :param T_ocean_freezing: Ocean freezing temperature time series [degC]
    :param density_ice: Ice density time series [kg m^-3]
    :param ice_type: Ice type time series
    :param add_h_ice_intervention: Additional ice thickness from intervention [m]
    :param time_step_size_in_years: Model time step size [years]
    :param seasonal_cycle_period: Length of the seasonal cycle [years]
    :param cycle_step: Current step within the seasonal cycle
    :param dt: Model time step size [s]
    :param days_per_year: Number of days per year
    :return: Updated sea ice thickness, sea ice concentration, ice type, freezing season start step, and thawing day counter
    """

    # Account for export of ice from land and North Pacific/Atlantic to boxes
    if box_idx == Constants.EArc_idx:
        if h_ice[step, box_idx] > 0:
            dSIC = (paramsBox.Lice_daily - paramsBox.Eice_daily) / (h_ice[step, box_idx] * box_dims.area_EArc)
        else:  # assume h_ice 1cm
            dSIC = (paramsBox.Lice_daily - paramsBox.Eice_daily) / (0.01 * box_dims.area_EArc)
    else:
        if h_ice[step, box_idx] > 0:
            dSIC = (paramsBox.Lice_daily - paramsBox.Eice_daily) / (h_ice[step, box_idx] * box_dims.area_WArc)
        else:  # assume h_ice 1cm
            dSIC = (paramsBox.Lice_daily - paramsBox.Eice_daily) / (0.01 * box_dims.area_WArc)

    # These settings can be different for each box if desired.
    if box_idx == Constants.EArc_idx:
        h_ice_eqn_switch = 2.0  # m
        desch_deltah_multiplier = 0.8 # reduce Desch delta_h
    elif box_idx == Constants.WArc_idx:
        h_ice_eqn_switch = 2.0  # m
        desch_deltah_multiplier = 0.8 # reduce Desch delta_h
    else:
        raise ValueError(f'box_idx must be {Constants.EArc_idx} or {Constants.WArc_idx}. Got {box_idx}')

    # For first time step, need to compute FDD that corresponds to initial h_ice.
    if step == 0:
        FDD_startstep = step
        FDD_from_h_ice = (((2 * h_ice[step, box_idx] * 100 + 16.8) ** 2) - 16.8 ** 2) / (4 * 12.9)
        FDD[step, box_idx] = FDD_from_h_ice
        h_ice[step + 1, box_idx] = ((-16.8 + np.sqrt(16.8 ** 2 - (4 * -12.9 * FDD_from_h_ice))) / 2) / 100
        SIC[step + 1, box_idx] = SIC[step, box_idx] + ((1 - SIC[step, box_idx] ** 2) ** (1 / 2)) * (
                ((1 - SIC[step, box_idx]) * -F_heat) / (Constants.LI * 900 * h_ice[step + 1, box_idx])) * dt
        ice_type[step + 1, box_idx] = ice_type[step, box_idx]

    else:
        # Freezing season
        if T_atm[step, box_idx] < T_ocean_freezing[step, box_idx]:
            # Next freezing season, need to start with ice thickness from end of thawing season.
            # FDD equals 0 during TDD, so start next season when FDD is > 0 again.

            if not np.isnan(FDD[step - 1, box_idx]):
                # Inside freezing season
                FDD[step, box_idx] = T_ocean_freezing[step, box_idx] - T_atm[step, box_idx]
                FDD_cumsum = np.cumsum(FDD[FDD_startstep:step + 1, box_idx], axis=0)

                if ice_type[step, box_idx] == 2:
                    # Multi-year ice, switch to Maykut 1986 fig 12 equations
                    if h_ice[step, box_idx] > h_ice_eqn_switch:
                        deltah = (1.18 * np.exp(-0.415 * h_ice[step, box_idx])) / 270
                        h_ice[step + 1, box_idx] = h_ice[step, box_idx] + deltah + add_h_ice_intervention[step, box_idx]
                    else:
                        h_ice[step + 1, box_idx] = (
                                (((-16.8 + np.sqrt(
                                    16.8 ** 2 - (4 * -12.9 * FDD_cumsum[step - FDD_startstep]))) / 2) / 100)
                                + np.sum(add_h_ice[FDD_startstep:step + 1, box_idx], axis=0)
                                + np.sum(add_h_ice_intervention[FDD_startstep:step + 1, box_idx], axis=0)
                        )
                else:
                    # First-year ice, ice_type = 1
                    h_ice[step + 1, box_idx] = (
                            (((-16.8 + np.sqrt(16.8 ** 2 - (4 * -12.9 * FDD_cumsum[step - FDD_startstep]))) / 2) / 100)
                            + np.sum(add_h_ice[FDD_startstep:step + 1, box_idx], axis=0)
                            + np.sum(add_h_ice_intervention[FDD_startstep:step + 1, box_idx], axis=0)
                    )

            else:  
                # Start of new FDD season
                FDD_startstep = step

                if h_ice[step, box_idx] > 0.0:
                    FDD_from_h_ice = (((2 * h_ice[step, box_idx] * 100 + 16.8) ** 2) - 16.8 ** 2) / (4 * 12.9)
                    FDD[step, box_idx] = FDD_from_h_ice
                else:
                    FDD[step, box_idx] = T_ocean_freezing[step, box_idx] - T_atm[step, box_idx]

                FDD_cumsum = np.cumsum(FDD[FDD_startstep:step + 1, box_idx], axis=0)

                if ice_type[step, box_idx] == 2: 
                    # Multi-year ice, switch to Maykut 1986 fig 12 equations
                    if h_ice[step, box_idx] > h_ice_eqn_switch:
                        deltah = (1.18 * np.exp(-0.415 * h_ice[step, box_idx])) / 270
                        h_ice[step + 1, box_idx] = h_ice[step, box_idx] + deltah
                    else:
                        h_ice[step + 1, box_idx] = ((-16.8 + np.sqrt(
                            16.8 ** 2 - (4 * -12.9 * FDD_cumsum[step - FDD_startstep]))) / 2) / 100
                else:
                    # First-year ice, ice_type = 1
                    h_ice[step + 1, box_idx] = ((-16.8 + np.sqrt(
                        16.8 ** 2 - (4 * -12.9 * FDD_cumsum[step - FDD_startstep]))) / 2) / 100
                
                # If h_ice is very small, then F_b at next time step is very large, 
                # put limit on how small h_ice can be on first step of freezing season
                if (h_ice[step + 1, box_idx] < 0.0) & (h_ice[step + 1, box_idx] < 0.01):
                    h_ice[step + 1, box_idx] = 0.01

            if F_heat > 0.0:
                SIC[step + 1, box_idx] = SIC[step, box_idx]
                if SIC[step, box_idx] == 0.0 and h_ice[step + 1, box_idx] > 0:
                    # Ice thickness starts growing before SIC increases (due to offest in T_atm < T_ocean_freezing
                    # and F_heat > 0), set SIC to 1%
                    SIC[step + 1, box_idx] = 0.01 + dSIC
            else:
                nextSIC = (SIC[step, box_idx]
                           + ((1 - SIC[step, box_idx] ** 2) ** (1 / 2))
                           * (
                                   ((1 - SIC[step, box_idx]) * -F_heat)
                                   / (Constants.LI * density_ice[step, box_idx]
                                      * h_ice[step + 1, box_idx])
                           ) * dt
                           )
                if nextSIC < 0.0:
                    SIC[step + 1, box_idx] = 0.0
                    h_ice[step + 1, box_idx] = 0.0
                elif SIC[step, box_idx] == 0.0 and nextSIC > 1:
                    add_h_ice[step, box_idx] = add_h_ice[step, box_idx] + h_ice[step + 1, box_idx] * (nextSIC - SIC[step, box_idx]) 
                    SIC[step + 1, box_idx] = 0.1
                elif SIC[step, box_idx] > 0.0 and nextSIC > 1:
                    add_h_ice[step, box_idx] = add_h_ice[step, box_idx] + h_ice[step + 1, box_idx] * (nextSIC - SIC[step, box_idx]) 
                    SIC[step + 1, box_idx] = SIC[step, box_idx] * 1.1
                else:
                    SIC[step + 1, box_idx] = nextSIC
        else:
            # Melting season
            TDD[step, box_idx] = T_ocean_freezing[step, box_idx] - T_atm[step, box_idx]

            # TDD counter for albedo; the counter refers to number of thawing days
            # TDD equals 0 during FDD, when TDD starts, the previous TDD will be 0
            if TDD[step - 1, box_idx] < 0.0:
                TDD_counter += 1
            else:
                TDD_counter = 1

            # Ice melt equation - Desch et al 2016, page 115
            if TDD_counter < 20:
                albedo = 0.75
            else:
                albedo = 0.3

            # Sun's declination
            declination = 23.44 * np.sin(
                2 * np.pi * (((cycle_step + 1 - 79) * time_step_size_in_years) / seasonal_cycle_period))
            insolation = 1366 * (np.sin(np.deg2rad(Constants.latitude)) * np.sin(np.deg2rad(declination)) + np.cos(
                np.deg2rad(Constants.latitude)) * np.cos(np.deg2rad(declination)) * 1)

            if insolation < 0.0:
                insolation = 0.0
            heattransfer = 10.45 - paramsBox.U10m_atm + 10 * paramsBox.U10m_atm ** (1 / 2)

            if h_ice[step, box_idx] > h_ice_eqn_switch:
                # Wwitch to Maykut 1986 fig 12 equations
                deltah = (0.28 + 0.374 * np.exp(-0.412 * h_ice[step, box_idx])) / 90
            else:
                # For thinner ice, use Desch melting equation
                if T_ocean[step, box_idx] < 0.0:  
                    # Desch et al 2016: if T < 273 K (0 degC), then h*T term vanishes
                    deltah = desch_deltah_multiplier * (dt / (density_ice[step, box_idx] * Constants.LI)) * (
                            insolation * (1 - albedo) * (1 - params.cloudcover))
                else:
                    deltah = desch_deltah_multiplier * (dt / (density_ice[step, box_idx] * Constants.LI)) * (
                            insolation * (1 - albedo) * (1 - params.cloudcover) + heattransfer * T_ocean[step, box_idx])

            nextH = h_ice[step, box_idx] - deltah

            if nextH > 0:
                h_ice[step + 1, box_idx] = nextH
                nextSIC = SIC[step, box_idx] - (SIC[step, box_idx] / (2 * h_ice[step + 1, box_idx])) * deltah + dSIC
                if nextSIC < 0.0:
                    SIC[step + 1, box_idx] = 0.0
                    h_ice[step + 1, box_idx] = 0.0
                else:
                    SIC[step + 1, box_idx] = nextSIC
            else:
                h_ice[step + 1, box_idx] = 0.0
                SIC[step + 1, box_idx] = 0.0 + dSIC

    if SIC[step + 1, box_idx] > 1.0:
        # Move excess SIC to thickness, set SIC to 1
        add_h_ice[step, box_idx] = add_h_ice[step, box_idx] + h_ice[step + 1, box_idx] * (SIC[step + 1, box_idx] - SIC[step, box_idx])
        SIC[step + 1, box_idx] = 1.0

    # Determine ice_type by looking back at the past year
    # If all days have h_ice > 0, ice_type = 2 (MY)
    # If >=1 day has h_ice = 0, ice_type = 1 (FY)
    if step + 1 < days_per_year:
        if np.sum(h_ice[:step + 1, box_idx] == 0.) > 0:
            ice_type[step + 1, box_idx] = 1
        else:
            ice_type[step + 1, box_idx] = 2
    else:
        if np.sum(h_ice[step + 1 - days_per_year:step + 1, box_idx] == 0.) > 0:
            ice_type[step + 1, box_idx] = 1
        else:
            ice_type[step + 1, box_idx] = 2

    return h_ice, SIC, ice_type, FDD_startstep, TDD_counter
