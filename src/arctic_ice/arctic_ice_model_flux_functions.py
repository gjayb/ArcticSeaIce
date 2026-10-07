# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from typing import Union, Tuple

import numpy as np

from .arctic_ice_model_args import (
    ArcticIceModelParameters, ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic
)
from .arctic_ice_model_constants import Constants
from .arctic_ice_model_utils import get_daily_precipitation, get_daily_river_discharge
from .numpy_thermo import specific_humidity_from_dewpoint, vapor_pressure as calc_vapor_pressure


def compute_heat_fluxes(step: int,
                        box_idx: int,
                        paramsBox: Union[ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic],
                        params: ArcticIceModelParameters,
                        SIC: np.ndarray,
                        T_ocean: np.ndarray,
                        T_atm: np.ndarray,
                        time_step_size_in_years: float,
                        seasonal_cycle_period: float,
                        cycle_step: int
                        ) -> Tuple[float, float, float, float]:
    """
    Compute the shortwave, longwave, sensible, and latent heat fluxes for an Arctic box.

    :param step: Current model time step index
    :param box_idx: Arctic box index
    :param paramsBox: Box-specific model parameters
    :param params: Model parameters shared by both Arctic boxes
    :param SIC: Sea ice concentration time series
    :param T_ocean: Ocean temperature time series [degC]
    :param T_atm: Air temperature time series [degC]
    :param time_step_size_in_years: Model time step size [years]
    :param seasonal_cycle_period: Length of the seasonal cycle [years]
    :param cycle_step: Current step within the seasonal cycle
    :return: Shortwave, longwave, sensible, and latent heat fluxes [W m^-2]
    """
    dewpt_atm = paramsBox.dewpt_atm_mean + paramsBox.dewpt_atm_amp * np.cos(
        2 * np.pi * ((cycle_step * time_step_size_in_years) / seasonal_cycle_period))
    dewpt_ocean = T_atm[step, box_idx]  # approx the same as air temperature, setting ocean dewpoint and Q to atm values.

    Q_atm = specific_humidity_from_dewpoint(Constants.pressure_atm, dewpt_atm)
    Q_ocean = specific_humidity_from_dewpoint(Constants.pressure_atm, dewpt_ocean)

    coalbedo_ice = 1 - params.albedo_ice
    coalbedo_ocean = 1 - params.albedo_ocean
    vapor_pressure = calc_vapor_pressure(Constants.pressure_atm, Constants.mixing_ratio)

    # Shortwave Flux [W m-2]
    F_sw = ((coalbedo_ice * SIC[step, box_idx] + coalbedo_ocean * (1 - SIC[step, box_idx]))
            * (paramsBox.shortwave_mean
               - paramsBox.shortwave_amp * np.cos(2 * np.pi * (((cycle_step + 11) * time_step_size_in_years)
                                                               / seasonal_cycle_period))))
    if F_sw < 0:
        F_sw = 0.

    # Longwave Flux [W m-2]
    F_lw = ((1 - SIC[step, box_idx])
            * (Constants.emittance * Constants.sb_constant
               * ((T_ocean[step, box_idx] + 273.15) ** 4) * (0.39 - 0.05 * vapor_pressure ** (1 / 2))
               * (1 - Constants.cloudcover_coefficient * params.cloudcover ** 2)
               + 4 * Constants.emittance * Constants.sb_constant * ((T_ocean[step, box_idx] + 273.15) ** 3)
               * (T_ocean[step, box_idx] - T_atm[step, box_idx])
               )
            )

    # Sensible Heat Flux [W m-2]
    F_sh = paramsBox.U10m_atm * Constants.density_atm * Constants.cp_atm * (
            (1 - SIC[step, box_idx]) * (Constants.CHE_water * (T_atm[step, box_idx] - T_ocean[step, box_idx])))

    # Latent Heat Flux [W m-2]
    diffQ_atm_ocean = Q_atm - Q_ocean
    if diffQ_atm_ocean > 0:
        diffQ_atm_ocean = 0
    F_lh = paramsBox.U10m_atm * Constants.density_atm * Constants.LE * (
            (1 - SIC[step, box_idx]) * (Constants.CHE_water * diffQ_atm_ocean))

    return F_sw, F_lw, F_sh, F_lh


def compute_salt_fluxes(step: int,
                        box_idx: int,
                        params: ArcticIceModelParameters,
                        h_ice: np.ndarray,
                        SIC: np.ndarray,
                        S_ocean: np.ndarray,
                        S_ice: np.ndarray,
                        density_ice: np.ndarray,
                        ice_type: np.ndarray,
                        dt: float,
                        ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute the freshwater uptake and brine discharge salt fluxes for an Arctic box.

    :param step: Current model time step index
    :param box_idx: Arctic box index
    :param params: Model parameters shared by both Arctic boxes
    :param h_ice: Sea ice thickness time series [m]
    :param SIC: Sea ice concentration time series
    :param S_ocean: Ocean salinity time series [ppt]
    :param S_ice: Ice salinity time series [ppt]
    :param density_ice: Ice density time series [kg m^-3]
    :param ice_type: Ice type time series
    :param dt: Model time step size [s]
    :return: Freshwater uptake flux, brine discharge flux, sea ice density, and sea ice salinity
    """
    if step == 0:
        S_ice[step, box_idx] = 4.0
        density_ice[step, box_idx] = 900.0
        F_bd = 0.0
        F_up = 0.0
    else:
        if h_ice[step, box_idx] == 0:
            S_ice[step, box_idx] = 14
            density_ice[step, box_idx] = 920
        elif ice_type[step, box_idx] == 2.:
            S_ice[step, box_idx] = 0.5
            density_ice[step, box_idx] = 800
        else:
            if h_ice[step, box_idx] > 0.3:
                S_ice[step, box_idx] = 4
                density_ice[step, box_idx] = 900
            elif (h_ice[step, box_idx] > 0.1) & (h_ice[step, box_idx] < 0.3):
                S_ice[step, box_idx] = 9
                density_ice[step, box_idx] = 900
            elif h_ice[step, box_idx] < 0.1:
                S_ice[step, box_idx] = 14
                density_ice[step, box_idx] = 920
            else:
                raise ValueError(f"Invalid values for h_ice and ice_type: "
                                 f"{h_ice[step, box_idx], ice_type[step, box_idx]}")

        dSice_dt = (S_ice[step, box_idx] - S_ice[step - 1, box_idx]) / dt
        dhice_dt = (h_ice[step, box_idx] - h_ice[step - 1, box_idx]) / dt

        # Brine Discharge Flux [g salt m-2 s-1]
        if (SIC[step - 1, box_idx] > 0.0) and (SIC[step, box_idx] == 0.0):
            # When SIC goes to 0, return salt to ocean
            F_bd = (h_ice[step - 1, box_idx] * SIC[step - 1, box_idx] * S_ice[step - 1, box_idx] * density_ice[
                step - 1, box_idx]) / dt
        elif dhice_dt > 0:
            F_bd = -density_ice[step, box_idx] * (
                    (h_ice[step, box_idx] * dSice_dt) + (S_ocean[step, box_idx] - params.S_newice) * dhice_dt)
        elif dhice_dt <= 0:
            F_bd = -density_ice[step, box_idx] * (
                    (h_ice[step, box_idx] * dSice_dt) - (S_ice[step, box_idx] * dhice_dt))
        else:
            raise ValueError(f"Invalid conditions for dhice_dt {dhice_dt} or SIC")

        # Freshwater Uptake Flux [g salt m-2 s-1]
        F_up = density_ice[step, box_idx] * S_ocean[step, box_idx] * (1 - Constants.phi_up) * dhice_dt

    return F_up, F_bd, density_ice, S_ice


def compute_freshwater_fluxes(step: int,
                              box_idx: int,
                              paramsBox: Union[ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic],
                              T_atm: np.ndarray,
                              box_area: float,
                              F_lh: np.ndarray,
                              days_per_year: int
                              ) -> Tuple[float, float, float]:
    """
    Compute the river discharge, precipitation, and latent heat freshwater fluxes for an Arctic box.

    :param step: Current model time step index
    :param box_idx: Arctic box index
    :param paramsBox: Box-specific model parameters
    :param T_atm: Air temperature time series [degC]
    :param box_area: Surface area of the Arctic box [m2]
    :param F_lh: Latent heat flux time series [W m^-2]
    :param days_per_year: Number of days per year
    :return: River discharge, precipitation, and latent heat freshwater fluxes [kg m^-2 s^-1]
    """
    river_discharge = get_daily_river_discharge(step, paramsBox, days_per_year)
    precip_rate = get_daily_precipitation(step, paramsBox, days_per_year)

    # Need four freezing days for river to freeze
    if step < 3:
        num_freezing_days = sum(T_atm[0:step + 1, box_idx] < 0)
    else:
        num_freezing_days = sum(T_atm[step - 3:step + 1, box_idx] < 0)

    # River Discharge Flux [kg freshwater m-2 s-1]
    if num_freezing_days == 4:
        F_r = 0.0
    else:
        F_r = (river_discharge * Constants.density_freshwater) / box_area

    # Precipitation Flux [kg freshwater m-2 s-1]
    F_p = precip_rate * Constants.density_freshwater

    # Latent Heat Flux for freshwater [kg freshwater m-2 s-1]
    F_lhfresh = (F_lh[step, box_idx] / Constants.LV)
    
    return F_r, F_p, F_lhfresh
