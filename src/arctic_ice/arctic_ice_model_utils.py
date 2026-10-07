# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

from typing import Union

import numpy as np

from .arctic_ice_model_args import ArcticIceModelParametersWestArctic, ArcticIceModelParametersEastArctic


def get_daily_river_discharge(step: int,
                              paramsBox: Union[ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic],
                              days_per_year: int) -> float:
    """
    Gets the daily river discharge for the model time step from monthly river data [Sv]
        and converts to m3/s.
    For simulations with ``days_per_year = 360``, each month is assumed to have 30 days.
    For simulations with ``days_per_year = 365``, the standard Gregorian month lengths are used.

    :param step: Current model time step index
    :param paramsBox: Box-specific model parameters
    :param days_per_year: Number of days per year
    :return: Daily river discharge [m3 s^-1]
    """
    daynum = (step + 1) % days_per_year

    if days_per_year == 360:
        if daynum <= 30:
            river_discharge = paramsBox.river_monthly[0] * 1e6
        elif (daynum > 30) & (daynum <= 60):
            river_discharge = paramsBox.river_monthly[1] * 1e6
        elif (daynum > 60) & (daynum <= 90):
            river_discharge = paramsBox.river_monthly[2] * 1e6
        elif (daynum > 90) & (daynum <= 120):
            river_discharge = paramsBox.river_monthly[3] * 1e6
        elif (daynum > 120) & (daynum <= 150):
            river_discharge = paramsBox.river_monthly[4] * 1e6
        elif (daynum > 150) & (daynum <= 180):
            river_discharge = paramsBox.river_monthly[5] * 1e6
        elif (daynum > 180) & (daynum <= 210):
            river_discharge = paramsBox.river_monthly[6] * 1e6
        elif (daynum > 210) & (daynum <= 240):
            river_discharge = paramsBox.river_monthly[7] * 1e6
        elif (daynum > 240) & (daynum <= 270):
            river_discharge = paramsBox.river_monthly[8] * 1e6
        elif (daynum > 270) & (daynum <= 300):
            river_discharge = paramsBox.river_monthly[9] * 1e6
        elif (daynum > 300) & (daynum <= 330):
            river_discharge = paramsBox.river_monthly[10] * 1e6
        elif (daynum > 330) & (daynum <= 360):
            river_discharge = paramsBox.river_monthly[11] * 1e6
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    elif days_per_year == 365:
        if daynum <= 31:
            river_discharge = paramsBox.river_monthly[0] * 1e6
        elif (daynum > 31) & (daynum <= 59):
            river_discharge = paramsBox.river_monthly[1] * 1e6
        elif (daynum > 59) & (daynum <= 91):
            river_discharge = paramsBox.river_monthly[2] * 1e6
        elif (daynum > 91) & (daynum <= 120):
            river_discharge = paramsBox.river_monthly[3] * 1e6
        elif (daynum > 120) & (daynum <= 151):
            river_discharge = paramsBox.river_monthly[4] * 1e6
        elif (daynum > 151) & (daynum <= 181):
            river_discharge = paramsBox.river_monthly[5] * 1e6
        elif (daynum > 181) & (daynum <= 212):
            river_discharge = paramsBox.river_monthly[6] * 1e6
        elif (daynum > 212) & (daynum <= 243):
            river_discharge = paramsBox.river_monthly[7] * 1e6
        elif (daynum > 243) & (daynum <= 273):
            river_discharge = paramsBox.river_monthly[8] * 1e6
        elif (daynum > 273) & (daynum <= 304):
            river_discharge = paramsBox.river_monthly[9] * 1e6
        elif (daynum > 304) & (daynum <= 334):
            river_discharge = paramsBox.river_monthly[10] * 1e6
        elif (daynum > 334) & (daynum <= 365):
            river_discharge = paramsBox.river_monthly[11] * 1e6
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    else:
        raise ValueError(f'Constants.days_per_year must be 360 or 365')

    return river_discharge


def get_daily_precipitation(step: int,
                            paramsBox: Union[ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic],
                            days_per_year: int) -> float:
    """
    Gets the daily precipitation rate for the model time step from monthly 
        precipitation data [mm/month] and converts to m/s.

    For simulations with ``days_per_year = 360``, each month is assumed to have 30 days.
    For simulations with ``days_per_year = 365``, the standard Gregorian month lengths are used.

    :param step: Current model time step index
    :param paramsBox: Box-specific model parameters
    :param days_per_year: Number of days per year
    :return: Daily precipitation rate [m s^-1]
    """
    daynum = (step + 1) % days_per_year

    if days_per_year == 360:
        if daynum <= 30:
            precip = paramsBox.precip_monthly[0] / (1000 * 30 * 86400)
        elif (daynum > 30) & (daynum <= 60):
            precip = paramsBox.precip_monthly[1] / (1000 * 30 * 86400)
        elif (daynum > 60) & (daynum <= 90):
            precip = paramsBox.precip_monthly[2] / (1000 * 30 * 86400)
        elif (daynum > 90) & (daynum <= 120):
            precip = paramsBox.precip_monthly[3] / (1000 * 30 * 86400)
        elif (daynum > 120) & (daynum <= 150):
            precip = paramsBox.precip_monthly[4] / (1000 * 30 * 86400)
        elif (daynum > 150) & (daynum <= 180):
            precip = paramsBox.precip_monthly[5] / (1000 * 30 * 86400)
        elif (daynum > 180) & (daynum <= 210):
            precip = paramsBox.precip_monthly[6] / (1000 * 30 * 86400)
        elif (daynum > 210) & (daynum <= 240):
            precip = paramsBox.precip_monthly[7] / (1000 * 30 * 86400)
        elif (daynum > 240) & (daynum <= 270):
            precip = paramsBox.precip_monthly[8] / (1000 * 30 * 86400)
        elif (daynum > 270) & (daynum <= 300):
            precip = paramsBox.precip_monthly[9] / (1000 * 30 * 86400)
        elif (daynum > 300) & (daynum <= 330):
            precip = paramsBox.precip_monthly[10] / (1000 * 30 * 86400)
        elif (daynum > 330) & (daynum <= 360):
            precip = paramsBox.precip_monthly[11] / (1000 * 30 * 86400)
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    elif days_per_year == 365:
        if daynum <= 31:
            precip = paramsBox.precip_monthly[0] / (1000 * 31 * 86400)
        elif (daynum > 31) & (daynum <= 59):
            precip = paramsBox.precip_monthly[1] / (1000 * 28 * 86400)
        elif (daynum > 59) & (daynum <= 91):
            precip = paramsBox.precip_monthly[2] / (1000 * 31 * 86400)
        elif (daynum > 91) & (daynum <= 120):
            precip = paramsBox.precip_monthly[3] / (1000 * 30 * 86400)
        elif (daynum > 120) & (daynum <= 151):
            precip = paramsBox.precip_monthly[4] / (1000 * 31 * 86400)
        elif (daynum > 151) & (daynum <= 181):
            precip = paramsBox.precip_monthly[5] / (1000 * 30 * 86400)
        elif (daynum > 181) & (daynum <= 212):
            precip = paramsBox.precip_monthly[6] / (1000 * 31 * 86400)
        elif (daynum > 212) & (daynum <= 243):
            precip = paramsBox.precip_monthly[7] / (1000 * 31 * 86400)
        elif (daynum > 243) & (daynum <= 273):
            precip = paramsBox.precip_monthly[8] / (1000 * 30 * 86400)
        elif (daynum > 273) & (daynum <= 304):
            precip = paramsBox.precip_monthly[9] / (1000 * 31 * 86400)
        elif (daynum > 304) & (daynum <= 334):
            precip = paramsBox.precip_monthly[10] / (1000 * 30 * 86400)
        elif (daynum > 334) & (daynum <= 365):
            precip = paramsBox.precip_monthly[11] / (1000 * 31 * 86400)
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    else:
        raise ValueError(f'Constants.days_per_year must be 360 or 365')
        
    return precip


def replace_nan_with_interp(step: int, box_id: int, var: np.ndarray) -> float:
    """
    Replace a NaN value with a linearly interpolated value.

    :param step: Current model time step index
    :param box_id: Arctic box index
    :param var: Time series to interpolate
    :return: Interpolated value
    """
    window_size = 10
    total_steps = var.shape[0]

    # To account for values at beginning/end of var time series
    window_start = max(0, step - window_size)
    window_end = min(total_steps, step + window_size + 1)

    x_vals = np.arange(window_start, window_end)
    y_vals = var[window_start:window_end, box_id]

    valid = ~np.isnan(y_vals)  # values that are not nans

    if np.sum(valid) >= 2:  # Need at least 2 points to interpolate
        new_y_vals = np.interp(step, x_vals[valid], y_vals[valid])
    else:
        raise ValueError(f"Not enough valid data to interpolate at step={step}")

    return new_y_vals


def get_daily_ice_intervention(step: int,
                               paramsBox: Union[ArcticIceModelParametersEastArctic, ArcticIceModelParametersWestArctic],
                               days_per_year: int):
    """
    Gets the daily add_h_ice_intervention rate for the model time step from monthly 
        ice intervention data [cm/month] and converts to m/day.

    For simulations with ``days_per_year = 360``, each month is assumed to have 30 days.
    For simulations with ``days_per_year = 365``, the standard Gregorian month lengths are used.

    :param step: Current model time step index
    :param paramsBox: Box-specific model parameters
    :param days_per_year: Number of days per year
    :return: Daily ice intervention rate [m day^-1]
    """
    daynum = (step + 1) % days_per_year

    if days_per_year == 360:
        if daynum <= 30:
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[0] / (100 * 31)
        elif (daynum > 30) & (daynum <= 60):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[1] / (100 * 28)
        elif (daynum > 60) & (daynum <= 90):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[2] / (100 * 31)
        elif (daynum > 90) & (daynum <= 120):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[3] / (100 * 30)
        elif (daynum > 120) & (daynum <= 150):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[4] / (100 * 31)
        elif (daynum > 150) & (daynum <= 180):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[5] / (100 * 30)
        elif (daynum > 180) & (daynum <= 210):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[6] / (100 * 31)
        elif (daynum > 210) & (daynum <= 240):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[7] / (100 * 31)
        elif (daynum > 240) & (daynum <= 270):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[8] / (100 * 30)
        elif (daynum > 270) & (daynum <= 300):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[9] / (100 * 31)
        elif (daynum > 300) & (daynum <= 330):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[10] / (100 * 30)
        elif (daynum > 330) & (daynum <= 360):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[11] / (100 * 31)
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    elif days_per_year == 365:
        if daynum <= 31:
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[0] / (100 * 31)
        elif (daynum > 31) & (daynum <= 59):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[1] / (100 * 28)
        elif (daynum > 59) & (daynum <= 91):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[2] / (100 * 31)
        elif (daynum > 91) & (daynum <= 120):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[3] / (100 * 30)
        elif (daynum > 120) & (daynum <= 151):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[4] / (100 * 31)
        elif (daynum > 151) & (daynum <= 181):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[5] / (100 * 30)
        elif (daynum > 181) & (daynum <= 212):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[6] / (100 * 31)
        elif (daynum > 212) & (daynum <= 243):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[7] / (100 * 31)
        elif (daynum > 243) & (daynum <= 273):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[8] / (100 * 30)
        elif (daynum > 273) & (daynum <= 304):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[9] / (100 * 31)
        elif (daynum > 304) & (daynum <= 334):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[10] / (100 * 30)
        elif (daynum > 334) & (daynum <= 365):
            add_h_ice_intervention = paramsBox.add_h_ice_intervention_monthly[11] / (100 * 31)
        else:
            raise ValueError(f'daynum out of range: {daynum}')
    else:
        raise ValueError(f'Constants.days_per_year must be 360 or 365')

    return add_h_ice_intervention
