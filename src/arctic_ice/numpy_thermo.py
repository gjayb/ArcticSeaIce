# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.
#
# Portions of this file are derived from MetPy:
# heps://github.com/Unidata/MetPy/blob/main/src/metpy/calc/thermo.py
#
# Original work:
# Copyright (c) 2008, 2015-2019 MetPy Developers
#
# This implementa:on has been adapted to:
# - remove Pint and xarray dependencies
# - operate directly on NumPy arrays
# - simplify interfaces for the Arctic model

"""
A pure NumPy version of the MetPy (https://unidata.github.io/MetPy/latest/index.html) thermodynamic functions used in
the Arctic model.
"""
import numpy as np


class MetpyUnits:
    epsilon = 0.6219569100577033
    Lv = 2500840.0
    Cp_l = 4219.400000000001
    Cp_v = 1860.078011865639
    T0 = 273.16
    Ls = 2834540.0
    Cp_i = 2090
    Rv = 461.52311572606084
    sat_pressure_0c = 611.2


def mixing_ratio(partial_press, total_press, molecular_weight_ratio=MetpyUnits.epsilon):
    r"""Calculate the mixing ratio of a gas.

    This calculates mixing ratio given its partial pressure and the total pressure of
    the air. There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    partial_press : `pint.Quantity`
        Partial pressure of the constituent gas

    total_press :
        Total air pressure

    molecular_weight_ratio :  or float, optional
        The ratio of the molecular weight of the constituent gas to that assumed
        for air. Defaults to the ratio for water vapor to dry air
        (:math:`\epsilon\approx0.622`).

    Returns
    -------
        The (mass) mixing ratio, dimensionless (e.g. Kg/Kg or g/g)
    """
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


def water_latent_heat_vaporization(temperature):
    r"""Calculate the latent heat of vaporization for water.

    Accounts for variations in latent heat across valid temperature range.

    Parameters
    ----------
    temperature :

    Returns
    -------
        Latent heat of vaporization
    """
    return MetpyUnits.Lv - (MetpyUnits.Cp_l - MetpyUnits.Cp_v) * (temperature - MetpyUnits.T0)


def water_latent_heat_sublimation(temperature):
    r"""Calculate the latent heat of sublimation for water.

    Accounts for variations in latent heat across valid temperature range.

    Parameters
    ----------
    temperature :

    Returns
    -------
        Latent heat of vaporization
    """
    # Note: temperature should be in kelvin
    return MetpyUnits.Ls - (MetpyUnits.Cp_i - MetpyUnits.Cp_v) * (temperature - MetpyUnits.T0)


def _saturation_vapor_pressure_liquid(temperature):
    r"""Calculate saturation (equilibrium) water vapor (partial) pressure over liquid water.

    Parameters
    ----------
    temperature :
        Air temperature

    Returns
    -------
        Saturation water vapor (partial) pressure over liquid water
    """
    latent_heat = water_latent_heat_vaporization(temperature)
    heat_power = (MetpyUnits.Cp_l - MetpyUnits.Cp_v) / MetpyUnits.Rv
    exp_term = (MetpyUnits.Lv / MetpyUnits.T0 - latent_heat / temperature) / MetpyUnits.Rv

    return MetpyUnits.sat_pressure_0c * (MetpyUnits.T0 / temperature) ** heat_power * np.exp(exp_term)


def _saturation_vapor_pressure_solid(temperature):
    r"""Calculate the saturation water vapor (partial) pressure over solid water (ice).

    Parameters
    ----------
    temperature :
        Air temperature

    Returns
    -------
        Saturation water vapor (partial) pressure over solid water (ice)
    """
    latent_heat = water_latent_heat_sublimation(temperature)
    heat_power = (MetpyUnits.Cp_i - MetpyUnits.Cp_v) / MetpyUnits.Rv
    exp_term = (MetpyUnits.Ls / MetpyUnits.T0 - latent_heat / temperature) / MetpyUnits.Rv

    return MetpyUnits.sat_pressure_0c * (MetpyUnits.T0 / temperature) ** heat_power * np.exp(exp_term)


def saturation_vapor_pressure(temperature, *, phase='liquid'):
    r"""Calculate the saturation (equilibrium) water vapor (partial) pressure.

    Parameters
    ----------
    temperature :
        Air temperature

    phase : {'liquid', 'solid', 'auto'}
        Where applicable, adjust assumptions and constants to make calculation valid in
        ``'liquid'`` water (default) or ``'solid'`` ice regimes. ``'auto'`` will change regime
        based on determination of phase boundaries, eg `temperature` relative to freezing.

    Returns
    -------
        Saturation water vapor (partial) pressure
    """
    if phase == 'liquid':
        return _saturation_vapor_pressure_liquid(temperature)
    elif phase == 'solid':
        return _saturation_vapor_pressure_solid(temperature)
    else:
        raise ValueError(
            f'{phase!r} is not a valid option for phase. '
            f"Valid options are {'liquid', 'solid', 'auto'}.")


def saturation_mixing_ratio(total_press, temperature, *, phase='liquid'):
    r"""Calculate the saturation mixing ratio of water vapor.

    This calculation is given total atmospheric pressure and air temperature.

    Parameters
    ----------
    total_press:
        Total atmospheric pressure

    temperature:
        Air temperature

    phase : {'liquid', 'solid', 'auto'}
        Where applicable, adjust assumptions and constants to make calculation valid in
        ``'liquid'`` water (default) or ``'solid'`` ice regimes. ``'auto'`` will change regime
        based on determination of phase boundaries, eg `temperature` relative to freezing.

    Returns
    -------
        Saturation mixing ratio, dimensionless
    """
    e_s = saturation_vapor_pressure(temperature, phase=phase)
    undefined = e_s >= total_press
    return np.where(undefined, np.nan, mixing_ratio(e_s, total_press))


def specific_humidity_from_mixing_ratio(mixing_ratio_):
    r"""Calculate the specific humidity from the mixing ratio.

    Parameters
    ----------
    mixing_ratio_: Mixing ratio

    Returns
    -------
        Specific humidity
    """
    return mixing_ratio_ / (1 + mixing_ratio_)


def specific_humidity_from_dewpoint(pressure, dewpoint, *, phase='liquid'):
    r"""Calculate the specific humidity from the dewpoint temperature and pressure.

    Parameters
    ----------
    pressure: units=Atm
        Pressure

    dewpoint: units= degrees C
        Dewpoint temperature

    phase : {'liquid', 'solid', 'auto'}
        Where applicable, adjust assumptions and constants to make calculation valid in
        ``'liquid'`` water (default) or ``'solid'`` ice regimes. ``'auto'`` will change regime
        based on determination of phase boundaries, eg `temperature` relative to freezing.

    Returns
    -------
        Specific humidity
    """
    # Note: convert dewpoint to Kelvin and Pascals here
    dewpoint_kelvin = dewpoint + 273.15
    pressure_pascals = pressure * 100

    mixing_ratio_ = saturation_mixing_ratio(pressure_pascals, dewpoint_kelvin, phase=phase)
    return specific_humidity_from_mixing_ratio(mixing_ratio_)


def vapor_pressure(pressure, mixing_ratio_):
    r"""Calculate water vapor (partial) pressure.

    Given total ``pressure`` and water vapor ``mixing_ratio``, calculates the
    partial pressure of water vapor.

    Parameters
    ----------
    pressure : units = Atm
        Total atmospheric pressure

    mixing_ratio_ : units = gram/kilogram
        Dimensionless mass mixing ratio

    Returns
    -------
        Ambient water vapor (partial) pressure in the same units as ``pressure``
    """
    # epsilon is in g/mol / kg/mol => g/kg
    # Note: convert to Pascals and kg/kg here
    pressure_pascals = pressure * 100
    mixing_ratio_kg_kg = mixing_ratio_ * 0.001  # g/kg to kg/kg
    out_Pa = pressure_pascals * mixing_ratio_kg_kg / (MetpyUnits.epsilon + mixing_ratio_kg_kg)
    return out_Pa * 0.01  # to hPa

