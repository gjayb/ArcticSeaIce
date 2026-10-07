# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import numpy as np
from copy import deepcopy


class ArcticIceModelBoxDimensions:
    def __init__(self,
                 area_EArc: float = 5.95449672e12,
                 area_WArc: float = 9.03862210e12,
                 area_NGreen: float = 1.18279185e12,
                 area_SGreen: float = 9.8596193e11,
                 area_NAtl: float = 0.22e14,
                 area_NPac: float = 0.1e14,
                 Lx_east: float = 6.4e6,
                 Ly_east: float = 1.93e5,
                 Lx_west: float = 8.2e4,
                 Ly_west: float = 1e6,
                 d_ocean: float = 100
                 ):
        """
        Dimensions and areas for the East and West Arctic boxes.

        :param area_EArc: Surface area of the east Arctic box [m2]
        :param area_WArc: Surface area of the west Arctic box [m2]
        :param area_NGreen: Surface area of the north Greenland ice sheet box [m2]
        :param area_SGreen: Surface area of the south Greenland ice sheet box [m2]
        :param area_NAtl: Surface area of the North Atlantic box (from MOC model) [m2]
        :param area_NPac: Surface area of the North Pacific box (from MOC model) [m2]
        :param Lx_east: Length of interface between the east and west Arctic boxes [m]
        :param Ly_east: Length across east Arctic box (~10deg longitude at 80degN) [m]
        :param Lx_west: Width of Bering Strait [m]
        :param Ly_west: Length across west Arctic box [m]
        :param d_ocean: Depth of Arctic boxes [m]
        """
        self.area_EArc = area_EArc
        self.area_WArc = area_WArc
        self.area_NGreen = area_NGreen
        self.area_SGreen = area_SGreen
        self.area_NAtl = area_NAtl
        self.area_NPac = area_NPac
        self.Lx_east = Lx_east
        self.Ly_east = Ly_east
        self.Lx_west = Lx_west
        self.Ly_west = Ly_west
        self.d_ocean = d_ocean

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelBoxDimensions(**self.to_dict())


class ArcticIceModelInitConditions:
    def __init__(self,
                 T_ocean_EArc: float = 1.,
                 T_ocean_WArc: float = 1.,
                 S_ocean_EArc: float = 34.,
                 S_ocean_WArc: float = 34.,
                 h_ice_EArc: float = 1.5,
                 h_ice_WArc: float = 1,
                 SIC_EArc: float = 0.75,
                 SIC_WArc: float = 0.55,
                 T_icesurface_EArc: float = -20.,
                 T_icesurface_WArc: float = -20.,
                 T_icebottom_EArc: float = -1.85,
                 T_icebottom_WArc: float = -1.85,
                 ice_type_EArc: int = 2,
                 ice_type_WArc: int = 2,
                 ):
        """
        Initial conditions for the ocean and sea ice variables.

        :param T_ocean_EArc: Initial ocean temperature in the east Arctic [degC]
        :param T_ocean_WArc: Initial ocean temperature in the west Arctic [degC]
        :param S_ocean_EArc: Initial salinity in the east Arctic [ppt]
        :param S_ocean_WArc: Initial salinity in the west Arctic [ppt]
        :param h_ice_EArc: Initial sea ice thickness in the east Arctic [m]
        :param h_ice_WArc: Initial sea ice thickness in the west Arctic [m]
        :param SIC_EArc: Initial sea ice concentration in the east Arctic
        :param SIC_WArc: Initial sea ice concentration in the west Arctic
        :param T_icesurface_EArc: Initial sea ice surface temperature in the east Arctic [degC]
        :param T_icesurface_WArc: Initial sea ice surface temperature in the west Arctic [degC]
        :param T_icebottom_EArc: Initial ice bottom temperature in the east Arctic [degC]
        :param T_icebottom_WArc: Initial ice bottom temperature in the west Arctic [degC]
        :param ice_type_EArc: Initial ice type for east arctic; 2 = Multi-year ice, 1 = First-year ice
        :param ice_type_WArc: Initial ice type for west arctic; 2 = Multi-year ice, 1 = First-year ice
        """
        if ice_type_EArc not in [1, 2]:
            raise ValueError(
                f"ice_type_EArc must be 1 or 2. Got {ice_type_EArc}.")
        if ice_type_WArc not in [1, 2]:
            raise ValueError(
                f"ice_type_WArc must be 1 or 2. Got {ice_type_WArc}.")

        self.T_ocean_EArc = T_ocean_EArc
        self.T_ocean_WArc = T_ocean_WArc
        self.S_ocean_EArc = S_ocean_EArc
        self.S_ocean_WArc = S_ocean_WArc
        self.h_ice_EArc = h_ice_EArc
        self.h_ice_WArc = h_ice_WArc
        self.SIC_EArc = SIC_EArc
        self.SIC_WArc = SIC_WArc
        self.T_icesurface_EArc = T_icesurface_EArc
        self.T_icesurface_WArc = T_icesurface_WArc
        self.T_icebottom_EArc = T_icebottom_EArc
        self.T_icebottom_WArc = T_icebottom_WArc
        self.ice_type_EArc = ice_type_EArc
        self.ice_type_WArc = ice_type_WArc

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelInitConditions(**self.to_dict())

    def get_Tocean_init_conditions(self) -> tuple:
        """
        Return the ocean temperature initial conditions.

        :return: East and West Arctic boxes ocean temperatures.
        """
        return self.T_ocean_EArc, self.T_ocean_WArc

    def get_Tice_init_conditions(self) -> tuple:
        """
        Return the surface and bottom sea ice temperature initial conditions.

        :return: East and West Arctic boxes surface and bottom sea ice temperatures.
        """
        return self.T_icesurface_EArc, self.T_icesurface_WArc, self.T_icebottom_EArc, self.T_icebottom_WArc

    def get_Socean_init_conditions(self) -> tuple:
        """
        Return the ocean salinity initial conditions.

        :return: East and West Arctic boxes ocean salinity.
        """
        return self.S_ocean_EArc, self.S_ocean_WArc

    def get_Ice_init_conditions(self) -> tuple:
        """
        Return the sea ice thickness, sea ice concentration (SIC), and ice type initial conditions.

        :return: East and West Arctic boxes sea ice thickness, sea ice concentration (SIC), and ice type.
        """
        return self.h_ice_EArc, self.h_ice_WArc, self.SIC_EArc, self.SIC_WArc, self.ice_type_EArc, self.ice_type_WArc


class ArcticIceModelParameters:
    def __init__(self,
                 albedo_ice: float = 0.6,
                 albedo_ocean: float = 0.06,
                 S_newice: float = 14,
                 cloudcover: float = 0.832,
                 mixing_coefficient: float = 1000,
                 current_NAtl: float = 3.5e6,
                 current_NPac: float = 1.0e6,
                 T_NAtl: float = 4,
                 S_NAtl: float = 35.06,
                 T_NPac: float = 5.19,
                 S_NPac: float = 33.83,
                 T_deep: float = -1.85,
                 S_deep: float = 34.5,
                 Kv: float = 1e-5
                 ):
        """
        Model parameters used by both boxes.

        :param albedo_ice = Ice albedo
        :param albedo_ocean = Ocean albedo
        :param S_newice: Salinity of new ice [ppt]
        :param cloudcover: Cloud cover
        :param mixing_coefficient: Mixing coefficient [m2/s]
        :param current_NAtl: Current from North Atlantic [m3/s]
        :param current_NPac: Current to/from North Pacific [m3/s]
        :param T_NAtl: Temperature in the north Atlantic box [degC]
        :param S_NAtl: Salinity in the north Atlantic box [ppt]
        :param T_NPac: Temperature in the north Pacific box [degC]
        :param S_NPac: Salinity in the north Pacific box [ppt]
        :param T_deep: Temperature in the deep box [degC]
        :param S_deep: Salinity in the deep box [ppt]
        :param Kv: Vertical diffusion coeffieicnt [m2/s]
        """
        self.albedo_ice = albedo_ice
        self.albedo_ocean = albedo_ocean
        self.S_newice = S_newice
        self.cloudcover = cloudcover
        self.mixing_coefficient = mixing_coefficient
        self.current_NAtl = current_NAtl
        self.current_NPac = current_NPac
        self.T_NAtl = T_NAtl
        self.S_NAtl = S_NAtl
        self.T_NPac = T_NPac
        self.S_NPac = S_NPac
        self.T_deep = T_deep
        self.S_deep = S_deep
        self.Kv = Kv

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelParameters(**self.to_dict())


class ArcticIceModelParametersEastArctic:
    def __init__(self,
                 k_ice_mean: float = 2.115,
                 k_ice_amp: float = -0.025,
                 T_atm: np.ndarray = np.full((1, ), np.nan, dtype=np.float64),
                 T_atm_mean: float = -12.8,
                 T_atm_amp: float = -14.6,
                 dewpt_atm_mean: float = -10.095,
                 dewpt_atm_amp: float = -16.205,
                 dewpt_ice_mean: float = -9.05,
                 dewpt_ice_amp: float = -16.95,
                 river_monthly: np.ndarray = np.array(
                     [0.0245, 0.0226, 0.0226, 0.0207, 0.0829, 0.2262, 0.1150, 0.0867, 0.0716, 0.0509, 0.0302, 0.0264]),
                 precip_monthly: np.ndarray = np.array(
                     [5, 5, 5, 5, 8, 12, 20, 23, 30, 25, 14, 10]),
                 shortwave_mean: float = 50,
                 shortwave_amp: float = 125,
                 U10m_atm: float = 5,
                 Lice_daily: float = 0.0,
                 Lice_length: float = 0.0,
                 Eice_daily: float = 0.0,
                 add_h_ice_intervention_monthly: np.ndarray = np.array(
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 T_atm_noise_scale=0.0
                 ):
        """
        Model parameters specific to the East box.

        :param k_ice_mean: Thermal conductivity of ice seasonal mean [W m^-1 K^-1]
        :param k_ice_amp: Thermal conductivity of ice seasonal amplitude [W m^-1 K^-1]
        :param T_atm: Air temperature time series [degC]. If set, this replaces the seasonal cycle that is computed in the model.
        :param T_atm_mean: Air temperature seasonal mean [degC]
        :param T_atm_amp: Air temperature seasonal amplitude [degC]
        :param dewpt_atm_mean: Air dewpoint seasonal mean
        :param dewpt_atm_amp: Air dewpoint seasonal amplitude
        :param dewpt_ice_mean: Dewpoint over ice seasonal mean
        :param dewpt_ice_amp: Dewpoint over ice seasonal amplitude
        :param river_monthly: Monthly river discharge [Sv] (converted to m3/s in arctic_ice_model.py)
        :param precip_monthly: Monthly precipitation [mm] (convert to m/s in arctic_ice_model.py)
        :param shortwave_mean: Shortwave flux seasonal mean [W m-2]
        :param shortwave_amp: Shortwave flux seasonal amplitude [W m-2]
        :param U10m_atm: 10m wind speed in the atmosphere [m/s]
        :param Lice_daily: flux of ice from land into EArctic
        :param Lice_length: length of land boundary with ice
        :param Eice_daily: export of ice from Arctic to North Pacific/Atlantic
        :param add_h_ice_intervention_monthly: Monthly ice added as part of intervention [cm] (converted to m/s in
            arctic_ice_model.py)
        :param T_atm_noise_scale: Spread (standard deviation) of the distribution for generating noise for T_atm, default is 0.0 (no noise), must be non-negative
        """
        self.k_ice_mean = k_ice_mean
        self.k_ice_amp = k_ice_amp
        self.T_atm = T_atm
        self.T_atm_mean = T_atm_mean
        self.T_atm_amp = T_atm_amp
        self.dewpt_atm_mean = dewpt_atm_mean
        self.dewpt_atm_amp = dewpt_atm_amp
        self.dewpt_ice_mean = dewpt_ice_mean
        self.dewpt_ice_amp = dewpt_ice_amp
        self.river_monthly = river_monthly
        self.precip_monthly = precip_monthly
        self.shortwave_mean = shortwave_mean
        self.shortwave_amp = shortwave_amp
        self.U10m_atm = U10m_atm
        self.Lice_daily = Lice_daily
        self.Lice_length = Lice_length
        self.Eice_daily = Eice_daily
        self.add_h_ice_intervention_monthly = add_h_ice_intervention_monthly
        self.T_atm_noise_scale = T_atm_noise_scale

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelParametersEastArctic(**self.to_dict())


class ArcticIceModelParametersWestArctic:
    def __init__(self,
                 k_ice_mean: float = 1.985,
                 k_ice_amp: float = -0.105,
                 T_atm: np.ndarray = np.full((1, ), np.nan, dtype=np.float64),
                 T_atm_mean: float = -10.5,
                 T_atm_amp: float = -13.7,
                 dewpt_atm_mean: float = -10.095,
                 dewpt_atm_amp: float = -16.205,
                 dewpt_ice_mean: float = -9.05,
                 dewpt_ice_amp: float = -16.95,
                 river_monthly: np.ndarray = np.array(
                     [0.0052, 0.0052, 0.0052, 0.0052, 0.0182, 0.0260, 0.0221, 0.0169, 0.0140, 0.0114, 0.0065, 0.0052]),
                 precip_monthly: np.ndarray = np.array(
                     [5, 5, 5, 5, 8, 12, 20, 23, 30, 25, 14, 10]),
                 shortwave_mean: float = 50,
                 shortwave_amp: float = 125,
                 U10m_atm: float = 5,
                 Lice_daily: float = 0.0,
                 Lice_length: float = 0.0,
                 Eice_daily: float = 0.0,
                 add_h_ice_intervention_monthly: np.ndarray = np.array(
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 T_atm_noise_scale=0.0
                 ):
        """
        Model parameters specific to the West box.

        :param k_ice_mean: Thermal conductivity of ice seasonal mean [W m^-1 K^-1]
        :param k_ice_amp: Thermal conductivity of ice seasonal amplitude [W m^-1 K^-1]
        :param T_atm: Air temperature time series [degC]. If set, this replaces the seasonal cycle that is computed in the model.
        :param T_atm_mean: Air temperature seasonal mean [degC]
        :param T_atm_amp: Air temperature seasonal amplitude [degC]
        :param dewpt_atm_mean: Air dewpoint seasonal mean
        :param dewpt_atm_amp: Air dewpoint seasonal amplitude
        :param dewpt_ice_mean: Dewpoint over ice seasonal mean
        :param dewpt_ice_amp: Dewpoint over ice seasonal amplitude
        :param river_monthly: Monthly river discharge [Sv] (converted to m3/s in arctic_ice_model.py)
        :param precip_monthly: Monthly precipitation [mm] (convert to m/s in arctic_ice_model.py)
        :param shortwave_mean: Shortwave flux seasonal mean [W m-2]
        :param shortwave_amp: Shortwave flux seasonal amplitude [W m-2]
        :param U10m_atm: 10m wind speed in the atmosphere [m/s]
        :param Lice_daily: flux of ice from land into EArctic
        :param Lice_length: length of land boundary with ice
        :param Eice_daily: export of ice from Arctic to North Pacific/Atlantic
        :param add_h_ice_intervention_monthly: Monthly ice added as part of intervention [cm] (converted to m/s in
            arctic_ice_model.py)
        :param T_atm_noise_scale: Spread (standard deviation) of the distribution for generating noise for T_atm, default is 0.0 (no noise), must be non-negative
        """
        self.k_ice_mean = k_ice_mean
        self.k_ice_amp = k_ice_amp
        self.T_atm = T_atm
        self.T_atm_mean = T_atm_mean
        self.T_atm_amp = T_atm_amp
        self.dewpt_atm_mean = dewpt_atm_mean
        self.dewpt_atm_amp = dewpt_atm_amp
        self.dewpt_ice_mean = dewpt_ice_mean
        self.dewpt_ice_amp = dewpt_ice_amp
        self.river_monthly = river_monthly
        self.precip_monthly = precip_monthly
        self.shortwave_mean = shortwave_mean
        self.shortwave_amp = shortwave_amp
        self.U10m_atm = U10m_atm
        self.Lice_daily = Lice_daily
        self.Lice_length = Lice_length
        self.Eice_daily = Eice_daily
        self.add_h_ice_intervention_monthly = add_h_ice_intervention_monthly
        self.T_atm_noise_scale = T_atm_noise_scale

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelParametersWestArctic(**self.to_dict())


class ArcticIceModelTimeStep:
    def __init__(self,
                 n_years: int = 50,
                 seasonal_cycle_period: float = 1,
                 seasonal_cycle_start_step: int = 0,
                 days_per_year: int = 365
                 ):
        """
        Model time step variables.

        :param n_years: Number of years to run the model
        :param seasonal_cycle_period: Period of the seasonal cycle (1 year)
        :param seasonal_cycle_start_step: On what step in the seasonal cycle to start (0-indexed)
        :param days_per_year: Number of days in one year.
        """
        self.n_years = n_years
        self.seasonal_cycle_period = seasonal_cycle_period
        self.seasonal_cycle_start_step = seasonal_cycle_start_step
        self.days_per_year = days_per_year

    def to_dict(self) -> dict:
        """
        Return a copy of the instance attributes as a dictionary.

        :return: Dictionary of attribute values.
        """
        return deepcopy(self.__dict__)

    def copy(self):
        """
        Create a copy of this instance.

        :return: New instance with the same values.
        """
        return ArcticIceModelTimeStep(**self.to_dict())


def dict_from_arctic_ice_model_args(box_dimensions: ArcticIceModelBoxDimensions,
                                    init_conditions: ArcticIceModelInitConditions,
                                    box_params: ArcticIceModelParameters,
                                    box_params_EArc: ArcticIceModelParametersEastArctic,
                                    box_params_WArc: ArcticIceModelParametersWestArctic,
                                    time_step: ArcticIceModelTimeStep) -> dict:
    """
    Given box model arguments, convert them into a dictionary with box model paramter names as keys and the 
        parameter values as the dictionary values.
    Useful for running the function to store the parameters and output to a netcdf.

    :param box_dimensions: (ArcticIceModelBoxDimensions) Parameters for the dimensions of the box model
    :param init_conditions: (ArcticIceModelInitConditions) Box model initial conditions
    :param box_params: (ArcticIceModelParameters) Model parameters that are the same for both boxes
    :param box_params_EArc: (ArcticIceModelParametersEastArctic) Model parameters for the East Arctic box
    :param box_params_WArc: (ArcticIceModelParametersWestArctic) Model parameters for the West Arctic box
    :param time_step: (ArcticIceModelTimeStep) Time step settings
    :return: (dict) Dictionary of box model parameters
    """
    d = {}
    d["box_dimensions"] = box_dimensions.to_dict()
    d["init_conditions"] = init_conditions.to_dict()
    d["box_params"] = box_params.to_dict()
    d["box_params_EArc"] = box_params_EArc.to_dict()
    d["box_params_WArc"] = box_params_WArc.to_dict()
    d["time_step"] = time_step.to_dict()
    return d
