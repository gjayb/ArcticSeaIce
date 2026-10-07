# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import numpy as np
import gsw
from typing import Tuple

from .arctic_ice_model_constants import Constants
from .arctic_ice_model_args import (
    ArcticIceModelBoxDimensions,
    ArcticIceModelInitConditions,
    ArcticIceModelParameters,
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic,
    ArcticIceModelTimeStep
)
from .arctic_ice_model_output import ArcticIceOutput, ArcticOceanOutput, ArcticFluxOutput, ArcticCurrentsOutput
from .arctic_ice_model_flux_functions import compute_heat_fluxes, compute_salt_fluxes, compute_freshwater_fluxes
from .arctic_ice_model_ice_functions import compute_new_ice_temperatures, compute_ice_thickness_concentration
from .arctic_ice_model_utils import get_daily_ice_intervention, replace_nan_with_interp

# Index for each box
EArc_idx, WArc_idx = Constants.EArc_idx, Constants.WArc_idx
NUM_BOXES = 2

def arctic_ice_model(box_dims: ArcticIceModelBoxDimensions,
                     init_conditions: ArcticIceModelInitConditions,
                     params: ArcticIceModelParameters,
                     paramsEArc: ArcticIceModelParametersEastArctic,
                     paramsWArc: ArcticIceModelParametersWestArctic,
                     time_step: ArcticIceModelTimeStep,
                     ) -> Tuple[ArcticIceOutput, ArcticOceanOutput, ArcticFluxOutput, ArcticCurrentsOutput]:
    """
    Run sea ice box model.
    
    box_dims: Dimensions (area, lengths) of the boxes for this run
    init_conditions: Initial conditions for T_ocean, S_ocean, T_icesurface, T_icebottom, h_ice, SIC
    params: Model parameters that are common to all boxes
    paramsEArc: Model parameters specifically for the east Arctic
    paramsWArc: Model parameters specifically for the west Arctic
    """
    # Time step and related variables
    n_steps = time_step.n_years * time_step.days_per_year
    assert n_steps.is_integer()
    n_steps = int(n_steps)
    time_step_size_in_years = 1 / time_step.days_per_year
    dt = time_step.days_per_year * Constants.seconds_per_day * time_step_size_in_years
    seasonal_cycle_period = time_step.seasonal_cycle_period

    # Initial conditions
    h_ice_EArc0, h_ice_WArc0, SIC_EArc0, SIC_WArc0, ice_type_EArc0, ice_type_WArc0 = init_conditions.get_Ice_init_conditions()
    T_ocean_EArc0, T_ocean_WArc0 = init_conditions.get_Tocean_init_conditions()
    S_ocean_EArc0, S_ocean_WArc0 = init_conditions.get_Socean_init_conditions()
    (T_icesurface_EArc0, T_icesurface_WArc0, T_icebottom_EArc0,
     T_icebottom_WArc0) = init_conditions.get_Tice_init_conditions()

    # Initialize recorded variables
    h_ice = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    SIC = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    ice_type = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    T_ocean = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    S_ocean = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    T_icesurface = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    T_icebottom = np.full((n_steps + 1, NUM_BOXES), np.nan, dtype=np.float64)
    S_ice = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    density_ice = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    T_atm = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    T_ocean_freezing = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_sw = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_lw = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_sh = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_lh = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_b = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_up = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_bd = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_r = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_p = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_lhfresh = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    Fheat_mix_EArc_to_WArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_mix_WArc_to_EArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_EArc_to_WArc_forEArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_EArc_to_WArc_forWArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_NAtl_to_EArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_EArc_to_NPac = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_NPac_to_WArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_cur_WArc_to_NAtl = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_upw_EArc_to_deep = np.full(n_steps, np.nan, dtype=np.float64)
    Fheat_upw_WArc_to_deep = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_mix_EArc_to_WArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_mix_WArc_to_EArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_EArc_to_WArc_forEArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_EArc_to_WArc_forWArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_NAtl_to_EArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_EArc_to_NPac = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_NPac_to_WArc = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_cur_WArc_to_NAtl = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_upw_EArc_to_deep = np.full(n_steps, np.nan, dtype=np.float64)
    Fsalt_upw_WArc_to_deep = np.full(n_steps, np.nan, dtype=np.float64)
    F_heat_currents = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    F_salt_currents = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    FDD = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    TDD = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    ice_volume = np.full((n_steps, NUM_BOXES), np.nan, dtype=np.float64)
    add_h_ice = np.full((n_steps, NUM_BOXES), 0, dtype=np.float64)
    add_h_ice_intervention = np.full((n_steps, NUM_BOXES), 0, dtype=np.float64)

    # Set initial values
    h_ice[0, :] = np.array([h_ice_EArc0, h_ice_WArc0])
    SIC[0, :] = np.array([SIC_EArc0, SIC_WArc0])
    ice_type[0, :] = np.array([ice_type_EArc0, ice_type_WArc0])
    T_ocean[0, :] = np.array([T_ocean_EArc0, T_ocean_WArc0])
    S_ocean[0, :] = np.array([S_ocean_EArc0, S_ocean_WArc0])
    T_icesurface[0, :] = np.array([T_icesurface_EArc0, T_icesurface_WArc0])
    T_icebottom[0, :] = np.array([T_icebottom_EArc0, T_icebottom_WArc0])

    # User has the option to set a time series for T_atm instead of using a modeled seasonal cycle.
    # paramsBOX.T_atm is shape (1, ) filled with nan by default; it will be filled with values
    # if the user is using this instead of T_atm_mean and T_atm_amp.
    if ~np.isnan(paramsEArc.T_atm[0]):
        T_atm[:, EArc_idx] = paramsEArc.T_atm
        compute_T_atm_EArc = False
    else:
        compute_T_atm_EArc = True

    if ~np.isnan(paramsWArc.T_atm[0]):
        T_atm[:, WArc_idx] = paramsWArc.T_atm
        compute_T_atm_WArc = False
    else:
        compute_T_atm_WArc = True

    # Initialize variables
    FDD_startstep_EArc = 0
    FDD_startstep_WArc = 0
    TDD_counter_EArc = 0
    TDD_counter_WArc = 0

    # Loop time steps
    for step in range(0, n_steps):        
        # Interventions
        add_h_ice_intervention[step, EArc_idx] = get_daily_ice_intervention(step, paramsEArc, time_step.days_per_year)
        add_h_ice_intervention[step, WArc_idx] = get_daily_ice_intervention(step, paramsWArc, time_step.days_per_year)

        # Compute seasonal cycle variables
        cycle_step = time_step.seasonal_cycle_start_step + step
        k_ice_EArc = paramsEArc.k_ice_mean + paramsEArc.k_ice_amp * np.sin(
            2 * np.pi * ((cycle_step * time_step_size_in_years) / seasonal_cycle_period))
        k_ice_WArc = paramsWArc.k_ice_mean + paramsWArc.k_ice_amp * np.sin(
            2 * np.pi * ((cycle_step * time_step_size_in_years) / seasonal_cycle_period))

        # Either compute seasonal cycle of T_atm or use user's time series 
        if compute_T_atm_EArc:
            noise = np.random.normal(loc=0, scale=paramsEArc.T_atm_noise_scale)
            T_atm[step, EArc_idx] = noise + paramsEArc.T_atm_mean + paramsEArc.T_atm_amp * np.cos(
                2 * np.pi * (((cycle_step - 30) * time_step_size_in_years) / seasonal_cycle_period))
        else:
            # replace any nans in user's input by interpolating 
            if np.isnan(T_atm[step, EArc_idx]):
                T_atm[step, EArc_idx] = replace_nan_with_interp(step, EArc_idx, T_atm)

        if compute_T_atm_WArc:
            noise = np.random.normal(loc=0, scale=paramsWArc.T_atm_noise_scale)
            T_atm[step, WArc_idx] = noise + paramsWArc.T_atm_mean + paramsWArc.T_atm_amp * np.cos(
                2 * np.pi * (((cycle_step - 30) * time_step_size_in_years) / seasonal_cycle_period))
        else:
            # replace any nans in user's input by interpolating 
            if np.isnan(T_atm[step, WArc_idx]):
                T_atm[step, WArc_idx] = replace_nan_with_interp(step, WArc_idx, T_atm)

        # Heat fluxes
        F_sw[step, EArc_idx], F_lw[step, EArc_idx], F_sh[step, EArc_idx], F_lh[step, EArc_idx] = compute_heat_fluxes(
            step, EArc_idx, paramsEArc, params, SIC, T_ocean, T_atm, time_step_size_in_years, seasonal_cycle_period,
            cycle_step)
        F_heat_EArc = F_sw[step, EArc_idx] - F_lw[step, EArc_idx] + F_sh[step, EArc_idx] + F_lh[step, EArc_idx]

        F_sw[step, WArc_idx], F_lw[step, WArc_idx], F_sh[step, WArc_idx], F_lh[step, WArc_idx] = compute_heat_fluxes(
            step, WArc_idx, paramsWArc, params, SIC, T_ocean, T_atm, time_step_size_in_years, seasonal_cycle_period,
            cycle_step)
        F_heat_WArc = F_sw[step, WArc_idx] - F_lw[step, WArc_idx] + F_sh[step, WArc_idx] + F_lh[step, WArc_idx]

        # Bottom heat flux
        if h_ice[step, EArc_idx] > 0.0:
            F_b[step, EArc_idx] = SIC[step, EArc_idx] * (
                    -(k_ice_EArc / h_ice[step, EArc_idx]) * (T_ocean[step, EArc_idx] - T_icebottom[step, EArc_idx]))
        else:
            F_b[step, EArc_idx] = 0.0

        if h_ice[step, WArc_idx] > 0.0:
            F_b[step, WArc_idx] = SIC[step, WArc_idx] * (
                    -(k_ice_WArc / h_ice[step, WArc_idx]) * (T_ocean[step, WArc_idx] - T_icebottom[step, WArc_idx]))
        else:
            F_b[step, WArc_idx] = 0.0

        # Salt fluxes
        F_up[step, EArc_idx], F_bd[step, EArc_idx], density_ice, S_ice = compute_salt_fluxes(step, EArc_idx, params,
                                                                                             h_ice, SIC, S_ocean, S_ice,
                                                                                             density_ice, ice_type, dt)
        F_salt_EArc = (F_bd[step, EArc_idx] + F_up[step, EArc_idx]) / 1000

        F_up[step, WArc_idx], F_bd[step, WArc_idx], density_ice, S_ice = compute_salt_fluxes(step, WArc_idx, params,
                                                                                             h_ice, SIC, S_ocean, S_ice,
                                                                                             density_ice, ice_type, dt)
        F_salt_WArc = (F_bd[step, WArc_idx] + F_up[step, WArc_idx]) / 1000

        # Freshwater fluxes
        (F_r[step, EArc_idx], F_p[step, EArc_idx],
         F_lhfresh[step, EArc_idx]) = compute_freshwater_fluxes(step,
                                                                EArc_idx,
                                                                paramsEArc,
                                                                T_atm,
                                                                box_dims.area_EArc,
                                                                F_lh,
                                                                time_step.days_per_year)
        F_freshwater_EArc = ((F_p[step, EArc_idx] * (1 - SIC[step, EArc_idx])) + F_r[step, EArc_idx] + F_lhfresh[step, EArc_idx])

        (F_r[step, WArc_idx], F_p[step, WArc_idx],
         F_lhfresh[step, WArc_idx]) = compute_freshwater_fluxes(step,
                                                                WArc_idx,
                                                                paramsWArc,
                                                                T_atm,
                                                                box_dims.area_WArc,
                                                                F_lh,
                                                                time_step.days_per_year)
        F_freshwater_WArc = ((F_p[step, WArc_idx] * (1 - SIC[step, WArc_idx])) + F_r[step, WArc_idx] + F_lhfresh[step, WArc_idx])

        # Compute seawater variables
        SA = gsw.SA_from_SP(S_ocean[step, :], 0, 150, 80)
        cp_ocean = gsw.cp_t_exact(SA, T_ocean[step, :], 0)
        density_ocean = gsw.rho(SA, T_ocean[step, :], 0)

        SA = gsw.SA_from_SP(params.S_NAtl, 0, 150, 80)
        cp_ocean_NAtl = gsw.cp_t_exact(SA, params.T_NAtl, 0)
        density_ocean_NAtl = gsw.rho(SA, params.T_NAtl, 0)

        SA = gsw.SA_from_SP(params.S_NPac, 0, 150, 80)
        cp_ocean_NPac = gsw.cp_t_exact(SA, params.T_NPac, 0)
        density_ocean_NPac = gsw.rho(SA, params.T_NPac, 0)

        SA = gsw.SA_from_SP(S_ocean[step, :], 0, 150, 80) # for deep ocean S, use S_ocean
        cp_ocean_deep = gsw.cp_t_exact(SA, np.array([params.T_deep, params.T_deep]), 0)
        density_ocean_deep = gsw.rho(SA, np.array([params.T_deep, params.T_deep]), 0)

        # Compute diffusive upwelling for deep box
        M_upw_EArc = (params.Kv * box_dims.area_EArc) / box_dims.d_ocean
        M_upw_WArc = (params.Kv * box_dims.area_WArc) / box_dims.d_ocean

        # Compute currents/mixing of heat between boxes
        # East Box
        Fheat_mix_EArc_to_WArc[step] = (
                (((params.mixing_coefficient * box_dims.d_ocean * box_dims.Lx_east) / box_dims.Ly_east)
                 * cp_ocean[WArc_idx] * (-density_ocean[EArc_idx] * (T_ocean[step, EArc_idx] + 273.15)
                                         + density_ocean[WArc_idx] * (T_ocean[step, WArc_idx] + 273.15))
                 ) / box_dims.area_EArc
        )
        Fheat_cur_EArc_to_WArc_forEArc[step] = -(
                (params.current_NAtl - params.current_NPac) * cp_ocean[EArc_idx] * density_ocean[EArc_idx] * (
                    T_ocean[step, EArc_idx] + 273.15)) / box_dims.area_EArc
        Fheat_cur_NAtl_to_EArc[step] = (params.current_NAtl * cp_ocean_NAtl * density_ocean_NAtl * (
                params.T_NAtl + 273.15)) / box_dims.area_EArc
        Fheat_cur_EArc_to_NPac[step] = -(params.current_NPac * cp_ocean[EArc_idx] * density_ocean[EArc_idx] * (
                T_ocean[step, EArc_idx] + 273.15)) / box_dims.area_EArc
        Fheat_upw_EArc_to_deep[step] = (M_upw_EArc * cp_ocean_deep[EArc_idx] * (
                density_ocean_deep[EArc_idx] * (params.T_deep + 273.15) - density_ocean[EArc_idx] * (
                    T_ocean[step, EArc_idx] + 273.15))) / box_dims.area_EArc

        # West Box
        Fheat_mix_WArc_to_EArc[step] = (
                (((params.mixing_coefficient * box_dims.d_ocean * box_dims.Lx_east) / box_dims.Ly_east)
                 * cp_ocean[EArc_idx] * (-density_ocean[WArc_idx] * (T_ocean[step, WArc_idx] + 273.15)
                                         +
                                         density_ocean[EArc_idx] * (T_ocean[step, EArc_idx] + 273.15))
                 ) / box_dims.area_WArc
        )
        Fheat_cur_EArc_to_WArc_forWArc[step] = ((params.current_NAtl - params.current_NPac) * cp_ocean[EArc_idx] *
                                                density_ocean[EArc_idx] * (
                                                        T_ocean[step, EArc_idx] + 273.15)) / box_dims.area_WArc
        Fheat_cur_NPac_to_WArc[step] = (params.current_NPac * cp_ocean_NPac * density_ocean_NPac * (
                params.T_NPac + 273.15)) / box_dims.area_WArc
        Fheat_cur_WArc_to_NAtl[step] = -(params.current_NAtl * cp_ocean[WArc_idx] * density_ocean[WArc_idx] * (
                T_ocean[step, WArc_idx] + 273.15)) / box_dims.area_WArc
        Fheat_upw_WArc_to_deep[step] = (
                (M_upw_WArc * cp_ocean_deep[WArc_idx] * (density_ocean_deep[WArc_idx] * (params.T_deep + 273.15)
                                                         - density_ocean[WArc_idx] * (T_ocean[step, WArc_idx] + 273.15))
                 ) / box_dims.area_WArc)

        F_heat_currents[step, EArc_idx] = (
                Fheat_mix_EArc_to_WArc[step] + Fheat_cur_EArc_to_WArc_forEArc[step] + Fheat_cur_NAtl_to_EArc[step]
                + Fheat_cur_EArc_to_NPac[step] + Fheat_upw_EArc_to_deep[step]
        )
        F_heat_currents[step, WArc_idx] = (
                Fheat_mix_WArc_to_EArc[step] + Fheat_cur_EArc_to_WArc_forWArc[step] + Fheat_cur_NPac_to_WArc[step]
                + Fheat_cur_WArc_to_NAtl[step] + Fheat_upw_WArc_to_deep[step]
        )

        # Compute currents/mixing of salt between boxes
        # East Box
        Fsalt_mix_EArc_to_WArc[step] = (
                (((params.mixing_coefficient * box_dims.d_ocean * box_dims.Lx_east) / box_dims.Ly_east)
                 * (-density_ocean[EArc_idx] * S_ocean[step, EArc_idx]
                    + density_ocean[WArc_idx] * S_ocean[step, WArc_idx])
                 ) / box_dims.area_EArc
        )
        Fsalt_cur_EArc_to_WArc_forEArc[step] = (-(
                (params.current_NAtl - params.current_NPac) * density_ocean[EArc_idx] * S_ocean[step, EArc_idx])
                                                / box_dims.area_EArc)
        Fsalt_cur_NAtl_to_EArc[step] = (params.current_NAtl * density_ocean_NAtl * params.S_NAtl) / box_dims.area_EArc
        Fsalt_cur_EArc_to_NPac[step] = -(
                params.current_NPac * density_ocean[EArc_idx] * S_ocean[step, EArc_idx]) / box_dims.area_EArc
        Fsalt_upw_EArc_to_deep[step] = (M_upw_EArc * (
                density_ocean_deep[EArc_idx] * S_ocean[step, EArc_idx]
                - density_ocean[EArc_idx] * S_ocean[step, EArc_idx])
                                        ) / box_dims.area_EArc

        # West Box
        Fsalt_mix_WArc_to_EArc[step] = (
                (((params.mixing_coefficient * box_dims.d_ocean * box_dims.Lx_east) / box_dims.Ly_east)
                 * (-density_ocean[WArc_idx] * S_ocean[step, WArc_idx]
                    + density_ocean[EArc_idx] * S_ocean[step, EArc_idx])
                 ) / box_dims.area_WArc
        )
        Fsalt_cur_EArc_to_WArc_forWArc[step] = ((params.current_NAtl - params.current_NPac) * density_ocean[EArc_idx] *
                                                S_ocean[step, EArc_idx]) / box_dims.area_WArc
        Fsalt_cur_NPac_to_WArc[step] = (params.current_NPac * density_ocean_NPac * params.S_NPac) / box_dims.area_WArc
        Fsalt_cur_WArc_to_NAtl[step] = -(
                params.current_NAtl * density_ocean[WArc_idx] * S_ocean[step, WArc_idx]) / box_dims.area_WArc
        Fsalt_upw_WArc_to_deep[step] = (M_upw_WArc *
                                        (density_ocean_deep[WArc_idx] * S_ocean[step, WArc_idx]
                                         - density_ocean[WArc_idx] * S_ocean[step, WArc_idx])) / box_dims.area_WArc

        F_salt_currents[step, EArc_idx] = (
                Fsalt_mix_EArc_to_WArc[step] + Fsalt_cur_EArc_to_WArc_forEArc[step] + Fsalt_cur_NAtl_to_EArc[step]
                + Fsalt_cur_EArc_to_NPac[step] + Fsalt_upw_EArc_to_deep[step]
        )
        F_salt_currents[step, WArc_idx] = (
                Fsalt_mix_WArc_to_EArc[step] + Fsalt_cur_EArc_to_WArc_forWArc[step] + Fsalt_cur_NPac_to_WArc[step]
                + Fsalt_cur_WArc_to_NAtl[step] + Fsalt_upw_WArc_to_deep[step]
        )

        # Compute new salinity and temperatures
        previous_S_ocean_as_flux = S_ocean[step, EArc_idx] * density_ocean[EArc_idx] * box_dims.d_ocean
        S_ocean[step + 1, EArc_idx] = (previous_S_ocean_as_flux + (
                (-F_salt_EArc * SIC[step, EArc_idx]) + F_salt_currents[step, EArc_idx]) * dt) / (
                                              density_ocean[EArc_idx] * box_dims.d_ocean + F_freshwater_EArc * dt)

        previous_S_ocean_as_flux = S_ocean[step, WArc_idx] * density_ocean[WArc_idx] * box_dims.d_ocean
        S_ocean[step + 1, WArc_idx] = (previous_S_ocean_as_flux + (
                (-F_salt_WArc * SIC[step, WArc_idx]) + F_salt_currents[step, WArc_idx]) * dt) / (
                                              density_ocean[WArc_idx] * box_dims.d_ocean + F_freshwater_WArc * dt)

        previous_T_ocean_as_flux = (T_ocean[step, EArc_idx] + 273.15) * density_ocean[EArc_idx] * cp_ocean[
            EArc_idx] * box_dims.d_ocean
        T_ocean[step + 1, EArc_idx] = (((previous_T_ocean_as_flux + ((F_heat_EArc + (
                F_b[step, EArc_idx] * SIC[step, EArc_idx]) + F_heat_currents[step, EArc_idx]) * dt))
                                        / (density_ocean[EArc_idx] * cp_ocean[EArc_idx] * box_dims.d_ocean))) - 273.15
        T_ocean_freezing[step, EArc_idx] = gsw.CT_freezing(S_ocean[step + 1, EArc_idx], 0, 1)

        # If T_ocean < T_ocean_freezing, use excess heat to make ice, East Box
        if T_ocean[step + 1, EArc_idx] < T_ocean_freezing[step, EArc_idx]:
            dT = T_ocean_freezing[step, EArc_idx] - T_ocean[step + 1, EArc_idx]
            T_ocean[step + 1, EArc_idx] = T_ocean_freezing[step, EArc_idx]
            add_h_ice[step, EArc_idx] = add_h_ice[step, EArc_idx] + (dT * density_ocean[EArc_idx] * cp_ocean[EArc_idx] * box_dims.d_ocean / (
                    Constants.LI * density_ice[step, EArc_idx] * SIC[step, EArc_idx]))
            
        previous_T_ocean_as_flux = (T_ocean[step, WArc_idx] + 273.15) * density_ocean[WArc_idx] * cp_ocean[
            WArc_idx] * box_dims.d_ocean
        T_ocean[step + 1, WArc_idx] = (((previous_T_ocean_as_flux + ((F_heat_WArc * (1 - SIC[step, WArc_idx]) + (
                F_b[step, WArc_idx] * SIC[step, WArc_idx]) + F_heat_currents[step, WArc_idx]) * dt))
                                        / (density_ocean[WArc_idx] * cp_ocean[WArc_idx] * box_dims.d_ocean))) - 273.15
        T_ocean_freezing[step, WArc_idx] = gsw.CT_freezing(S_ocean[step + 1, WArc_idx], 0, 1)

        # If T_ocean < T_ocean_freezing, use excess heat to make ice, West Box
        if T_ocean[step + 1, WArc_idx] < T_ocean_freezing[step, WArc_idx]:
            dT = T_ocean_freezing[step, WArc_idx] - T_ocean[step + 1, WArc_idx]
            T_ocean[step + 1, WArc_idx] = T_ocean_freezing[step, WArc_idx]
            add_h_ice[step, WArc_idx] = add_h_ice[step, WArc_idx] + (dT * density_ocean[WArc_idx] * box_dims.d_ocean * cp_ocean[WArc_idx] / (
                    Constants.LI * density_ice[step, WArc_idx] * SIC[step, WArc_idx]))
        
        # Compute ice temperatures
        (T_icesurface[step + 1, EArc_idx],
         T_icebottom[step + 1, EArc_idx]
         ) = compute_new_ice_temperatures(step, EArc_idx, h_ice, T_atm, T_ocean_freezing, k_ice_EArc)
        
        (T_icesurface[step + 1, WArc_idx],
         T_icebottom[step + 1, WArc_idx]
         ) = compute_new_ice_temperatures(step, WArc_idx, h_ice, T_atm, T_ocean_freezing, k_ice_WArc)

        # Compute ice thickness and sea ice concentration
        # East box
        (h_ice,
         SIC,
         ice_type,
         FDD_startstep_EArc,
         TDD_counter_EArc
         ) = compute_ice_thickness_concentration(step,
                                                 EArc_idx,
                                                 paramsEArc,
                                                 params,
                                                 box_dims,
                                                 h_ice,
                                                 add_h_ice, SIC,
                                                 FDD,
                                                 FDD_startstep_EArc,
                                                 TDD,
                                                 TDD_counter_EArc,
                                                 F_heat_EArc,
                                                 T_atm,
                                                 T_ocean,
                                                 T_ocean_freezing,
                                                 density_ice,
                                                 ice_type,
                                                 add_h_ice_intervention,
                                                 time_step_size_in_years, 
                                                 seasonal_cycle_period, 
                                                 cycle_step, 
                                                 dt, 
                                                 time_step.days_per_year)

        # West box
        (h_ice,
         SIC,
         ice_type,
         FDD_startstep_WArc,
         TDD_counter_WArc
         ) = compute_ice_thickness_concentration(step,
                                                 WArc_idx,
                                                 paramsWArc,
                                                 params,
                                                 box_dims,
                                                 h_ice,
                                                 add_h_ice,
                                                 SIC,
                                                 FDD,
                                                 FDD_startstep_WArc,
                                                 TDD,
                                                 TDD_counter_WArc,
                                                 F_heat_WArc,
                                                 T_atm,
                                                 T_ocean,
                                                 T_ocean_freezing,
                                                 density_ice,
                                                 ice_type,
                                                 add_h_ice_intervention,
                                                 time_step_size_in_years, 
                                                 seasonal_cycle_period, 
                                                 cycle_step, 
                                                 dt, 
                                                 time_step.days_per_year)

    # Compute volume of ice
    ice_volume[:, EArc_idx] = SIC[:-1, EArc_idx] * h_ice[:-1, EArc_idx] * box_dims.area_EArc
    ice_volume[:, WArc_idx] = SIC[:-1, WArc_idx] * h_ice[:-1, WArc_idx] * box_dims.area_WArc

    # Remove extra data points where needed and reshape to have the columns be time
    h_ice = h_ice[1:].transpose() # Do not return initial condition
    SIC = SIC[1:].transpose() # Do not return initial condition
    ice_type = ice_type[1:].transpose() # Do not return initial condition
    ice_volume = ice_volume.transpose()
    T_icesurface = T_icesurface[1:].transpose() # Do not return initial condition
    T_icebottom = T_icebottom[1:].transpose() # Do not return initial condition
    T_ocean = T_ocean[1:].transpose() # Do not return initial condition
    S_ocean = S_ocean[1:].transpose() # Do not return initial condition
    T_atm = T_atm.transpose()
    F_sw = F_sw.transpose()
    F_lw = F_lw.transpose()
    F_sh = F_sh.transpose()
    F_lh = F_lh.transpose()
    F_b = F_b.transpose()
    F_up = F_up.transpose()
    F_bd = F_bd.transpose()
    F_r = F_r.transpose()
    F_p = F_p.transpose()
    F_lhfresh = F_lhfresh.transpose()

    # Variables to output
    ice_results = ArcticIceOutput(h_ice, SIC, ice_type, ice_volume, T_icesurface, T_icebottom)
    ocean_results = ArcticOceanOutput(T_ocean, S_ocean, T_atm)
    flux_results = ArcticFluxOutput(F_sw, F_lw, F_sh, F_lh, F_b, F_up, F_bd, F_r, F_p, F_lhfresh)
    currents_results = ArcticCurrentsOutput(Fheat_cur_NAtl_to_EArc, Fheat_cur_EArc_to_WArc_forEArc,
                                            Fheat_cur_EArc_to_WArc_forWArc, Fheat_cur_EArc_to_NPac,
                                            Fheat_cur_NPac_to_WArc, Fheat_cur_WArc_to_NAtl,
                                            Fheat_mix_EArc_to_WArc, Fheat_mix_WArc_to_EArc, Fheat_upw_EArc_to_deep,
                                            Fheat_upw_WArc_to_deep,
                                            Fsalt_cur_NAtl_to_EArc, Fsalt_cur_EArc_to_WArc_forEArc,
                                            Fsalt_cur_EArc_to_WArc_forWArc, Fsalt_cur_EArc_to_NPac,
                                            Fsalt_cur_NPac_to_WArc, Fsalt_cur_WArc_to_NAtl,
                                            Fsalt_mix_EArc_to_WArc, Fsalt_mix_WArc_to_EArc, Fsalt_upw_EArc_to_deep,
                                            Fsalt_upw_WArc_to_deep)

    return ice_results, ocean_results, flux_results, currents_results
