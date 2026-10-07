# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import netCDF4
import numpy as np
from datetime import datetime

from .arctic_ice_model_constants import Constants

EArc_idx, WArc_idx = Constants.EArc_idx, Constants.WArc_idx


def save_run_to_netcdf(
        args_dims: dict,
        args_init_conditions: dict,
        args_params: dict,
        args_params_EArc: dict,
        args_params_WArc: dict,
        args_time_step: dict,
        results_data: dict,
        results_to_save: list | None = None,
        nc_filename: str = './arctic_ice_box_model_results.nc',
        run_info: str | None = None
) -> None:
    """
    Save a run of the Arctic Ice Box model data to a NetCDF file.
    
    :param args_dims: (dict) Dictionary object with box dimensions for model setup
    :param args_init_conditions: (dict) Dictionary object with initial conditions for model setup
    :param args_params: (dict) Dictionary object with parameters for model setup
    :param args_paramsEArc: (dict) Dictionary object with East Arctic parameters for model setup
    :param args_paramsWArc: (dict) Dictionary object with West Arctic parameters for model setup
    :param args_time_setp: (dict) Dictionary object with time step variables for model setup
    :param results_data: (dict) Dictionary object with the model results
    :param results_to_save: (list) Optional, list of results to save in the netCDF
    :param nc_filename: (str) Path to the NetCDF file. If one is not present, create a new one at this location
    :param run_info: (str) Optional, information about the run
    """
    print(f'Saving results to {nc_filename}')
 
    # Create the netCDF file
    root_group = netCDF4.Dataset(nc_filename, 'w')
      
    # Attributes
    root_group.description = "Arctic Sea Ice Box Model"
    root_group.history = (
        "File created "
        + datetime.now().astimezone().strftime("%d %b %Y %H:%M:%S %Z")
    )
    root_group.run_information = run_info

    # Create time step variable
    time_steps = np.arange(0, results_data['h_ice'].shape[1], 1)

    # Dimensions
    time_dim = root_group.createDimension("time", time_steps.shape[0])
    param_dim = root_group.createDimension("parameter", )
    param_monthly_dim = root_group.createDimension("monthly parameter", )

    # Create variables
    time_steps_var = root_group.createVariable("time_steps", "f8", ("time", ))

    # Write box dimensions to nc file
    for name, value in args_dims.items():
        save_name = "dims_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "box dimension"

    # Write initial conditions to nc file
    for name, value in args_init_conditions.items():
        save_name = "init_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "initial condition"

    # Write parameters to nc file
    for name, value in args_params.items():
        save_name = "params_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "parameter"

    # Write East Arctic parameters to nc file
    for name, value in args_params_EArc.items():
        save_name = "paramsEArc_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "East Arctic parameter"

    # Write West Arctic parameters to nc file
    for name, value in args_params_WArc.items():
        save_name = "paramsWArc_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "West Arctic parameter"

    # Write time step variables to nc file
    for name, value in args_time_step.items():
        save_name = "timestep_" + name
        arr = np.asarray(value)

        if arr.ndim == 1 and arr.shape[0] == root_group.dimensions["time"].size: 
            var = root_group.createVariable(save_name, "f8", ("time",))
        elif arr.ndim == 1 and arr.shape[0] == 12:
            var = root_group.createVariable(save_name, "f8", ("monthly parameter", ))
        else:
            var = root_group.createVariable(save_name, "f8", ("parameter", ))
        var[:] = arr
        var.type = "time step parameter"

    # Write results to nc file
    if results_to_save is None:
        results_to_save = ["h_ice", "SIC", "ice_type"]
    elif not isinstance(results_to_save, list):
            raise TypeError("results_to_save must be a list")

    for name in results_to_save:
        if name in results_data:
            # East Arctic
            save_name = name + "_EArc"
            var = root_group.createVariable(save_name, "f8", ("time", ))
            var[:] = results_data[name][EArc_idx]
            var.type = "result"

            # West Arctic
            save_name = name + "_WArc"
            var = root_group.createVariable(save_name, "f8", ("time", ))
            var[:] = results_data[name][WArc_idx]
            var.type = "result"

    root_group.close()
    