# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

"""
Example script to run the Arctic ice box model.

Usage
-----
1. Create the uv environment with `uv venv` and then `uv pip install -r requirements.txt`
2. Run the script:
    uv run example_run_arctic_ice_model.py
"""

import time
import argparse
import yaml
import ast
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from arctic_ice.arctic_ice_model import (
    arctic_ice_model,
    EArc_idx,
    WArc_idx,
)
from arctic_ice.arctic_ice_model_args import (
    ArcticIceModelBoxDimensions,
    ArcticIceModelInitConditions,
    ArcticIceModelParameters,
    ArcticIceModelParametersEastArctic,
    ArcticIceModelParametersWestArctic,
    ArcticIceModelTimeStep,
    dict_from_arctic_ice_model_args,
)
from arctic_ice.arctic_ice_model_storage import save_run_to_netcdf


@dataclass
class ModelConfig:
    box_dimensions: ArcticIceModelBoxDimensions = field(
        default_factory=ArcticIceModelBoxDimensions)
    init_conditions: ArcticIceModelInitConditions = field(
        default_factory=ArcticIceModelInitConditions)
    box_params: ArcticIceModelParameters = field(
        default_factory=ArcticIceModelParameters)
    box_params_EArc: ArcticIceModelParametersEastArctic = field(
        default_factory=ArcticIceModelParametersEastArctic)
    box_params_WArc: ArcticIceModelParametersWestArctic = field(
        default_factory=ArcticIceModelParametersWestArctic)
    time_step: ArcticIceModelTimeStep = field(
        default_factory=ArcticIceModelTimeStep)

    def to_dict(self) -> dict:
        return dict_from_arctic_ice_model_args(
            self.box_dimensions,
            self.init_conditions,
            self.box_params,
            self.box_params_EArc,
            self.box_params_WArc,
            self.time_step
        )


def update_config_with_settings(config: ModelConfig, values: dict):
    """
    Updates configuration with settings provided in a dictionary (from yaml file)
    """
    for section, params in values.items():
        if not hasattr(config, section):
            raise ValueError(f"Unknown config section '{section}'")

        obj = getattr(config, section)

        for name, value in params.items():
            if not hasattr(obj, name):
                raise ValueError(f"Unknown parameter '{section}.{name}'")

            if isinstance(getattr(obj, name), np.ndarray):
                value = np.array(value)
            setattr(obj, name, value)


def apply_cli_overrides(config: ModelConfig, overrides: list[str]):
    """
    Updates configuration with manual overrides provided as command line options
    """
    if len(overrides) % 2 != 0:
        raise ValueError(
            "Overrides must be supplied as '--section.parameter value'")

    for key, value in zip(overrides[0::2], overrides[1::2]):

        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument '{key}'")

        key = key[2:]

        try:
            section, name = key.split(".", 1)
        except ValueError:
            raise ValueError(
                f"Override '{key}' must have the form section.parameter"
            )

        if not hasattr(config, section):
            raise ValueError(f"Unknown config section '{section}'")

        obj = getattr(config, section)

        if not hasattr(obj, name):
            raise ValueError(f"Unknown parameter '{section}.{name}'")

        current = getattr(obj, name)

        # Convert to same type as existing value
        if isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, np.ndarray):
            value = np.asarray(ast.literal_eval(value), dtype=current.dtype)
        else:
            value = type(current)(value)

        setattr(obj, name, value)


def yamlify(obj):
    """
    Casts objects to native python types to make compatible with yaml saving/loading
    """
    if isinstance(obj, dict):
        return {k: yamlify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [yamlify(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [yamlify(v) for v in obj.tolist()]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj


def parse_args():
    parser = argparse.ArgumentParser(description='Run Arctic ice box model.')
    parser.add_argument('--config', default=None, type=Path,
                        help='A configuration file specifying parameters to run model with.')
    parser.add_argument('--results_dir', default=Path('.'), type=Path,
                        help='Where to store results. Default is current working directory.')
    parser.add_argument('--nc_filename', default='results.nc',
                        type=str, help='Name of file to save results to.')
    parser.add_argument('--run_info', default='example model run',
                        type=str, help='Information on the model run.')
    parser.add_argument('--save_results', default=True, dest='save_results',
                        action=argparse.BooleanOptionalAction, help='If True will save results. Defaults to True.')
    parser.add_argument('--plot_results', default=True, dest='plot_results',
                        action=argparse.BooleanOptionalAction, help='If True will plot results. Defaults to True.')
    parser.add_argument('--save_config', default=True, dest='save_config', action=argparse.BooleanOptionalAction,
                        help='If True will save configuration for run. Defaults to True.')
    args, overrides = parser.parse_known_args()
    return args, overrides


def main(args, overrides):
    # Set directory to store results
    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load model parameters
    config = ModelConfig()

    # Set parameter values with provided configuration file
    if args.config is not None:
        with open(args.config, 'r') as f:
            new_settings = yaml.safe_load(f)
            update_config_with_settings(config, new_settings)

    # Set parameter values with CLI overrides
    apply_cli_overrides(config, overrides)

    # Run model
    start = time.time()
    ice_results, ocean_results, flux_results, currents_results = arctic_ice_model(
        config.box_dimensions,
        config.init_conditions,
        config.box_params,
        config.box_params_EArc,
        config.box_params_WArc,
        config.time_step
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    # Unpack results
    h_ice, SIC, ice_type, ice_volume, T_icesurface, T_icebottom = ice_results.unpack()
    T_ocean, S_ocean, T_atm = ocean_results.unpack()

    # Save config
    if args.save_config:
        with open(args.results_dir / 'config.yaml', 'w') as f:
            yaml.safe_dump(yamlify(config.to_dict()), f)

    # Save results to netCDF
    if args.save_results:
        results_data = ice_results.to_dict() | ocean_results.to_dict()
        results_to_save = ['h_ice', 'SIC',
                           'ice_type', 'T_atm', 'T_ocean', 'S_ocean']

        save_run_to_netcdf(
            config.box_dimensions.to_dict(),
            config.init_conditions.to_dict(),
            config.box_params.to_dict(),
            config.box_params_EArc.to_dict(),
            config.box_params_WArc.to_dict(),
            config.time_step.to_dict(),
            results_data,
            results_to_save=results_to_save,
            nc_filename=Path.joinpath(args.results_dir, args.nc_filename),
            run_info=args.run_info
        )

    # Plot results
    if args.plot_results:
        from matplotlib import pyplot as plt
        timesteps = np.arange(1, config.time_step.n_years * 365 + 1, 1)
        xticks = timesteps[::365]
        xlabels = [i + 1 for i in range(0, len(xticks))]

        # Plot ice results
        fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(10, 5.5))
        ax1[0].plot(h_ice[EArc_idx], label='East Arctic')
        ax1[0].plot(h_ice[WArc_idx], label='West Arctic', linestyle='solid')
        ax1[0].set_title('Ice thickness [m]')
        ax1[1].plot(SIC[EArc_idx], label='East Arctic')
        ax1[1].plot(SIC[WArc_idx], label='West Arctic', linestyle='solid')
        ax1[1].set_title('SIC')
        ax1[1].set_ylim([-0.1, 1.1])
        ax1[2].plot(ice_type[EArc_idx], label='East Arctic')
        ax1[2].plot(ice_type[WArc_idx], label='West Arctic', linestyle='solid')
        ax1[2].set_title('Ice Type (2=MY, 1=FY, 0=No ice)')
        for a in ax1:
            a.grid()
            a.set_xticks(xticks, labels='')
            a.set_xlim([xticks[0], len(timesteps)])
            if a == ax1[-1]:
                a.set_xticks(xticks)
                a.set_xticklabels([xlabels[i] if i %
                                  2 == 0 else '' for i in range(len(xticks))])
        plt.xlabel('Year')
        ax1[0].legend(loc='upper right')
        plt.tight_layout()
        plot_name = 'ice_results.png'
        plt.savefig(Path.joinpath(args.results_dir, plot_name))
        plt.close()

        # plot ocean results
        fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(10, 5.5))
        ax2[0].plot(T_ocean[EArc_idx], label='East Arctic')
        ax2[0].plot(T_ocean[WArc_idx], label='West Arctic', linestyle='solid')
        ax2[0].axhline(y=-1.85, color='c', label='Freezing Temp')
        ax2[0].set_title('Ocean Temperature [degC]')
        ax2[1].plot(S_ocean[EArc_idx], label='East Arctic')
        ax2[1].plot(S_ocean[WArc_idx], label='West Arctic', linestyle='solid')
        ax2[1].set_title('Ocean Salinity')
        ax2[2].plot(T_atm[EArc_idx], label='East Arctic')
        ax2[2].plot(T_atm[WArc_idx], label='West Arctic', linestyle='solid')
        ax2[2].set_title('Atmosphere Temperature [degC]')
        for a in ax2:
            a.grid()
            a.set_xticks(xticks, labels='')
            a.set_xlim([xticks[0], len(timesteps)])
            if a == ax2[-1]:
                a.set_xticks(xticks)
                a.set_xticklabels([xlabels[i] if i %
                                  2 == 0 else '' for i in range(len(xticks))])
        plt.xlabel('Year')
        ax2[0].legend(loc='upper right')
        plt.tight_layout()
        plot_name = 'ocean_results.png'
        plt.savefig(Path.joinpath(args.results_dir, plot_name))
        plt.close()


if __name__ == "__main__":
    args, overrides = parse_args()
    main(args, overrides)
