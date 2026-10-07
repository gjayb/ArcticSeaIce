# Arctic Ice Model

**This is a reduced-order model to simulate annual sea ice growth and melt in the Arctic to study tipping states under different climate conditions. The Arctic basin is divided into two domains ("boxes"): the East Arctic and the West Arctic. The user can set several atmospheric, oceanic, and sea ice parameters to reflect different climate conditions. The model outputs sea ice thickness, concentration, and type as a function of time, as well as atmospheric and oceanic variables and heat and salt fluxes, if desired.** 

## Installation

This requires python version `3.13`. It may work in other versions but has only been tested with 3.13.
Install using `uv`.

```bash
uv sync --system-certs
```

## Running

To run the box model, use the provided run script:
```bash
uv run run_arctic_ice_model.py
```

By default this will run the arctic ice model, and save the following 3 artifacts of the run:
1. A netcdf file storing the results
2. Timeseries plots as png files of the ocean and ice variables
3. A configuration file of the settings the model was run with

You can disable any of these options with the options `--no-save_results`, `--no-plot_results`, and `--no-save_config`.
For example
```bash
uv run run_arctic_ice_model.py --no-save_results --no-plot_results
```

To save the results to a specific directory provide a results dir. For example:
```bash
uv run run_arctic_ice_model.py --results_dir model_run1
```

### Running with Custom Parameters

To run the ice model with custom parameters there are two options. 

The first thing you can do is provide a `yaml` file.
Any settings provided in the yaml file will override the defaults of the ice model.

You can also do this to recreate a previous ice model run as long as you save the configuration for that run.
```bash
uv run run_arctic_ice_model.py --results_dir model_run1_recreation --config model_run1/config.yaml
```

A configuration file containing the default settings is provided in `default_config.yaml`. As such,
the following two commands should produce identical results:
```bash
uv run run_arctic_ice_model.py --results_dir default1 
uv run run_arctic_ice_model.py --results_dir default2 --config default_config.yaml
```

The other way you can specify parameters is via the command line. The options must be of the format
`--section.parameter value`

For example, to set the `n_years` parameter of the `time_step` section of the ice model configuration to 100:
```bash
uv run run_arctic_ice_model.py --results_dir model_run2 --time_step.n_years 100
```

You may supply multiple command line overrides, there is no limit.

For arrays, they must be provided as valid python list literals, for example:
```bash
uv run run_arctic_ice_model.py --results_dir model_run3 --box_params_EArc.add_h_ice_intervention_monthly "[0,0,0,0,0,0,0,0,0,0,30,0]"
```

**Note: Command line options will override options given in a yaml file if you use both settings.**

All settings are provided below, broken out by section:

#### `box_dimensions`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `area_EArc` | Surface area of the East Arctic box (m²) | `5.95449672e12` |
| `area_WArc` | Surface area of the West Arctic box (m²) | `9.03862210e12` |
| `area_NGreen` | Surface area of the North Greenland ice sheet box (m²) | `1.18279185e12` |
| `area_SGreen` | Surface area of the South Greenland ice sheet box (m²) | `9.8596193e11` |
| `area_NAtl` | Surface area of the North Atlantic box (m²) | `2.2e13` |
| `area_NPac` | Surface area of the North Pacific box (m²) | `1.0e13` |
| `Lx_east` | Length of the East/West Arctic interface (m) | `6.4e6` |
| `Ly_east` | Width of the East Arctic box (m) | `1.93e5` |
| `Lx_west` | Width of the Bering Strait (m) | `8.2e4` |
| `Ly_west` | Width of the West Arctic box (m) | `1.0e6` |
| `d_ocean` | Ocean depth (m) | `100` |

#### `init_conditions`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `T_ocean_EArc` | Initial East Arctic ocean temperature (°C) | `1.0` |
| `T_ocean_WArc` | Initial West Arctic ocean temperature (°C) | `1.0` |
| `S_ocean_EArc` | Initial East Arctic ocean salinity (ppt) | `34.0` |
| `S_ocean_WArc` | Initial West Arctic ocean salinity (ppt) | `34.0` |
| `h_ice_EArc` | Initial East Arctic ice thickness (m) | `1.5` |
| `h_ice_WArc` | Initial West Arctic ice thickness (m) | `1.0` |
| `SIC_EArc` | Initial East Arctic sea ice concentration | `0.75` |
| `SIC_WArc` | Initial West Arctic sea ice concentration | `0.55` |
| `T_icesurface_EArc` | Initial East Arctic ice surface temperature (°C) | `-20.0` |
| `T_icesurface_WArc` | Initial West Arctic ice surface temperature (°C) | `-20.0` |
| `T_icebottom_EArc` | Initial East Arctic ice bottom temperature (°C) | `-1.85` |
| `T_icebottom_WArc` | Initial West Arctic ice bottom temperature (°C) | `-1.85` |
| `ice_type_EArc` | Initial East Arctic ice type (`2` = multi-year, `1` = first-year) | `2` |
| `ice_type_WArc` | Initial West Arctic ice type (`2` = multi-year, `1` = first-year) | `2` |

#### `box_params`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `albedo_ice` | Ice albedo | `0.6` |
| `albedo_ocean` | Ocean albedo | `0.06` |
| `S_newice` | Salinity of newly formed ice (ppt) | `14` |
| `cloudcover` | Cloud cover fraction | `0.832` |
| `mixing_coefficient` | Ocean mixing coefficient (m²/s) | `1000` |
| `current_NAtl` | North Atlantic current (m³/s) | `3.5e6` |
| `current_NPac` | North Pacific current (m³/s) | `1.0e6` |
| `T_NAtl` | North Atlantic temperature (°C) | `4` |
| `S_NAtl` | North Atlantic salinity (ppt) | `35.06` |
| `T_NPac` | North Pacific temperature (°C) | `5.19` |
| `S_NPac` | North Pacific salinity (ppt) | `33.83` |
| `T_deep` | Deep ocean temperature (°C) | `-1.85` |
| `S_deep` | Deep ocean salinity (ppt) | `34.5` |
| `Kv` | Vertical diffusion coefficient (m²/s) | `1e-5` |

#### `box_params_EArc`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `k_ice_mean` | Ice thermal conductivity seasonal mean (W m⁻¹ K⁻¹) | `2.115` |
| `k_ice_amp` | Ice thermal conductivity seasonal amplitude | `-0.025` |
| `T_atm` | Air temperature time series (overrides seasonal cycle if provided) | `[nan]` |
| `T_atm_mean` | Air temperature seasonal mean (°C) | `-12.8` |
| `T_atm_amp` | Air temperature seasonal amplitude (°C) | `-14.6` |
| `dewpt_atm_mean` | Air dewpoint seasonal mean | `-10.095` |
| `dewpt_atm_amp` | Air dewpoint seasonal amplitude | `-16.205` |
| `dewpt_ice_mean` | Dewpoint over ice seasonal mean | `-9.05` |
| `dewpt_ice_amp` | Dewpoint over ice seasonal amplitude | `-16.95` |
| `river_monthly` | Monthly river discharge (Sv) | `[0.0245, 0.0226, 0.0226, 0.0207, 0.0829, 0.2262, 0.1150, 0.0867, 0.0716, 0.0509, 0.0302, 0.0264]` |
| `precip_monthly` | Monthly precipitation (mm) | `[5, 5, 5, 5, 8, 12, 20, 23, 30, 25, 14, 10]` |
| `shortwave_mean` | Shortwave radiation seasonal mean (W m⁻²) | `50` |
| `shortwave_amp` | Shortwave radiation seasonal amplitude (W m⁻²) | `125` |
| `U10m_atm` | 10 m wind speed (m/s) | `5` |
| `Lice_daily` | Daily land ice flux into the East Arctic | `0.0` |
| `Lice_length` | Length of land boundary with ice | `0.0` |
| `Eice_daily` | Daily sea ice export | `0.0` |
| `add_h_ice_intervention_monthly` | Monthly ice intervention (cm) | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |
| `T_atm_noise_scale` | Standard deviation of air temperature noise | `0.0` |

#### `box_params_WArc`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `k_ice_mean` | Ice thermal conductivity seasonal mean (W m⁻¹ K⁻¹) | `1.985` |
| `k_ice_amp` | Ice thermal conductivity seasonal amplitude | `-0.105` |
| `T_atm` | Air temperature time series (overrides seasonal cycle if provided) | `[nan]` |
| `T_atm_mean` | Air temperature seasonal mean (°C) | `-10.5` |
| `T_atm_amp` | Air temperature seasonal amplitude (°C) | `-13.7` |
| `dewpt_atm_mean` | Air dewpoint seasonal mean | `-10.095` |
| `dewpt_atm_amp` | Air dewpoint seasonal amplitude | `-16.205` |
| `dewpt_ice_mean` | Dewpoint over ice seasonal mean | `-9.05` |
| `dewpt_ice_amp` | Dewpoint over ice seasonal amplitude | `-16.95` |
| `river_monthly` | Monthly river discharge (Sv) | `[0.0052, 0.0052, 0.0052, 0.0052, 0.0182, 0.0260, 0.0221, 0.0169, 0.0140, 0.0114, 0.0065, 0.0052]` |
| `precip_monthly` | Monthly precipitation (mm) | `[5, 5, 5, 5, 8, 12, 20, 23, 30, 25, 14, 10]` |
| `shortwave_mean` | Shortwave radiation seasonal mean (W m⁻²) | `50` |
| `shortwave_amp` | Shortwave radiation seasonal amplitude (W m⁻²) | `125` |
| `U10m_atm` | 10 m wind speed (m/s) | `5` |
| `Lice_daily` | Daily land ice flux into the West Arctic | `0.0` |
| `Lice_length` | Length of land boundary with ice | `0.0` |
| `Eice_daily` | Daily sea ice export | `0.0` |
| `add_h_ice_intervention_monthly` | Monthly ice intervention (cm) | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |
| `T_atm_noise_scale` | Standard deviation of air temperature noise | `0.0` |

#### `time_step`

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `n_years` | Number of years to simulate | `50` |
| `seasonal_cycle_period` | Period of the seasonal cycle (years) | `1` |
| `seasonal_cycle_start_step` | Starting step within the seasonal cycle (0-indexed) | `0` |
| `days_per_year` | Number of days in one year | `365` |
