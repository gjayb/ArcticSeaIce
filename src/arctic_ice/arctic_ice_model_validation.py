# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

# Functions for reading validation data
from scipy.io import loadmat
from datetime import datetime, timedelta


class ValidationData:
    """
    Stores validation datasets used to compare with model output.
    """
    pass


def read_avhrr_from_mat(mat_file):
    """
    Read AVHRR validation data from a MATLAB file.

    :param mat_file: Path to the MATLAB data file
    :return: Dates and data for the East Arctic, West Arctic, North Greenland, and South Greenland
    """
    # All AVHRR CDR mat files have the same variable names
    # Dates for AVHRR CDR data since python cannot read matlab datetime
    date1 = datetime(1990, 1, 1, 0, 0, 0)
    date2 = datetime(2025, 12, 31, 0, 0, 0)
    num_days = (date2 - date1).days + 1
    dates = [date1 + timedelta(days=i) for i in range(num_days)]

    # Read data
    data = loadmat(mat_file)
    var_names = {
        'data_EArc': 'dataArcticEast',
        'data_WArc': 'dataArcticWest',
        'data_NGreen': 'dataGreenlandNorth',
        'data_SGreen': 'dataGreenlandSouth'
    }

    out = {}
    for var_name, data_name in var_names.items():
        if data_name in data:
            out[var_name] = data[data_name]
        else:
            print(f"{data_name} does not exist in data")

    # If a variable name does not exist, then data_xxxx is empty
    data_EArc = out.get('data_EArc')
    data_WArc = out.get('data_WArc')
    data_NGreen = out.get('data_NGreen')
    data_SGreen = out.get('data_SGreen')

    return dates, data_EArc, data_WArc, data_NGreen, data_SGreen


def read_merra_from_mat(mat_file):
    """
    Read MERRA validation data from a MATLAB file.

    :param mat_file: Path to the MATLAB data file
    :return: Dates and data for the East Arctic, West Arctic, North Greenland, and South Greenland
    """
    # All AVHRR CDR mat files have the same variable names
    # Dates for the data since python cannot read matlab datetime
    date1 = datetime(1990, 1, 1, 0, 0, 0)
    date2 = datetime(2025, 12, 31, 0, 0, 0)
    num_days = (date2 - date1).days + 1
    dates = [date1 + timedelta(days=i) for i in range(num_days)]

    # Read data
    data = loadmat(mat_file)
    var_names = {
        'data_EArc': 'dataArcticEast',
        'data_WArc': 'dataArcticWest',
        'data_NGreen': 'dataGreenlandNorth',
        'data_SGreen': 'dataGreenlandSouth'
    }

    out = {}
    for var_name, data_name in var_names.items():
        if data_name in data:
            out[var_name] = data[data_name]
        else:
            print(f"{data_name} does not exist in data")

    # If a variable name does not exist, then data_xxxx is empty
    data_EArc = out.get('data_EArc')
    data_WArc = out.get('data_WArc')
    data_NGreen = out.get('data_NGreen')
    data_SGreen = out.get('data_SGreen')

    return dates, data_EArc, data_WArc, data_NGreen, data_SGreen


def read_sic_from_mat(mat_file):
    """
    Read sea ice concentration validation data from a MATLAB file.

    :param mat_file: Path to the MATLAB data file
    :return: Dates and sea ice concentration data for the East and West Arctic
    """
    # Dates for the data since python cannot read matlab datetime
    date1 = datetime(1990, 1, 1, 0, 0, 0)
    date2 = datetime(2024, 6, 30, 0, 0, 0)
    num_days = (date2 - date1).days + 1
    dates = [date1 + timedelta(days=i) for i in range(num_days)]

    # Read data
    data = loadmat(mat_file)
    var_names = {
        'data_EArc': 'icecon_east',
        'data_WArc': 'icecon_west'
    }

    out = {}
    for var_name, data_name in var_names.items():
        if data_name in data:
            out[var_name] = data[data_name]
        else:
            print(f"{data_name} does not exist in data")

    # If a variable name does not exist, then data_xxxx is empty
    data_EArc = out.get('data_EArc')
    data_WArc = out.get('data_WArc')

    return dates, data_EArc, data_WArc
