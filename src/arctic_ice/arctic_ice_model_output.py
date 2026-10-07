# Copyright 2026, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import numpy as np


class ArcticIceOutput:
    def __init__(self,
                 h_ice: np.ndarray,
                 SIC: np.ndarray,
                 ice_type: np.ndarray,
                 ice_volume: np.ndarray,
                 T_icesurface: np.ndarray,
                 T_icebottom: np.ndarray
                 ):
        """
        Stores the sea ice model output variables.

        :param h_ice: Ice thickness [m]
        :param SIC: Sea ice concentration
        :param ice_type: 2 for multi-year ice and 1 for first year ice
        :param ice_volume: Volume of sea ice [m3]
        :param T_icesurface: Temperature of the top of the sea ice [degC]
        :param T_icebottom: Temperature of the bottom of the sea ice [degC]
        """
        self.h_ice = h_ice
        self.SIC = SIC
        self.ice_type = ice_type
        self.ice_volume = ice_volume
        self.T_icesurface = T_icesurface
        self.T_icebottom = T_icebottom

    def unpack(self) -> tuple:
        """
        Return the sea ice output variables as a tuple.

        :return: Sea ice output variables
        """
        return self.h_ice, self.SIC, self.ice_type, self.ice_volume, self.T_icesurface, self.T_icebottom

    def to_dict(self) -> dict:
        """
        Return the output variables as a dictionary.

        :return: Dictionary of output variables
        """
        return dict(
            h_ice=self.h_ice,
            SIC=self.SIC,
            ice_type=self.ice_type,
            ice_volume=self.ice_volume,
            T_icesurface=self.T_icesurface,
            T_icebottom=self.T_icebottom
        )


class ArcticOceanOutput:
    def __init__(self,
                 T_ocean: np.ndarray,
                 S_ocean: np.ndarray,
                 T_atm: np.ndarray
                 ):
        """
        Stores the ocean and atmosphere model output variables.

        :param T_ocean: Ocean temperature [degC]
        :param S_ocean: Ocean salinity [ppt]
        :param T_atm: Atmosphere temperature [degC]
        """
        self.T_ocean = T_ocean
        self.S_ocean = S_ocean
        self.T_atm = T_atm

    def unpack(self) -> tuple:
        """
        Return the ocean and atmosphere output variables as a tuple.

        :return: Ocean and atmosphere output variables
        """
        return self.T_ocean, self.S_ocean, self.T_atm

    def to_dict(self) -> dict:
        """
        Return the output variables as a dictionary.

        :return: Dictionary of output variables
        """
        return dict(
            T_ocean=self.T_ocean,
            S_ocean=self.S_ocean,
            T_atm=self.T_atm,
        )


class ArcticFluxOutput:
    def __init__(self,
                 F_sw: np.ndarray,
                 F_lw: np.ndarray,
                 F_sh: np.ndarray,
                 F_lh: np.ndarray,
                 F_b: np.ndarray,
                 F_up: np.ndarray,
                 F_bd: np.ndarray,
                 F_r: np.ndarray,
                 F_p: np.ndarray,
                 F_lhfresh: np.ndarray
                 ):
        """
        Stores the heat, salt, and freshwater flux output variables.

        :param F_sw: Shortwave flux [W/m2]
        :param F_lw: Longwave flux [W/m2]
        :param F_sh: Shortwave flux [W/m2]
        :param F_lh: Latent heat flux [W/m2]
        :param F_b: Bottom heat flux from ocean [W/m2]
        :param F_up: Freshwater uptake flux for salt flux [g salt/m2/s]
        :param F_bd: Brine discharge flux [g salt/m2/s]
        :param F_r: River discharge flux [kg freshwater/m2/s]
        :param F_p: Precipitation flux [kg freshwater/m2/s]
        :param F_lhfresh: Freshwater latent heat flux [kg freshwater/m2/s]
        """
        self.F_sw = F_sw
        self.F_lw = F_lw
        self.F_sh = F_sh
        self.F_lh = F_lh
        self.F_b = F_b
        self.F_up = F_up
        self.F_bd = F_bd
        self.F_r = F_r
        self.F_p = F_p
        self.F_lhfresh = F_lhfresh

    def unpack(self) -> tuple:
        """
        Return the surface flux output variables as a tuple.

        :return: Surface heat, salt, and freshwater flux output variables
        """
        return (self.F_sw, self.F_lw, self.F_sh, self.F_lh, self.F_b,
                self.F_up, self.F_bd, self.F_r, self.F_p, self.F_lhfresh)

    def to_dict(self) -> dict:
        """
        Return the output variables as a dictionary.

        :return: Dictionary of output variables
        """
        return dict(
            F_sw=self.F_sw,
            F_lw=self.F_lw,
            F_sh=self.F_sh,
            F_lh=self.F_lh,
            F_b=self.F_b,
            F_up=self.F_up,
            F_bd=self.F_bd,
            F_r=self.F_r,
            F_p=self.F_p,
            F_lhfresh=self.F_lhfresh
        )


class ArcticCurrentsOutput:
    def __init__(self,
                 Fheat_cur_NAtl_to_EArc: np.ndarray,
                 Fheat_cur_EArc_to_WArc_forEArc: np.ndarray,
                 Fheat_cur_EArc_to_WArc_forWArc: np.ndarray,
                 Fheat_cur_EArc_to_NPac: np.ndarray,
                 Fheat_cur_NPac_to_WArc: np.ndarray,
                 Fheat_cur_WArc_to_NAtl: np.ndarray,
                 Fheat_mix_EArc_to_WArc: np.ndarray,
                 Fheat_mix_WArc_to_EArc: np.ndarray,
                 Fheat_upw_EArc_to_deep: np.ndarray,
                 Fheat_upw_WArc_to_deep: np.ndarray,
                 Fsalt_cur_NAtl_to_EArc: np.ndarray,
                 Fsalt_cur_EArc_to_WArc_forEArc: np.ndarray,
                 Fsalt_cur_EArc_to_WArc_forWArc: np.ndarray,
                 Fsalt_cur_EArc_to_NPac: np.ndarray,
                 Fsalt_cur_NPac_to_WArc: np.ndarray,
                 Fsalt_cur_WArc_to_NAtl: np.ndarray,
                 Fsalt_mix_EArc_to_WArc: np.ndarray,
                 Fsalt_mix_WArc_to_EArc: np.ndarray,
                 Fsalt_upw_EArc_to_deep: np.ndarray,
                 Fsalt_upw_WArc_to_deep: np.ndarray
                 ):
        """
        Stores the heat and salt transport between Arctic model boxes.

        :param Fheat_cur_NAtl_to_EArc: Heat transport from the North Atlantic to the East Arctic [W m^-2]
        :param Fheat_cur_EArc_to_WArc_forEArc: Heat transport from the East Arctic to the West Arctic applied to the East Arctic [W m^-2]
        :param Fheat_cur_EArc_to_WArc_forWArc: Heat transport from the East Arctic to the West Arctic applied to the West Arctic [W m^-2]
        :param Fheat_cur_EArc_to_NPac: Heat transport from the East Arctic to the North Pacific [W m^-2]
        :param Fheat_cur_NPac_to_WArc: Heat transport from the North Pacific to the West Arctic [W m^-2]
        :param Fheat_cur_WArc_to_NAtl: Heat transport from the West Arctic to the North Atlantic [W m^-2]
        :param Fheat_mix_EArc_to_WArc: Heat transport due to mixing from the East Arctic to the West Arctic [W m^-2]
        :param Fheat_mix_WArc_to_EArc: Heat transport due to mixing from the West Arctic to the East Arctic [W m^-2]
        :param Fheat_upw_EArc_to_deep: Heat transport due to upwelling between the East Arctic and deep ocean [W m^-2]
        :param Fheat_upw_WArc_to_deep: Heat transport due to upwelling between the West Arctic and deep ocean [W m^-2]
        :param Fsalt_cur_NAtl_to_EArc: Salt transport from the North Atlantic to the East Arctic [g m^-2 s^-1]
        :param Fsalt_cur_EArc_to_WArc_forEArc: Salt transport from the East Arctic to the West Arctic applied to the East Arctic [g m^-2 s^-1]
        :param Fsalt_cur_EArc_to_WArc_forWArc: Salt transport from the East Arctic to the West Arctic applied to the West Arctic [g m^-2 s^-1]
        :param Fsalt_cur_EArc_to_NPac: Salt transport from the East Arctic to the North Pacific [g m^-2 s^-1]
        :param Fsalt_cur_NPac_to_WArc: Salt transport from the North Pacific to the West Arctic [g m^-2 s^-1]
        :param Fsalt_cur_WArc_to_NAtl: Salt transport from the West Arctic to the North Atlantic [g m^-2 s^-1]
        :param Fsalt_mix_EArc_to_WArc: Salt transport due to mixing from the East Arctic to the West Arctic [g m^-2 s^-1]
        :param Fsalt_mix_WArc_to_EArc: Salt transport due to mixing from the West Arctic to the East Arctic [g m^-2 s^-1]
        :param Fsalt_upw_EArc_to_deep: Salt transport due to upwelling between the East Arctic and deep ocean [g m^-2 s^-1]
        :param Fsalt_upw_WArc_to_deep: Salt transport due to upwelling between the West Arctic and deep ocean [g m^-2 s^-1]
        """
        self.Fheat_cur_NAtl_to_EArc = Fheat_cur_NAtl_to_EArc
        self.Fheat_cur_EArc_to_WArc_forEArc = Fheat_cur_EArc_to_WArc_forEArc
        self.Fheat_cur_EArc_to_WArc_forWArc = Fheat_cur_EArc_to_WArc_forWArc
        self.Fheat_cur_EArc_to_NPac = Fheat_cur_EArc_to_NPac
        self.Fheat_cur_NPac_to_WArc = Fheat_cur_NPac_to_WArc
        self.Fheat_cur_WArc_to_NAtl = Fheat_cur_WArc_to_NAtl
        self.Fheat_mix_EArc_to_WArc = Fheat_mix_EArc_to_WArc
        self.Fheat_mix_WArc_to_EArc = Fheat_mix_WArc_to_EArc
        self.Fheat_upw_EArc_to_deep = Fheat_upw_EArc_to_deep
        self.Fheat_upw_WArc_to_deep = Fheat_upw_WArc_to_deep
        self.Fsalt_cur_NAtl_to_EArc = Fsalt_cur_NAtl_to_EArc
        self.Fsalt_cur_EArc_to_WArc_forEArc = Fsalt_cur_EArc_to_WArc_forEArc
        self.Fsalt_cur_EArc_to_WArc_forWArc = Fsalt_cur_EArc_to_WArc_forWArc
        self.Fsalt_cur_EArc_to_NPac = Fsalt_cur_EArc_to_NPac
        self.Fsalt_cur_NPac_to_WArc = Fsalt_cur_NPac_to_WArc
        self.Fsalt_cur_WArc_to_NAtl = Fsalt_cur_WArc_to_NAtl
        self.Fsalt_mix_EArc_to_WArc = Fsalt_mix_EArc_to_WArc
        self.Fsalt_mix_WArc_to_EArc = Fsalt_mix_WArc_to_EArc
        self.Fsalt_upw_EArc_to_deep = Fsalt_upw_EArc_to_deep
        self.Fsalt_upw_WArc_to_deep = Fsalt_upw_WArc_to_deep

    def unpack(self) -> tuple:
        """
        Return the current transport output variables as a tuple.

        :return: Heat and salt transport output variables
        """
        return (self.Fheat_cur_NAtl_to_EArc, self.Fheat_cur_EArc_to_WArc_forEArc, self.Fheat_cur_EArc_to_WArc_forWArc,
                self.Fheat_cur_EArc_to_NPac,
                self.Fheat_cur_NPac_to_WArc, self.Fheat_cur_WArc_to_NAtl,
                self.Fheat_mix_EArc_to_WArc, self.Fheat_mix_WArc_to_EArc,
                self.Fheat_upw_EArc_to_deep, self.Fheat_upw_WArc_to_deep,
                self.Fsalt_cur_NAtl_to_EArc, self.Fsalt_cur_EArc_to_WArc_forEArc, self.Fsalt_cur_EArc_to_WArc_forWArc,
                self.Fsalt_cur_EArc_to_NPac,
                self.Fsalt_cur_NPac_to_WArc, self.Fsalt_cur_WArc_to_NAtl,
                self.Fsalt_mix_EArc_to_WArc, self.Fsalt_mix_WArc_to_EArc,
                self.Fsalt_upw_EArc_to_deep, self.Fsalt_upw_WArc_to_deep,
                )

    def to_dict(self) -> dict:
        """
        Return the output variables as a dictionary.

        :return: Dictionary of output variables
        """
        return dict(
            Fheat_cur_NAtl_to_EArc=self.Fheat_cur_NAtl_to_EArc,
            Fheat_cur_EArc_to_WArc_forEArc=self.Fheat_cur_EArc_to_WArc_forEArc,
            Fheat_cur_EArc_to_WArc_forWArc=self.Fheat_cur_EArc_to_WArc_forWArc,
            Fheat_cur_EArc_to_NPac=self.Fheat_cur_EArc_to_NPac,
            Fheat_cur_NPac_to_WArc=self.Fheat_cur_NPac_to_WArc,
            Fheat_cur_WArc_to_NAtl=self.Fheat_cur_WArc_to_NAtl,
            Fheat_mix_EArc_to_WArc=self.Fheat_mix_EArc_to_WArc,
            Fheat_mix_WArc_to_EArc=self.Fheat_mix_WArc_to_EArc,
            Fheat_upw_EArc_to_deep=self.Fheat_upw_EArc_to_deep,
            Fheat_upw_WArc_to_deep=self.Fheat_upw_WArc_to_deep,
            Fsalt_cur_NAtl_to_EArc=self.Fsalt_cur_NAtl_to_EArc,
            Fsalt_cur_EArc_to_WArc_forEArc=self.Fsalt_cur_EArc_to_WArc_forEArc,
            Fsalt_cur_EArc_to_WArc_forWArc=self.Fsalt_cur_EArc_to_WArc_forWArc,
            Fsalt_cur_EArc_to_NPac=self.Fsalt_cur_EArc_to_NPac,
            Fsalt_cur_NPac_to_WArc=self.Fsalt_cur_NPac_to_WArc,
            Fsalt_cur_WArc_to_NAtl=self.Fsalt_cur_WArc_to_NAtl,
            Fsalt_mix_EArc_to_WArc=self.Fsalt_mix_EArc_to_WArc,
            Fsalt_mix_WArc_to_EArc=self.Fsalt_mix_WArc_to_EArc,
            Fsalt_upw_EArc_to_deep=self.Fsalt_upw_EArc_to_deep,
            Fsalt_upw_WArc_to_deep=self.Fsalt_upw_WArc_to_deep
        )
