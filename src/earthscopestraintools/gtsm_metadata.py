import numpy as np
import pandas as pd
import math
import urllib.request as request

from importlib.resources import files
from datetime import datetime
from earthscopestraintools.datasources_api_interact import lookup_fdsn_network_from_bnum
import requests

import logging

logger = logging.getLogger(__name__)


class GtsmMetadata:
    """
    Class for loading and storing GTSM processing metadata for a given station.  Metadata sources include 
    a summary table \n
    https://www.unavco.org/data/strain-seismic/bsm-data/lib/docs/bsm_metadata.txt \n
    and individual station pages ie \n
    http://bsm.unavco.org/bsm/level2/B001/B001.README.txt 
    
    :param network: 2 character FDSN network code
    :type network: str
    :param station: 4 character FDSN station code
    :type station: str
    :param gauge_weights: optional list of 'good' channels, where 1 is good and 0 is bad, defaults to [1, 1, 1, 1]
    :type gauge_weights: list, optional
    
    :ivar network: str
    :ivar station: str
    :ivar latitude: float
    :ivar longitude: float
    :ivar elevation: float
    :ivar gap: float, instrument gap in meters
    :ivar diameter: float, instrument sensing diameter in meters
    :ivar start_date: str, formatted as "%Y-%m-%d"
    :ivar orientation: float, degrees East of North for CH0
    :ivar reference_strains: dict, containing 'linear_date':'YYYY:DOY' and each channel ie 'CH0':reference counts 
    :ivar strain_matrices: dict, contains one or more calibration matrices, keyed to the name of the calibration.  
    :ivar atmp_response: dict, reponse coefficients for each channel
    :ivar tidal_params: dict, keys are tuple of (channel, tidal constituent, phz/amp/doodson)

    """
    def __init__(self, network, station, gauge_weights=None):
        """
        Loads metadata for given station.  
        """
        self.network = network
        self.station = station  
        self.meta_df = self.get_meta_table()
        self.latitude = self.get_latitude()
        self.longitude = self.get_longitude()
        self.elevation = self.get_elevation()
        self.gap = self.get_gap()
        self.diameter = 0.087
        self.orientation = self.get_orientation()
        self.reference_strains = self.get_reference_strains()
        self.start_date = self.get_start_date()
        matrices = {}
        if not gauge_weights:
            self.gauge_weights = [1, 1, 1, 1]
        # matrices['weighted_strain_matrix'] = self.make_weighted_strain_matrix(gauge_weights=gauge_weights)
        matrices["lab"] = self.get_lab_strain_matrix()
        matrices["ER2010"] = self.get_er2010_strain_matrix()
        matrices["CH_prelim"] = self.get_ch_prelim_strain_matrix()
        self.strain_matrices = {k: v for k, v in matrices.items()}
        self.atmp_response = self.get_atmp_response()
        self.tidal_params = self.get_tidal_params()

    #        self.xml = self.get_xml()
    #        self.detrend = self.get_detrend_xml()

    # def get_xml(self):
    #     metadir = "../xml/"
    #     xml_path = "ftp://bsm.unavco.org/pub/bsm/level2/" + self.station + "/"
    #     xml_file = self.station + ".xml"
    #     with closing(request.urlopen(xml_path + xml_file)) as r:
    #         with open(metadir + xml_file, "wb") as f:
    #             shutil.copyfileobj(r, f)
    #     return xmltodict.parse(open(metadir + xml_file).read(), process_namespaces=True)

    def get_meta_table(self):
        """loads summary metadata table into dataframe

        :return: basic metadata for all strainmeter stations
        :rtype: pandas.DataFrame
        """
        url = "https://www.unavco.org/data/strain-seismic/bsm-data/lib/docs/bsm_metadata.txt"
        meta_df = pd.read_csv(url, sep="\s+", index_col="BNUM")
        return meta_df

    def get_latitude(self):
        """parse latitude from metadata table

        :return: station latitude
        :rtype: float
        """
        try:
            lats = self.meta_df["LAT"]
            return float(lats[self.station])
        except:
            logger.info("no latitude found")
            exit(1)

    def get_longitude(self):
        """parse longitude from metadata table

        :return: station longitude
        :rtype: float
        """
        try:
            longs = self.meta_df["LONG"]
            return float(longs[self.station])
        except:
            logger.info("no longitude found")
            exit(1)

    def get_elevation(self):
        """parse elevation from metadata table

        :return: station elevation
        :rtype: float
        """
        try:
            elevations = self.meta_df["ELEV(m)"]
            return float(elevations[self.station])
        except:
            logger.info("no elevation found")
            exit(1)

    def get_gap(self):
        """parse instrument gap from metadata table

        :return: station instument gap in meters
        :rtype: float
        """
        try:
            gaps = self.meta_df["GAP(m)"]
            return float(gaps[self.station])
        except:
            logger.info("no gap found for %s, using .0001" % self.station)
            return 0.0001

    def get_orientation(self):
        """parse instrument orientation from metadata table

        :return: orientation of CH0 in degrees East of North
        :rtype: float
        """
        try:
            orientations = self.meta_df["CH0(EofN)"]
            return float(orientations[self.station])
        except Exception as e:
            logger.error(e)
            logger.error("No orientation found for %s, using 0 deg" % self.station)
            return 0

    def get_start_date(self):
        """parse data start date from metadata table

        :return: start date string as "%Y-%m-%d"
        :rtype: str
        """
        try:
            start_date = self.meta_df.loc[self.station]["DATA_START"]
            return datetime.strptime(start_date, "%Y:%j").strftime("%Y-%m-%d")

        except Exception as e:
            logger.error(e)
            return None

    # def get_orientation_xml(self):
    #     for i, dic in enumerate(
    #         self.xml["strain_xml"]["inst_info"]["sensor_information"]["sensor_response"]
    #     ):
    #         if dic["sensor_type"] == "Gladwin_BSM_component_1_":
    #             orientation = float(
    #                 self.xml["strain_xml"]["inst_info"]["sensor_information"][
    #                     "sensor_response"
    #                 ][i]["orientation"]["#text"]
    #             )
    #     try:
    #         return orientation
    #     except Exception as e:
    #         logger.error(e)
    #         logger.error("No orientation found for %s, using 0 deg" % self.station)
    #         return 0

    def make_weighted_strain_matrix(self, gauge_weights=[1, 1, 1, 1]):
        """calculates lab strain matrix, and allows exclusion of a single gauge

        :param gauge_weights: which gauges to use, set to 0 to exclude one gauge, defaults to [1, 1, 1, 1]
        :type gauge_weights: list, optional
        :return: lab calibration matrix
        :rtype: numpy.array
        """
        # make strain matrix from manufacturers coefficients
        # logger.info(gauge_weights)
        c = 1.5
        d = 3
        scale_factors = np.array([[c, 0, 0], [0, d, 0], [0, 0, d]])
        gage_weights = np.array(
            [
                [gauge_weights[0], 0, 0, 0],
                [0, gauge_weights[1], 0, 0],
                [0, 0, gauge_weights[2], 0],
                [0, 0, 0, gauge_weights[3]],
            ]
        )
        orientations = np.array(
            [
                [
                    0.5,
                    0.5 * math.cos(math.radians(2 * (90 - self.orientation))),
                    0.5 * math.sin(math.radians(2 * (90 - self.orientation))),
                ],
                [
                    0.5,
                    0.5 * math.cos(math.radians(2 * (-30 - self.orientation))),
                    0.5 * math.sin(math.radians(2 * (-30 - self.orientation))),
                ],
                [
                    0.5,
                    0.5 * math.cos(math.radians(2 * (-150 - self.orientation))),
                    0.5 * math.sin(math.radians(2 * (-150 - self.orientation))),
                ],
                [
                    0.5,
                    0.5 * math.cos(math.radians(2 * (-120 - self.orientation))),
                    0.5 * math.sin(math.radians(2 * (-120 - self.orientation))),
                ],
            ]
        )

        # remove row from orientation matrix corresponding to gage weight of 0
        orientations = np.matmul(gage_weights, orientations)
        orientations = orientations[~np.all(orientations == 0, axis=1)]
        # scale gage_weights down to 3x3 identity if only using 3 gages
        gage_weights = gage_weights[:, ~np.all(gage_weights == 0, axis=0)]
        gage_weights = gage_weights[~np.all(gage_weights == 0, axis=1)]

        # calculate the strain matrix
        strain_matrix = np.matmul(
            np.matmul(np.linalg.inv(scale_factors), np.linalg.pinv(orientations)),
            np.linalg.inv(gage_weights),
        )

        # insert a column of zeros back in if one gage was dropped, leaving a 4x3 matrix
        for i in [0, 1, 2, 3]:
            if gauge_weights[i] == 0:
                strain_matrix = np.insert(strain_matrix, i, 0, axis=1)
        # print("strain_matrix: \n",strain_matrix)
        return strain_matrix

    def get_lab_strain_matrix(self):
        """parse lab strain matrix from station metadata 

        :return: lab calibration matrix
        :rtype: np.array
        """
        try:
            url = f"http://bsm.unavco.org/bsm/level2/{self.station}/{self.station}.README.txt"

            with request.urlopen(url) as response:
                lines = response.readlines()
            for i, line in enumerate(lines):
                line = line.decode("utf-8").rstrip()
                if line.startswith("  Manufacturer's Isotropic Strain Matrix"):
                    lab = np.array(
                        [
                            lines[i + 1].decode("utf-8").rstrip().split()[1:],
                            lines[i + 2].decode("utf-8").rstrip().split()[1:],
                            lines[i + 3].decode("utf-8").rstrip().split()[1:],
                        ]
                    )
            return lab.astype(float)
        except Exception as e:
            logger.error("Could not load lab strain matrix")
            return None

    def get_er2010_strain_matrix(self):
        """parse ER2010 strain matrix from station metadata if available.  From
        Roeloffs, E. (2010), Tidal calibration of Plate Boundary Observatory borehole strainmeters: 
        Roles of vertical and shear coupling , J. Geophys. Res., 115, B06405, doi:10.1029/2009JB006407
        
        :return: ER2010 calibration matrix
        :rtype: np.array
        """
        url = f"http://bsm.unavco.org/bsm/level2/{self.station}/{self.station}.README.txt"
        try:
            with request.urlopen(url) as response:
                lines = response.readlines()
            for i, line in enumerate(lines):
                line = line.decode("utf-8").rstrip()
                if line.startswith("  Roeloffs 2010 Tidal Calibration"):
                    er2010 = np.array(
                        [
                            lines[i + 3].decode("utf-8").rstrip().split()[1:],
                            lines[i + 4].decode("utf-8").rstrip().split()[1:],
                            lines[i + 5].decode("utf-8").rstrip().split()[1:],
                        ]
                    )
                    return er2010.astype(float)
            return None
        except Exception as e:
            logger.error("Could not load ER2010 strain matrix")
            return None

    def get_ch_prelim_strain_matrix(self):
        """parse CH preliminary strain matrix from station metadata if available.  
        Calibrations of TABOO-STAR stations performed by Cassie Hanagan in 2022.
        
        :return: CH_PRELIM calibration matrix
        :rtype: np.array
        """
        url = f"http://bsm.unavco.org/bsm/level2/{self.station}/{self.station}.README.txt"
        try:
            with request.urlopen(url) as response:
                lines = response.readlines()
            for i, line in enumerate(lines):
                line = line.decode("utf-8").rstrip()
                if line.startswith("  CH Preliminary Tidal Calibration"):
                    ch_prelim = np.array(
                        [
                            lines[i + 3].decode("utf-8").rstrip().split()[1:],
                            lines[i + 4].decode("utf-8").rstrip().split()[1:],
                            lines[i + 5].decode("utf-8").rstrip().split()[1:],
                        ]
                    )
                    return ch_prelim.astype(float)
            return None
        except Exception as e:
            logger.exception("Could not load ch_prelim strain matrix")
            return None

    def get_reference_strains(self):
        """Parse reference strains from metadata 

        :return: reference strains
        :rtype: dict
        """
        reference_strains = {}
        reference_strains["linear_date"] = self.meta_df.loc[self.station]["L_DATE"]
        reference_strains["CH0"] = int(self.meta_df.loc[self.station]["L0(cnts)"])
        reference_strains["CH1"] = int(self.meta_df.loc[self.station]["L1(cnts)"])
        reference_strains["CH2"] = int(self.meta_df.loc[self.station]["L2(cnts)"])
        reference_strains["CH3"] = int(self.meta_df.loc[self.station]["L3(cnts)"])
        return reference_strains

    # def get_linearization_params_xml(self):
    #     # get reference strains from xml
    #     linear_dict = self.xml["strain_xml"]["inst_info"]["processing"][
    #         "bsm_processing_history"
    #     ][-1]["bsm_processing"]["linearization"]
    #     self.reference_strains = {}
    #     for key in linear_dict:
    #         # print(key, linear_dict[key])
    #         if key == "linear_date":
    #             self.reference_strains["linear_date"] = linear_dict[key]
    #         if key == "g0_value":
    #             self.reference_strains["CH0"] = float(linear_dict[key])
    #         if key == "g1_value":
    #             self.reference_strains["CH1"] = float(linear_dict[key])
    #         if key == "g2_value":
    #             self.reference_strains["CH2"] = float(linear_dict[key])
    #         if key == "g3_value":
    #             self.reference_strains["CH3"] = float(linear_dict[key])
    #
    # def get_gauge_weightings_xml(self):
    #     weight_dict = self.xml["strain_xml"]["inst_info"]["processing"][
    #         "bsm_processing_history"
    #     ][-1]["bsm_processing"]["gauge_weightings"]
    #     self.gauge_weightings = [
    #         int(weight_dict["gw0"]),
    #         int(weight_dict["gw1"]),
    #         int(weight_dict["gw2"]),
    #         int(weight_dict["gw3"]),
    #     ]
    #
    # def get_detrend_xml(self):
    #     detrend = {}
    #     for channel in [
    #         "detrend_start_date",
    #         "detrend_g0",
    #         "detrend_g1",
    #         "detrend_g2",
    #         "detrend_g3",
    #     ]:
    #         if channel == "detrend_start_date":
    #             detrend[channel] = self.xml["strain_xml"]["inst_info"]["processing"][
    #                 "bsm_processing_history"
    #             ][-1]["bsm_processing"]["timeseries_start_date"]
    #         else:
    #             detrend_dict = {}
    #             detrend_params = self.xml["strain_xml"]["inst_info"]["processing"][
    #                 "bsm_processing_history"
    #             ][-1]["bsm_processing"][channel]
    #             for key in detrend_params:
    #                 if key[0] != "@":
    #                     detrend_dict[key] = float(detrend_params[key])
    #             detrend[channel] = detrend_dict
    #     return detrend

    def get_atmp_response(self):
        """Parse atmospheric pressure response from station processing metadata

        :return: atmospheric response coefficients in microstrain/hPa
        :rtype: dict
        """
        url = f"http://bsm.unavco.org/bsm/level2/{self.station}/{self.station}.README.txt"
        try:
            with request.urlopen(url) as response:
                lines = response.readlines()
            baro = False
            atmp_coefficients = {}
            for line in lines:
                line = line.decode("utf-8").rstrip()
                if line.startswith("Barometric Response"):
                    baro = True
                if baro:
                    if line.startswith("CH"):
                        line = line.split()
                        atmp_coefficients[line[0]] = (
                            float(line[1]) * 1e-3
                        )  # microstrain
                        if line[0] == "CH3":
                            baro = False
            return atmp_coefficients
        except Exception as e:
            logger.error("Could not load atmp response")
            return None

    def get_tidal_params(self):
        """Parse tidal params from station processing metadata

        :return: tidal coefficients
        :rtype: dict
        """
        url = f"http://bsm.unavco.org/bsm/level2/{self.station}/{self.station}.README.txt"
        try:
            with request.urlopen(url) as response:
                lines = response.readlines()
            tide = False
            tide_coefficients = {}
            for line in lines:
                line = line.decode("utf-8").rstrip()

                if line.startswith("Tidal Constituents"):
                    tide = True

                if tide:
                    if line.startswith("CH"):
                        line = line.split()
                        if line[1] == "M2":
                            doodson = "2 0 0 0 0 0"
                        elif line[1] == "O1":
                            doodson = "1-1 0 0 0 0"
                        elif line[1] == "P1":
                            doodson = "1 1-2 0 0 0"
                        elif line[1] == "K1":
                            doodson = "1 1 0 0 0 0"
                        elif line[1] == "N2":
                            doodson = "2-1 0 1 0 0"
                        elif line[1] == "S2":
                            doodson = "2 2-2 0 0 0"
                        tide_coefficients[(line[0], line[1], "phz")] = line[2]
                        tide_coefficients[(line[0], line[1], "amp")] = line[3]
                        tide_coefficients[(line[0], line[1], "doodson")] = doodson
            return tide_coefficients
        except Exception as e:
            logger.error("Could not load tidal parameters")
            return None

    def load_site_terms(self):
        """Read site terms from file"""
        terms_file = files("earthscopestraintools").joinpath("event_site_terms.txt")
        site_terms = pd.read_csv(terms_file, sep=" ", header=0).fillna(0)
        return site_terms

    def get_event_terms(self):
        """
        Loads site and longitude terms for a particular strainmeter using Barbour et al 2021.  
        These terms are used for event magnitude estimation.
        """
        site_terms = self.load_site_terms()
        if self.station in site_terms.Station.values:
            self.site_term = site_terms[site_terms.Station == self.station].delta_s.item()
        else:
            self.site_term = 0
        if self.longitude < -124:
            self.longitude_term = -0.41
        else:
            self.longitude_term = 0

    def show(self):
        """print metadata
        """
        # pp = pprint.PrettyPrinter()
        logger.info(f"network: {self.network}")
        logger.info(f"station: {self.station}")
        logger.info(f"latitude: {self.latitude}")
        logger.info(f"longitude: {self.longitude}")
        logger.info(f"gap: {self.gap}")
        logger.info(f"orientation (CH0EofN): {self.orientation}")
        logger.info(f"reference_strains:\n {self.reference_strains}")
        logger.info("strain_matrices:")
        if len(self.strain_matrices):
            for key in self.strain_matrices:
                logger.info(f"{key}:\n {self.strain_matrices[key]}")
                # logger.info(self.strain_matrices[key])
                # pp.pprint(self.strain_matrices[key])
        logger.info(f"atmp_response:\n {self.atmp_response}")
        # logger.info(self.atmp_response)
        logger.info(f"tidal_params:\n {self.tidal_params}")
        # logger.info(self.tidal_params)

        # print("reference strains:")
        # pp.pprint(self.linearization)
        # if len(self.strain_matrices):
        #     for key in self.strain_matrices:
        #         print(f"{key}:")
        #         print(self.strain_matrices[key])
        #         #pp.pprint(self.strain_matrices[key])
        # print("atmp coefficients:")
        # pp.pprint(self.atmp_response)
        # print("tidal params: ")
        # pp.pprint(self.tidal_params)
        # print("detrend params:")
        # pp.pprint(self.detrend)


def fdsn2bottlename(channel):
    """convert FDSN channel into bottlename

    :param channel: FDSN channel code
    :type channel: str
    :return: bottlename 
    :rtype: str
    """

    codes = {
        "RS1": "CH0",
        "LS1": "CH0",
        "BS1": "CH0",
        "RS2": "CH1",
        "LS2": "CH1",
        "BS2": "CH1",
        "RS3": "CH2",
        "LS3": "CH2",
        "BS3": "CH2",
        "RS4": "CH3",
        "LS4": "CH3",
        "BS4": "CH3",
        "RDO": "atmp",
        "LDO": "atmp",
    }

    return codes[channel]


def get_metadata_df():
    """Function loads strainmeter metadata into pandas dataframe

    :return: bsm metadata 
    :rtype: pandas.DataFrame
    """
        
    # """
    # 

    # Parameters
    # ----------
    # :return: metadata: pandas DataFrame

    # """
    url = (
        "https://www.unavco.org/data/strain-seismic/bsm-data/lib/docs/bsm_metadata.txt"
    )
    metadata = pd.read_csv(url, index_col="BNUM", sep="\s+", on_bad_lines="skip")
    return metadata


def get_fdsn_network(station):
    """Get FDSN network name from station fourcharid using es-datasources-api

    :param station: station four character id
    :type station: string
    :return: 2 char FDSN network code for station
    :rtype: string
    """
    try:
        DATASOURCES_API_URL = "https://datasources-api.prod.earthscope.org"
        return lookup_fdsn_network_from_bnum(DATASOURCES_API_URL,station)
    except requests.exceptions.ConnectionError:
        logger.error(
            "Error: Unable to connect to datasources-api to lookup FDSN network code"
        )
        return None
