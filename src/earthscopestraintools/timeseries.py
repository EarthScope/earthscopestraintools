import os
import numpy as np
import pandas as pd
import datetime

from earthscopestraintools.processing import (
    linearize,
    interpolate,
    decimate_1s_to_300s,
    butterworth_filter,
    decimate_to_hourly,
    apply_calibration_matrix,
    calculate_offsets,
    calculate_pressure_correction,
    calculate_tide_correction,
    calculate_linear_trend_correction,
    baytap_analysis
)
from earthscopestraintools.event_processing import dynamic_strain, calculate_magnitude
from earthscopestraintools.strain_visualization import strain_video
from scipy import signal, stats
import matplotlib.pyplot as plt
from matplotlib import cm
import logging

logger = logging.getLogger(__name__)


def test():
    print("test1")


class Timeseries:
    """
    Class for storing strainmeter data or correction timeseries.  Each Timeseries object 
    contains the following attributes: \n
    - data: (pd.DataFrame) with datetime index and one or more columns of timeseries data \n
    - quality_df: (pd.DataFrame) with same shape as data, but with a character mapped to each
      data point.  flags include "g"=good, "m"=missing, "i"=interpolated, "b"=bad \n
    - series: (str) description of timeseries, ie 'raw', 'microstrain', 'atmp_c', 'tide_c', 'offset_c', 'trend_c' \n
    - units: (str) units of timeseries \n
    - level: (str) level of data. ie. '0','1','2a','2b' \n
    - period: (float) sample period of data \n
    - name: (str) name of timeseries, including station name.  useful for showing stats.  
      defaults to network.station \n
    - network: (str) FDSN two character network code\n
    - station: (str) FDSN four character station code
    """
    def __init__(
        self,
        data: pd.DataFrame = None,
        quality_df: pd.DataFrame = None,
        series: str = "",
        units: str = "",
        level: str = "",
        period: float = 0,
        name: str = None,
        network: str = "",
        station: str = "",
    ):
        if data is not None:
            self.data = data
            self.columns = list(data.columns)
        else:
            self.data = pd.DataFrame()
            self.columns = []

        if quality_df is not None:
            self.quality_df = quality_df
        else:
            self.quality_df = self.set_initial_quality_flags()

        self.series = series
        self.units = units
        self.level = level
        self.period = period
        self.network = network
        self.station = station
        if name:
            self.name = name
        else:
            self.name = f"{self.network}.{self.station}"
        self.check_for_gaps()

    def set_data(self, df):
        self.data = df

    def set_period(self, period):
        self.period = period

    def set_units(self, units):
        self.units = units

    def set_local_tdb_uri(self, local_tdb_uri):
        if self.local_archive:
            local_tdb_uri = f"{self.local_archive}/{local_tdb_uri}"
        self.local_tdb_uri = local_tdb_uri

    def set_s3_tdb_uri(self, s3_tdb_uri):
        self.s3_tdb_uri = s3_tdb_uri

    def set_initial_quality_flags(self, missing_data=999999):
        """used to flag any missing data

        :param missing_data: value used to represent gaps, defaults to 999999
        :type missing_data: int, optional
        :return: dataframe of quality flags
        :rtype: pandas.DataFrame
        """
        qual_df = pd.DataFrame(index=self.data.index)
        for ch in self.columns:
            qual_df[ch] = "g"
            qual_df[ch][self.data[ch] == missing_data] = "m"

        # print(qual_df.value_counts())
        return qual_df

    def set_initial_level_flags(self, level):
        """used to set level flag 

        :param level: level of data ['0','1','2a','2b']
        :type level: str, optional
        :return: dataframe of level flags
        :rtype: pandas.DataFrame
        """
        level_df = pd.DataFrame(index=self.data.index)
        for ch in self.columns:
            level_df[ch] = level
        return level_df

    def set_initial_version(self):
        """adds a version timestamp to each data point

        :return: dataframe of version flags
        :rtype: pandas.DataFrame
        """
        version_df = pd.DataFrame(index=self.data.index)
        version = int(datetime.datetime.now().strftime("%Y%j%H%M%S"))
        for ch in self.columns:
            version_df[ch] = version
        return version_df

    def check_for_gaps(self):
        """generates some statistics around nans, fill values, and missing epochs.
        """
        if self.period != 0 and self.data is not None:
            days = (self.data.index[-1] - self.data.index[0]).days
            seconds = (self.data.index[-1] - self.data.index[0]).seconds
            self.nans = int(self.data.isna().sum().sum() / len(self.data.columns))
            self.nines = round(
                self.data[self.data == 999999].count().sum() / len(self.data.columns), 2
            )
            expected_epochs = int((days * 86400 + seconds) / self.period) + 1
            self.epochs = len(self.data) - self.nans - self.nines
            # print(self.nans, self.nines, self.epochs, expected_epochs)
            self.gap_percentage = round((1 - (self.epochs / expected_epochs)) * 100, 2)

            logger.info(
                f"    Found {self.nans} epochs with nans, {self.nines} epochs with 999999s, and "
                f"{expected_epochs - len(self.data)} missing epochs.\n    Total missing data is "
                f"{self.gap_percentage}%"
            )
        else:
            self.gaps = None
            self.gap_percentage = None

    def stats(self):
        """displays summary information describing the Timeseries object
        """
        if len(self.data):
            outputstring = f"{self.name}\n{'':4}| Channels: {str(self.columns):40} "
            outputstring += f"\n{'':4}| TimeRange: {self.data.index[0]} - {self.data.index[-1]}        | Period: {self.period:>13}s"
            outputstring += f"\n{'':4}| Series: {self.series:>11}| Units: {self.units:>13}| Level: {self.level:>10}| Gaps: {self.gap_percentage:>15}% "
            if len(self.quality_df):
                cols = len(self.quality_df.columns)
                outputstring += f"\n{'':4}| Epochs: {len(self.quality_df):11}"
                outputstring += (
                    f"| Good: {(self.quality_df == 'g').sum().sum() / cols:14}"
                )
                outputstring += (
                    f"| Missing: {(self.quality_df == 'm').sum().sum() / cols:8}"
                )
                outputstring += (
                    f"| Interpolated: {(self.quality_df == 'i').sum().sum() / cols:8}"
                )
                outputstring += f"\n{'':4}| Samples: {len(self.quality_df) * cols:10}"
                outputstring += f"| Good: {(self.quality_df == 'g').sum().sum():14}"
                outputstring += f"| Missing: {(self.quality_df == 'm').sum().sum():8}"
                outputstring += (
                    f"| Interpolated: {(self.quality_df == 'i').sum().sum():8}"
                )
            logger.info(f"{outputstring}")

    def show_flags(self):
        """returns a dataframe with all flags that are not 'g'

        :return: times and channels with flagged data within the timeseries
        :rtype: pandas.DataFrame
        """
        return self.quality_df[self.quality_df[self.quality_df != "g"].any(axis=1)]

    def show_flagged_data(self):
        """returns dataframe containing any data with a flag other than 'g'

        :return: data that has been flagged
        :rtype: pandas.DataFrame
        """
        return self.data[self.quality_df[self.quality_df != "g"].any(axis=1)]


    def save_csv(self, filename: str, datadir: str = "./", sep=",", compression=None):
        """save data attribute as csv.  flattens object, does not save quality flags, level, or version information

        :param filename: name of csv file to save
        :type filename: str
        :param datadir: path to local directory to save file, defaults to "./"
        :type datadir: str, optional
        :param sep: separator to use in csv, defaults to ","
        :type sep: str, optional
        :param compression: compression algorthim ['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd'], defaults to None
        :type compression: str, optional
        """
        filepath = os.path.join(datadir, filename)
        logger.info(f"saving to {filepath}")
        if compression:
            self.data.to_csv(filepath, sep=sep, compression={"method": compression})
        else:
            self.data.to_csv(filepath, sep=sep)

    # def save_tdb(self):
    # TODO: update, handle quality fields
    # if self.local_tdb_uri:
    #     logger.info(f"saving to {self.local_tdb_uri}")
    #     self.array = StrainArray(uri=self.local_tdb_uri, period=self.period)
    #     if not tiledb.array_exists(self.array.uri):
    #         logger.info(f"Creating array {self.array.uri}")
    #         self.array.create()
    #     self.array.df_2_tiledb(
    #         df=self.data,
    #         data_types=self.columns,
    #         timeseries=self.series,
    #         level=self.level,
    #         quality_df=self.quality_df,
    #     )
    #     self.array.consolidate_fragments()
    #     self.array.vacuum_fragments()
    # else:
    #     logger.error(
    #         "Error, no local array specified.  Set with Timeseries.set_local_tdb_uri()"
    #     )

    def remove_999999s(
        self,
        interpolate: bool = False,
        method: str = "linear",
        limit_direction: str = "both",
        limit: any = None,
    ):
        """remove 999999 gap fill values from data, options to either replace with nans or interpolate

        :param interpolate: boolean of whether to interpolate across gaps using pd.DataFrame.interpolate(), defaults to False
        :type interpolate: bool, optional
        :param method: interpolation method from pd.DataFrame.interpolate(), defaults to "linear"
        :type method: str, optional
        :param limit_direction: limit direction from pd.DataFrame.interpolate(), defaults to "both"
        :type limit_direction: str, optional
        :param limit: limit from pd.DataFrame.interpolate(), defaults to None
        :type limit: any, optional
        :return: Timeseries with 999999 gap fills removed, and appropriate flags set 
        :rtype: Timeseries
        """
        logger.info("  Converting 999999 values to nan")
        if interpolate:
            df = self.data.replace(999999, np.nan).interpolate(
                method=method, limit_direction=limit_direction, limit=limit
            )
            quality_df = self.quality_df.replace("m", "i")
        else:
            df = self.data.replace(999999, np.nan)
            quality_df = self.quality_df

        return Timeseries(
            data=df,
            quality_df=quality_df,
            series=self.series,
            period=self.period,
            units=self.units,
            level=self.level,
            name=self.name,
        )

    def decimate_1s_to_300s(
        self, method: str = "linear", limit: int = 3600, name: str = None
    ):
        """decimate 1hz data to 5 min data using \n
        Agnew, Duncan Carr, and K. Hodgkinson (2007), Designing compact causal digital filters for 
        low-frequency strainmeter data , Bulletin Of The Seismological Society Of America, 97, No. 1B, 91-99

        :param method: method to interpolate across gaps, defaults to "linear"
        :type method: str, optional
        :param limit: largest gap to interpolate, defaults to 3600 samples
        :type limit: int, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: Timeseries containing 300s decimated data
        :rtype: Timeseries
        """
        
        data = decimate_1s_to_300s(self.data, method=method, limit=limit)
        quality_df = self.quality_df.copy().reindex(data.index)
        quality_df[quality_df.isna()] = "i"
        # find any differences using the original data index
        mask1 = (data.reindex(self.data.index) != self.data).any(axis=1)

        # any nans from the original index
        mask2 = self.data[mask1].isna()
        quality_df[mask2] = "i"

        # any 999999s from the original index
        mask3 = self.data[mask1] == 999999
        quality_df[mask3] = "i"
        # quality_df = self.quality_df.reindex(data.index)
        if not name:
            name = f"{self.name}.decimated"
        ts2 = Timeseries(
            data=data,
            quality_df=quality_df,
            series=self.series,
            units=self.units,
            level="1",
            period=300,
            name=name,
        )
        return ts2

    def linearize(self, reference_strains: dict, gap: float, name: str = None):
        """Processing step to convert digital counts to microstrain based on geometry of GTSM gauges

        :param reference_strains: dict containing keys of CHX and values of reference strains
        :type reference_strains: dict
        :param gap: instrument gap in meters
        :type gap: float
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: Timeseries of linearized data in microstrain
        :rtype: Timeseries
        """

        # remove any 999999 values in data, ok to leave as Nan rather than interpolate.
        if self.nines:
            logger.info(f"Found {self.nines} 999999s, replacing with nans")
            df = self.remove_999999s(interpolate=False)
        else:
            df = self.data
        linearized_data = linearize(df, reference_strains, gap)
        if not name:
            name = f"{self.name}.linearized"
        ts2 = Timeseries(
            data=linearized_data,
            quality_df=self.quality_df,
            series="microstrain",
            units="microstrain",
            level="1",
            period=self.period,
            name=name,
        )
        return ts2

    def interpolate(
        self,
        replace: int = 999999,
        method: str = "linear",
        limit_seconds: int = 3600,
        limit_direction="both",
        name: str = None,
        new_index: pd.DatetimeIndex = None,
        period=None,
        level=None,
        series=None,
    ):
        """Interpolate across gaps in data using pd.DataFrame.interpolate()

        :param replace: gap fill value to interpolate across, defaults to 999999
        :type replace: int, optional
        :param method: interpolation method, defaults to "linear"
        :type method: str, optional
        :param limit_seconds: max gap (in seconds) to interpolate , defaults to 3600
        :type limit_seconds: int, optional
        :param limit_direction: ['forward', 'backward', 'both'], defaults to "both"
        :type limit_direction: str, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :param new_index: option to manually set the index of the interpolated data, defaults to None
        :type new_index: pd.DatetimeIndex, optional
        :param period: sample rate of data in seconds, defaults to None
        :type period: float, optional
        :param level: level of data, defaults to None
        :type level: str, optional
        :param series: series name, defaults to None
        :type series: str, optional
        :return: Timeseries containing interpolated data
        :rtype: Timeseries
        """
        
        if new_index is None:
            new_index = self.data.index
            period = self.period
        elif not period:
            period = (new_index[1] - new_index[0]).total_seconds()
        
        limit = int(limit_seconds / period)  # defaults to 1 hr
        data = interpolate(
            self.data.reindex(self.data.index.union(new_index)),
            replace=replace,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
        ).reindex(new_index)
        quality_df = self.quality_df.copy().reindex(data.index)
        quality_df[quality_df.isna()] = "i"
        # find any differences using the original data index
        mask1 = (data.reindex(self.data.index) != self.data).any(axis=1)

        # any nans from the original index
        mask2 = self.data[mask1].isna()
        quality_df[mask2] = "i"

        # any 999999s from the original index
        mask3 = self.data[mask1] == 999999
        quality_df[mask3] = "i"
        if not name:
            name = f"{self.name}.interpolated"
        if not level:
            level = self.level
        if not series:
            series = self.series
        return Timeseries(
            data=data,
            quality_df=quality_df,
            series=series,
            units=self.units,
            level=level,
            period=period,
            name=name,
        )

    def butterworth_filter(
        self,
        filter_type: str,
        filter_order: int,
        filter_cutoff_s: float,
        series: str = "",
        name: str = None,
    ):
        """Apply a butterworth filter to a DataFrame using scipy.signal.butter()

        :param filter_type: {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        :type filter_type: str
        :param filter_order: the order of the filter
        :type filter_order: int
        :param filter_cutoff_s: the filter cutoff in seconds
        :type filter_cutoff_s: float
        :param series: series name, defaults to ""
        :type series: str, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: Timeseries containing filtered data
        :rtype: Timeseries
        """

        df2 = butterworth_filter(
            df=self.data,
            period=self.period,
            filter_type=filter_type,
            filter_order=filter_order,
            filter_cutoff_s=filter_cutoff_s,
        )
        if not name:
            name = f"{self.name}.filtered"
        return Timeseries(
            data=df2,
            quality_df=self.quality_df,
            series=series,
            units=self.units,
            period=self.period,
            level=self.level,
            name=name,
        )

    def decimate_to_hourly(
        self, name: str = None
    ):
        """ Decimates a timeseries to hourly by selecting the first and second and minute of each hour

        :param df: time series data to decimate
        :type df: pd.DataFrame
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: Timeseries containing hourly decimated data
        :rtype: Timeseries
        """
        data = decimate_to_hourly(self.data)
        quality_df = self.quality_df.copy().reindex(data.index)
        quality_df[quality_df.isna()] = "i"
        # find any differences using the original data index
        mask1 = (data.reindex(self.data.index) != self.data).any(axis=1)

        # any nans from the original index
        mask2 = self.data[mask1].isna()
        quality_df[mask2] = "i"

        # any 999999s from the original index
        mask3 = self.data[mask1] == 999999
        quality_df[mask3] = "i"
        # quality_df = self.quality_df.reindex(data.index)
        if not name:
            name = f"{self.name}.decimated"
        ts2 = Timeseries(
            data=data,
            quality_df=quality_df,
            series=self.series,
            units=self.units,
            level="1",
            period=3600,
            name=name,
        )
        return ts2

    def calculate_offsets(
        self,
        limit_multiplier: int = 10,
        cutoff_percentile: float = 0.75,
        name: str = None,
    ):
        """Calculate offsets using first differencing method (add more details).  

        :param limit_multiplier: _description_, defaults to 10
        :type limit_multiplier: int, optional
        :param cutoff_percentile: _description_, defaults to 0.75
        :type cutoff_percentile: float, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: _description_
        :rtype: _type_
        """
        data = calculate_offsets(self.data, limit_multiplier, cutoff_percentile)
        if not name:
            name = f"{self.name}.offset_c"
        ts = Timeseries(
            data=data,
            quality_df=self.quality_df,
            series="offset_c",
            units=self.units,
            period=self.period,
            level="2a",
            name=name,
        )
        return ts

    def apply_calibration_matrix(
        self,
        calibration_matrix: np.array,
        calibration_matrix_name: str = None,
        use_channels: list = [1, 1, 1, 1],
        name: str = None,
    ):
        """Applies a calibration matrix to convert 4 gauges into areal, differential, and shear strains

        :param calibration_matrix: calibration matrix 
        :type calibration_matrix: np.array
        :param calibration_matrix_name:  name of calibration matrix used, defaults to None
        :type calibration_matrix_name: str, optional
        :param use_channels: not yet implemented, set to 0 to ignore a bad channel, defaults to [1, 1, 1, 1]
        :type use_channels: list, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: areal, differential, and shear strains based on the given calibration
        :rtype: Timeseries
        """
        # calculate areal and shear strains from gauge strains
        # todo: implement UseChannels to arbitrary matrices
        data = apply_calibration_matrix(
            self.data, calibration_matrix, calibration_matrix_name, use_channels
        )
        quality_df = pd.DataFrame(
            index=self.quality_df.index,
            columns=data.columns,
            data="g",
        )
        quality_df[self.quality_df[self.quality_df == "m"].any(axis=1)] = "m"
        if not name:
            name = f"{self.name}.calibrated"
        ts2 = Timeseries(
            data=data,
            quality_df=quality_df,
            series=self.series,
            units="microstrain",
            level="2a",
            period=self.period,
            name=name,
        )
        return ts2

    def calculate_pressure_correction(
        self, response_coefficients: dict, name: str = None
    ):
        """Generate a pressure correction timeseries from pressure data and response coefficients

        :param response_coefficients: response coefficients for each channel loaded from metadata
        :type response_coefficients: dict
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: pressure corrections for each channel
        :rtype: Timeseries
        """
        data = calculate_pressure_correction(self.data, response_coefficients)
        quality_df = pd.DataFrame(index=data.index)
        for key in response_coefficients:
            quality_df[key] = self.quality_df
        if not name:
            name = f"{self.name}.atmp_c"
        ts2 = Timeseries(
            data=data,
            quality_df=quality_df,
            series="atmp_c",
            units="microstrain",
            level="2a",
            period=self.period,
            name=name,
        )
        return ts2

    def calculate_tide_correction(
        self, tidal_parameters: dict, longitude: float, name: str = None
    ):
        """Generate tidal correction timeseries using SPOTL hartid

        :param tidal_parameters: tidal parameters loaded from station metadata
        :type tidal_parameters: dict
        :param longitude: station longitude
        :type longitude: float
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: tidal correction Timeseries calculated for each column/channel in input data
        :rtype: Timeseries
        """
        data = calculate_tide_correction(
            self.data, self.period, tidal_parameters, longitude
        )
        if not name:
            name = f"{self.name}.tide_c"
        ts = Timeseries(
            data=data,
            series="tide_c",
            units="microstrain",
            period=self.period,
            level="2a",
            name=name,
        )
        return ts

    def linear_trend_correction(
        self, method='linear',trend_start=None, trend_end=None, name: str = None
    ):
        """Generate a linear trend correction
        :param method: linear or median
        :type method: str, defaults to linear
        :param trend_start: start of window to calculate trend, defaults to first_valid_index()
        :type trend_start: datetime.datetime, optional
        :param trend_end: end of window to calculate trend, defaults to last_valid_index()
        :type trend_end: datetime.datetime, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: trend correction timeseries for each column/channel in input data
        :rtype: Timeseries
        """
        data = calculate_linear_trend_correction(self.data, method, trend_start, trend_end)
        if not name:
            name = f"{self.name}.trend_c"
        ts = Timeseries(
            data=data,
            series="trend_c",
            units="microstrain",
            period=self.period,
            level="2a",
            name=name,
        )
        return ts

    def apply_corrections(self, corrections: list = [], name: str = None):
        """applies one or more corrections to a Timeseries and returns the corrected Timeseries

        :param corrections: List of correction Timeseries to apply, defaults to []
        :type corrections: list, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: corrected Timeseries
        :rtype: Timeseries
        """
        logger.info(f"Applying corrections")
        if len(corrections):
            data = self.data.copy()
            for ts in corrections:
                data -= ts.data
        if not name:
            name = f"{self.name}.corrected"
        ts2 = Timeseries(
            data=data,
            quality_df=self.quality_df,
            series="corrected",
            units="microstrain",
            level="2a",
            period=self.period,
            name=name,
        )
        return ts2

    def dynamic_strain(
        self,
        gauge_weights: list = [1, 1, 1, 1],
        series="dynamic",
        name=None,
    ):
        """calculates dynamic strain for a given Timeseries as RMS of gauge strains

        :param gauge_weights: list of which channels to use, defaults to [1, 1, 1, 1]
        :type gauge_weights: list, optional
        :param series: series name, defaults to "dynamic"
        :type series: str, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: calculated dynamic strain as a Timeseries object
        :rtype: Timeseries
        """
        df2 = dynamic_strain(self.data, gauge_weights)
        quality_df = pd.DataFrame(index=self.data.index)
        quality_df["dynamic"] = "g"
        quality_df[self.quality_df[(self.quality_df == "m")].any(axis=1)] = "m"
        quality_df[self.quality_df[(self.quality_df == "i")].any(axis=1)] = "i"
        if not name:
            name = f"{self.name}.dynamic"
        return Timeseries(
            data=df2,
            quality_df=quality_df,
            series=series,
            units=self.units,
            period=self.period,
            level=self.level,
            name=name,
        )

    def calculate_magnitude(
        self,
        hypocentral_distance,
        site_term: float = 0,
        longitude_term=0,
        name: str = None,
    ):
        """Calculates a magnitude estimate based on Barbour et al 2021.
        :param hypocentral_distance: distance from station to event hypocenter, in km
        :type hypocentral_distance: float
        :param site_term: site term from Barbour et al 2021, defaults to 0
        :type site_term: float, optional
        :param longitude_term: longitude term from Barbour et al 2021, defaults to 0
        :type longitude_term: int, optional
        :param name: name for new Timeseries, defaults to None
        :type name: str, optional
        :return: Timeseries containing a dynamic strain based magnitude estimate as a function of time
        :rtype: Timeseries
        """
        data = calculate_magnitude(
            self.data, hypocentral_distance, site_term, longitude_term
        )
        if not name:
            name = f"{self.name}.magnitude"
        return Timeseries(
            data=data,
            quality_df=self.quality_df,
            series="magnitude",
            units="magnitude",
            period=self.period,
            level=self.level,
            name=name,
        )

    def baytap_analysis(self,atmp_ts,latitude=None,longitude=None,elevation=None,dmin=0.001):
        '''
        This function accesses a docker container to run BAYTAP08 (Tamura 1991; Tamura and Agnew 2008) for tidal analysis. Time series (e.g. strain) and additional auxiliary input (e.g. pressure) are analyzed together to determine the amplitudes and phases of a combination of tidal constituents (M2, O1, P1, K1, N2, S2) in the time series, as well as a coefficient for the auxiliary input response. 
        :param atmp_ts: Atmospheric pressure time series with same sample period and time frame as the strain data.
        :type atmp_ts: Timeseries
        :param latitude: latitude of the station
        :type latitude: float
        :param longitude: longitude of the station
        :type longitude: float
        :param elevation: elevation of the station
        :type elevation: float
        :param dmin: Drift parameter for the program. Large drift expects a linear trend. Small drift allows for rapid changes in the residual time series. 
        :type dmin: float
        :return: Dictionary of amplitudes and phases for each tidal constituent per gauge, and atmospheric pressure coefficient.
        :rtype: dict
        '''

        baytap_results = baytap_analysis(df=self.data,
                                         atmp_df=atmp_ts.data,
                                         quality_df=self.quality_df,
                                         atmp_quality_df=atmp_ts.quality_df,
                                         latitude=latitude,
                                         longitude=longitude,
                                         elevation=elevation,
                                         dmin=0.001)

        return baytap_results

    def strain_video(
        self,
        start:str=None,
        end:str=None,
        skip:int=1,
        interval:float=None,
        title:str=None,
        units:str=None,
        savegif:str=None
        ):
        """Displays a gif of the strain time series provided, with time series and strain axes displayed. Strain is shown relative to the first data point. 
        :param start: (Optional) Start of the video as a datetime string.
        :type start: str
        :param end: (Optional) End of the video as a datetime string.
        :type end: str
        param skip: (optional) number of data points to skip per frame (eg. if using 5 minute Timeseries, skip=2 will decimate the dataset to a 10 minute period)
        :type skip: int
        :param interval: (Optional) Time between frames (in microseconds). 
        :type interval:
        :param title: (Optional) Plot title
        :type title: str
        :param repeat: (Optional) Choose if the animation repeats. Defaults to false.
        :type repeat: bool
        :param units: (Optional) Units to label strain
        :type units: str 
        :return: Gif of the strain time series
        :rtype: matplotlib.animation
        
        Example
        -------
        >>> # Import relevant modules from the earscopestraintools package
        >>> from earthscopestraintools.mseed_tools import ts_from_mseed
        >>> from earthscopestraintools.gtsm_metadata import GtsmMetadata
        >>> # Metadata
        >>> network = 'PB'
        >>> station = 'B004' 
        >>> meta = GtsmMetadata(network,station)
        >>> # Provide the start and end times 
        >>> start = '2019-07-01'
        >>> end = '2019-07-07'
        >>> 
        >>> # load data
        >>> strain_raw = ts_from_mseed(network=network, station=station, location='T0', channel='RS*', start=start, end=end)
        >>> strain_linearized = strain_raw.linearize(reference_strains=meta.reference_strains,gap=meta.gap)
        >>> strain_reg = strain_linearized.apply_calibration_matrix(calibration_matrix=meta.strain_matrices['ER2010'])
        >>> # make video, save .gif
        >>> %matplotlib widget 
        >>> anim = strain_reg.strain_video(interval=1, title=f'{station}, One Week',units='ms',savegif=f'{station}.{start}.{end}.gif')
        """
        anim = strain_video(self.data,start=start,end=end,skip=skip,interval=interval,title=title,units=units,savegif=savegif)
        return anim

    def plot(
        self,
        title: str = None,
        remove_9s: bool = False,
        zero: bool = False,
        detrend: str = None,
        ymin: float = None,
        ymax: float = None,
        type: str = "line",
        show_quality_flags: bool = False,
        atmp=None,
        rainfall=None,
        save_as: str = None,
    ):
        """Generic plotting function for Timeseries data

        :param title: plot title, defaults to None
        :type title: str, optional
        :param remove_9s: option to remove gap fill values, defaults to False
        :type remove_9s: bool, optional
        :param zero: option to zero against first_valid_index(), defaults to False
        :type zero: bool, optional
        :param detrend: signal.detrend type, only 'linear' implented currently, defaults to None
        :type detrend: str, optional
        :param ymin: y-axis minimum for plot, defaults to None
        :type ymin: float, optional
        :param ymax: y-axis maximum for plot, defaults to None
        :type ymax: float, optional
        :param type: matplotlib plot type. option of ['scatter','line'], defaults to "line"
        :type type: str, optional
        :param show_quality_flags: option to highlight missing data flags, defaults to False
        :type show_quality_flags: bool, optional
        :param atmp: optional Timeseries containing atmospheric pressure data to be plotted in an extra subplot, defaults to None
        :type atmp: Timeseries, optional
        :param rainfall: optional Timeseries containing rainfall data to be plotted in an extra subplot.  will also plot cumsum of rainfall during time window. defaults to None
        :type rainfall: Timeseries, optional
        :param save_as: filename to save as, defaults to None
        :type save_as: str, optional
        """
        num_plots = len(self.columns)
        num_colors = 1
        if atmp is not None:
            num_plots += 1
        if rainfall is not None:
            num_plots += 1
            num_colors += 1
        fig, axs = plt.subplots(
            num_plots, 1, figsize=(12, 3 * num_plots), squeeze=False
        )
        colors = [cm.gnuplot(x) for x in np.linspace(0, 0.8, num_colors)]
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(self.name)
        if remove_9s:
            df = self.remove_999999s(interpolate=False).data
        else:
            df = self.data.copy()
        if zero:
            df -= df.loc[df.first_valid_index()]
        if detrend:
            if detrend == "linear":
                for ch in self.columns:
                    df[ch] = signal.detrend(df[ch], type="linear")
            else:
                logger.error("Only linear detrend implemented")
        for i, ch in enumerate(self.columns):
            if type == "line":
                axs[i][0].plot(df[ch], color=colors[0], label=ch)
            elif type == "scatter":
                axs[i][0].scatter(df.index, df[ch], color=colors[0], s=2, label=ch)
            else:
                logger.error("Plot type must be either 'line' or 'scatter'")
            if self.units:
                axs[i][0].set_ylabel(self.units)
            if ymin or ymax:
                axs[i][0].set_ylim(ymin, ymax)
            axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
            if show_quality_flags:
                missing = self.quality_df[self.quality_df[ch] == "m"].index
                if len(missing) and len(missing) < 100:
                    axs[i][0].text(
                        0.02, 0.95, "missing", color="r", transform=axs[i][0].transAxes
                    )
                    for time in missing:
                        axs[i][0].axvline(time, color="r")
                elif len(missing) > 100:
                    logger.info("too many missing points to plot")
                interpolated = self.quality_df[self.quality_df[ch] == "i"].index
                if len(interpolated) and len(interpolated) < 100:
                    axs[i][0].text(
                        0.08,
                        0.95,
                        "interpolated",
                        color="b",
                        transform=axs[i][0].transAxes,
                    )
                    for time in interpolated:
                        axs[i][0].axvline(time, color="b")
                elif len(interpolated) > 100:
                    logger.info("too many interpolated points to plot")
            axs[i][0].legend()
        if atmp is not None:
            i += 1
            if type == "line":
                axs[i][0].plot(atmp.data, color=colors[0], label="atmp")
            elif type == "scatter":
                axs[i][0].scatter(
                    atmp.data.index, atmp.data, color=colors[0], s=2, label="atmp"
                )
            else:
                logger.error("Plot type must be either 'line' or 'scatter'")
            if atmp.units:
                axs[i][0].set_ylabel(atmp.units)
            axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
            axs[i][0].legend()
        if rainfall is not None:
            i += 1
            ax2 = axs[i][0].twinx()
            axs[i][0].bar(
                rainfall.data.index,
                rainfall.data.values.reshape(len(rainfall.data)),
                width=0.02,
                color=colors[0],
                label="rainfall",
            )

            # axs[i][0].scatter(
            #     rainfall.data.index,
            #     rainfall.data,
            #     color=colors[0],
            #     s=2,
            #     label="rainfall",
            # )
            ax2.plot(
                rainfall.data.cumsum(), color=colors[1], label="cumulative rainfall"
            )

            axs[i][0].set_ylabel("mm/30m")
            ax2.set_ylabel("mm")
            axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
            lines, labels = axs[i][0].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)

        fig.tight_layout()
        if save_as:
            logger.info(f"Saving plot to {save_as}")
            plt.savefig(save_as)


def plot_timeseries_comparison(
    timeseries: list = [],
    title: str = None,
    names: list = [],
    remove_9s: bool = False,
    zero: bool = False,
    detrend: str = None,
    type: str = "line",
    save_as: str = None,
):
    """plot multiple Timeseries in the same plot to compare values.
       useful for viewing uncorrected vs corrected data

    :param timeseries: list of Timeseries to plot, defaults to []
    :type timeseries: list, optional
    :param title: plot title, defaults to None
    :type title: str, optional
    :param names: list of names to use in legend, defaults to []
    :type names: list, optional
    :param remove_9s: option to remove gap fill values, defaults to False
    :type remove_9s: bool, optional
    :param zero: option to zero against first_valid_index(), defaults to False
    :type zero: bool, optional
    :param detrend: signal.detrend type, only 'linear' implented currently, defaults to None
    :type detrend: str, optional
    :param type: matplotlib plot type. option of ['scatter','line'], defaults to "line"
    :type type: str, optional
    :param save_as: filename to save as, defaults to None
    :type save_as: str, optional
    """
    if not isinstance(timeseries, list):
        timeseries = [timeseries]
    colors = [cm.gnuplot(x) for x in np.linspace(0, 0.8, len(timeseries))]

    fig, axs = plt.subplots(
        len(timeseries[0].columns),
        1,
        figsize=(12, 3 * len(timeseries[0].columns)),
        squeeze=False,
    )
    for j, ts in enumerate(timeseries):
        if title:
            fig.suptitle(title)
        if remove_9s:
            df = ts.remove_999999s(interpolate=False).data
        else:
            df = ts.data.copy()
        if zero:
            df -= df.loc[df.first_valid_index()]
        if detrend:
            if detrend == "linear":
                for ch in ts.columns:
                    df[ch] = signal.detrend(df[ch], type="linear")
            else:
                logger.error("Only linear detrend implemented")
        for i, ch in enumerate(ts.columns):
            if len(names) > j:
                label = f"{names[j]}"
            else:
                label = ch
            if type == "line":
                axs[i][0].plot(df[ch], color=colors[j], label=label)
            elif type == "scatter":
                axs[i][0].scatter(df.index, df[ch], color=colors[j], s=2, label=label)
            else:
                logger.error("Plot type must be either 'line' or 'scatter'")
            if ts.units:
                axs[i][0].set_ylabel(ts.units)
            axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
            axs[i][0].set_title(ch)
            axs[i][0].legend()

        fig.tight_layout()
        if save_as:
            logger.info(f"Saving plot to {save_as}")
            plt.savefig(save_as)
