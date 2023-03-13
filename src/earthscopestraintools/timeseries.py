import os
import numpy as np
import pandas as pd
import tiledb
import datetime
from earthscopestraintools.mseed_tools import load_mseed_to_df
from earthscopestraintools.edid import get_station_edid, get_session_edid
from earthscopestraintools.tiledbtools import (
    StrainArray,
    RawStrainWriter,
    RawStrainReader,
    ProcessedStrainWriter,
    ProcessedStrainReader,
)
from scipy import signal
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def test():
    print("test1")


class Timeseries:
    def __init__(
        self,
        data: pd.DataFrame = None,
        quality_df: pd.DataFrame = None,
        # level_df: pd.DataFrame = None,
        # version_df: pd.DataFrame = None,
        series: str = "",
        units: str = "",
        level: str = "",
        period: float = 0,
        name: str = None,
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

        # if level_df is not None:
        #     self.level_df = level_df
        # # else:
        # #     self.level_df = self.set_initial_level_flags(level)
        #
        # if version_df is not None:
        #     self.version_df = version_df
        # # else:
        # #     self.version_df = self.set_initial_version()

        self.series = series
        self.units = units
        self.level = level
        self.period = period
        if name:
            self.name = name
        else:
            self.name = ""
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
        qual_df = pd.DataFrame(index=self.data.index)
        for ch in self.columns:
            qual_df[ch] = "g"
            qual_df[ch][self.data[ch] == missing_data] = "m"

        # print(qual_df.value_counts())
        return qual_df

    def set_initial_level_flags(self, level):
        level_df = pd.DataFrame(index=self.data.index)
        for ch in self.columns:
            level_df[ch] = level
        return level_df

    def set_initial_version(self):
        version_df = pd.DataFrame(index=self.data.index)
        version = int(datetime.datetime.now().strftime("%Y%j%H%M%S"))
        for ch in self.columns:
            version_df[ch] = version
        return version_df

    def check_for_gaps(self):
        if self.period != 0 and self.data is not None:
            days = (self.data.index[-1] - self.data.index[0]).days
            seconds = (self.data.index[-1] - self.data.index[0]).seconds
            self.nans = int(self.data.isna().sum().sum() / len(self.data.columns))
            self.nines = round(
                self.data[self.data == 999999].count().sum() / len(self.data.columns), 2
            )
            expected_samples = int((days * 86400 + seconds) / self.period) + 1
            self.samples = len(self.data) - self.nans - self.nines
            # print(self.nans, self.nines, self.samples, expected_samples)
            self.gap_percentage = round(
                (1 - (self.samples / expected_samples)) * 100, 2
            )

            logger.info(
                f"Found {self.nans} nans, {self.nines} 999999s, and "
                f"{expected_samples - len(self.data)} gaps.  Total missing data is "
                f"{self.gap_percentage}%"
            )
        else:
            self.gaps = None

    def stats(self):
        if len(self.data):
            outputstring = f"{self.name:30} | Channels: {str(self.columns):40} "
            outputstring += (
                f"\n{'':30} | TimeRange: {self.data.index[0]} - {self.data.index[-1]} "
            )
            outputstring += f"\n{'':30} | Period: {self.period:9}s | Samples: {len(self.data):10} | Gaps: {self.gap_percentage:4}% "
            outputstring += f"\n{'':30} | Series: {self.series:10} | Units: {self.units:12} | Level: {self.level:4}\n"
            logger.info(f"{outputstring}")

    def save_csv(self, filename: str, datadir: str = "./", sep=",", compression=None):
        filepath = os.path.join(datadir, filename)
        logger.info(f"saving to {filepath}")
        if compression:
            self.data.to_csv(filepath, sep=sep, compression={"method": compression})
        else:
            self.data.to_csv(filepath, sep=sep)

    def save_tdb(self):
        # TODO: handle quality fields
        if self.local_tdb_uri:
            logger.info(f"saving to {self.local_tdb_uri}")
            self.array = StrainTiledbArray(
                uri=self.local_tdb_uri, period=self.period, location="local"
            )
            if not tiledb.array_exists(self.array.uri):
                logger.info(f"Creating array {self.array.uri}")
                self.array.create()
            self.array.df_2_tiledb(
                df=self.data,
                data_types=self.columns,
                timeseries=self.series,
                level=self.level,
                quality_df=self.quality_df,
            )
            self.array.consolidate_fragments()
            self.array.vacuum_fragments()
        else:
            logger.error(
                "Error, no local array specified.  Set with Timeseries.set_local_tdb_uri()"
            )

    def remove_999999s(
        self,
        interpolate: bool = False,
        method: str = "linear",
        limit_direction: str = "both",
        limit: any = None,
    ):
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

    def plot(
        self,
        title: str = None,
        remove_9s: bool = False,
        zero: bool = False,
        detrend: str = None,
        ymin: float = None,
        ymax: float = None,
        type: str = "scatter",
        save_as: str = None,
    ):
        fig, axs = plt.subplots(len(self.columns), 1, figsize=(12, 10), squeeze=False)
        if title:
            fig.suptitle(title)
        # else:
        #     if self.period < 1:
        #         sample_rate = f"{int(1/self.period)}hz"
        #     else:
        #         sample_rate = f"{self.period}s"
        #     fig.suptitle(f"{self.metadata.network}_{self.metadata.fcid}_{sample_rate}_{self.series} ")
        if remove_9s:
            df = self.remove_999999s(interpolate=False).data
        else:
            df = self.data.copy()
        if zero:
            # df = df - df.iloc[0]
            df -= df.loc[df.first_valid_index()]
        if detrend:
            if detrend == "linear":
                for ch in self.columns:
                    df[ch] = signal.detrend(df[ch], type="linear")
            else:
                logger.error("Only linear detrend implemented")
        for i, ch in enumerate(self.columns):
            if type == "line":
                axs[i][0].plot(df[ch], color="black", label=ch)
            elif type == "scatter":
                axs[i][0].scatter(df.index, df[ch], color="black", s=2, label=ch)
            else:
                logger.error("Plot type must be either 'line' or 'scatter'")
            if self.units:
                axs[i][0].set_ylabel(self.units)
            if ymin or ymax:
                axs[i][0].set_ylim(ymin, ymax)
            axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
            axs[i][0].legend()
        fig.tight_layout()
        if save_as:
            logger.info(f"Saving plot to {save_as}")
            plt.savefig(save_as)


def ts_from_mseed(
    network: str,
    station: str,
    location: str,
    channel: str,
    start: str,
    end: str,
    name: str = None,
):
    df = load_mseed_to_df(network, station, location, channel, start, end)
    level = "0"
    if channel.startswith("RS"):
        period = 600
        series = "raw"
        units = "counts"
    elif channel.startswith("LS"):
        period = 1
        series = "raw"
        units = "counts"
    elif channel.startswith("BS"):
        period = 0.05
        series = "raw"
        units = "counts"
    else:
        period = 0
        series = ""
        units = ""
    if name == None:
        name = f"{network}.{station}.{location}.{channel}"
    ts = Timeseries(
        data=df, series=series, units=units, level=level, period=period, name=name
    )
    logger.info("Converting missing data from 999999 to nan")
    return ts.remove_999999s()


def ts_from_tiledb_raw(
    network: str,
    station: str,
    channels: list,
    period: float,
    uri: str = None,
    start_ts: int = None,
    end_ts: int = None,
    start_str: str = None,
    end_str: str = None,
    start_dt: datetime.datetime = None,
    end_dt: datetime.datetime = None,
    name: str = None,
):
    if not uri:
        uri = lookup_s3_uri(network, station, period)
    reader = RawStrainReader(uri, period)
    df = reader.to_df(
        channels=channels,
        start_str=start_str,
        end_str=end_str,
        start_ts=start_ts,
        end_ts=end_ts,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    series = "raw"
    units = "counts"
    level = "0"
    if name == None:
        name = f"{network}.{station}.{series}.{units}"
    ts = Timeseries(
        data=df, series=series, units=units, level=level, period=period, name=name
    )
    logger.info("Converting missing data from 999999 to nan")
    return ts.remove_999999s()



def lookup_s3_uri(network, station, period):
    bucket = "tiledb-strain"
    if period == 600:
        session = "Day"
    elif period == 1:
        session = "Hour"
    elif period == 0.05:
        session = "Min"
    edid = get_session_edid(network, station, session)
    uri = f"s3://{bucket}/{edid}.tdb"
    return uri
