import os
import numpy as np
import pandas as pd
import tiledb
import datetime
from earthscopestraintools.gtsm_metadata import GtsmMetadata
from earthscopestraintools.mseed_tools import load_mseed_to_df
from earthscopestraintools.edid import get_station_edid, get_session_edid
from earthscopestraintools.processing import (
    linearize,
    interpolate,
    butterworth_filter,
    apply_calibration_matrix,
    calculate_offsets,
    calculate_pressure_correction,
    calculate_tide_correction,
    calculate_linear_trend_correction,
)
from earthscopestraintools.event_processing import dynamic_strain
from earthscopestraintools.tiledbtools import (
    StrainArray,
    RawStrainWriter,
    RawStrainReader,
    ProcessedStrainWriter,
    ProcessedStrainReader,
)
from scipy import signal, stats
import matplotlib.pyplot as plt
from matplotlib import cm
import logging

logger = logging.getLogger(__name__)


def test():
    print("test1")


class Timeseries:
    def __init__(
        self,
        data: pd.DataFrame = None,
        quality_df: pd.DataFrame = None,
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
            self.gap_percentage = None

    def stats(self):
        if len(self.data):
            outputstring = f"{self.name}\n{'':6} | Channels: {str(self.columns):40} "
            outputstring += (
                f"\n{'':6} | TimeRange: {self.data.index[0]} - {self.data.index[-1]} "
            )
            outputstring += f"\n{'':6} | Period: {self.period:9}s | Epochs: {len(self.data):10} | Gaps: {self.gap_percentage:4}% "
            outputstring += f"\n{'':6} | Series: {self.series:10} | Units: {self.units:12} | Level: {self.level:4}\n"
            logger.info(f"{outputstring}")

    def quality_stats(self):
        if len(self.quality_df):
            outputstring = f"{self.name}\n{'':6} | Channels: {str(self.columns):43} "
            cols = len(self.quality_df.columns)
            outputstring += f"\n{'':6} | Epochs: {len(self.quality_df):9}"
            outputstring += f"| Good: {(self.quality_df == 'g').sum().sum() / cols:10}"
            outputstring += (
                f"| Missing: {(self.quality_df == 'm').sum().sum() / cols:8}"
            )
            outputstring += (
                f"| Interpolated: {(self.quality_df == 'i').sum().sum() / cols:8}"
            )

            outputstring += f"\n{'':6} | Samples: {len(self.quality_df) * cols:8}"
            outputstring += f"| Good: {(self.quality_df == 'g').sum().sum():10}"
            outputstring += f"| Missing: {(self.quality_df == 'm').sum().sum():8}"
            outputstring += f"| Interpolated: {(self.quality_df == 'i').sum().sum():8}"
            logger.info(f"{outputstring}")

    def save_csv(self, filename: str, datadir: str = "./", sep=",", compression=None):
        filepath = os.path.join(datadir, filename)
        logger.info(f"saving to {filepath}")
        if compression:
            self.data.to_csv(filepath, sep=sep, compression={"method": compression})
        else:
            self.data.to_csv(filepath, sep=sep)

    def save_tdb(self, uri):
        # TODO: update, handle quality fields
        if self.local_tdb_uri:
            logger.info(f"saving to {self.local_tdb_uri}")
            self.array = StrainArray(uri=self.local_tdb_uri, period=self.period)
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

    def linearize(self, reference_strains: dict, gap: float, name: str = None):
        """
        Processing step to convert digital counts to microstrain based on geometry of GTSM gauges
        :param ts: Timeseries object, containing data to convert to microstrain
        :param reference_strains: dict, containing keys of CHX and values of reference strains
        :param gap: float, instrument gap
        :return: Timeseries object, in units of microstrain
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
        if new_index is None:
            new_index = self.data.index
            period = self.period
        elif not period:
            period = (new_index[1] - new_index[0]).total_seconds()

        limit = int(limit_seconds / period)  # defaults to 1 hr
        data = interpolate(
            self.data.reindex(new_index),
            replace=replace,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
        )
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

    def calculate_offsets(
        self,
        limit_multiplier: int = 10,
        cutoff_percentile: float = 0.75,
        name: str = None,
    ):
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

    def dynamic_strain(
        self,
        gauge_weights: list = [1, 1, 1, 1],
        series="dynamic",
        name=None,
    ):
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

    def apply_calibration_matrix(
        self,
        calibration_matrix: np.array,
        use_channels: list = [1, 1, 1, 1],
        name: str = None,
    ):
        """
        Processing step to convert gauge strains into areal and shear strains
        :param ts: Timeseries object containing gauge data in microstrain
        :param calibration_matrix: np.array containing strain matrix
        :return: Timeseries object, in units of microstrain
        """
        # calculate areal and shear strains from gauge strains
        logger.info(f"Applying matrix: {calibration_matrix}")
        # todo: implement UseChannels to arbitrary matrices
        data = apply_calibration_matrix(self.data, calibration_matrix, use_channels)
        quality_df = pd.DataFrame(
            index=self.quality_df.index,
            columns=["Eee+Enn", "Eee-Enn", "2Ene"],
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
        self, trend_start=None, trend_end=None, name: str = None
    ):
        data = calculate_linear_trend_correction(self.data, trend_start, trend_end)
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


def ts_from_mseed(
    network: str,
    station: str,
    location: str,
    channel: str,
    start: str,
    end: str,
    name: str = None,
    period: float = None,
    series: str = "raw",
    units: str = "",
    to_nan: bool = True,
    scale_factor: float = None,
):
    df = load_mseed_to_df(
        net=network,
        sta=station,
        loc=location,
        cha=channel,
        start=start,
        end=end,
    )
    level = "0"
    if channel.startswith("RS"):
        period = 600
        series = series
        units = "counts"
    elif channel.startswith("LS"):
        period = 1
        series = series
        units = "counts"
    elif channel.startswith("BS"):
        period = 0.05
        series = series
        units = "counts"
    else:
        period = period
        series = series
    if name is None:
        name = f"{network}.{station}.{location}.{channel}"
    ts = Timeseries(
        data=df, series=series, units=units, level=level, period=period, name=name
    )
    if to_nan:
        logger.info("Converting missing data from 999999 to nan")
        ts = ts.remove_999999s()
    if scale_factor:
        ts.data = ts.data * scale_factor
    return ts


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
