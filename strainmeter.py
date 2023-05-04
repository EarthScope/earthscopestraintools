import os
import sys
import subprocess

import numpy as np
import pandas as pd
import math
from scipy import signal, interpolate, stats
from scipy.signal import lfilter
import datetime
import matplotlib.pyplot as plt
import json

import logging

logger = logging.getLogger(__name__)

from obspy import read, Stream
from obspy.clients.fdsn import Client

inv_client = Client("IRIS")
from requests.exceptions import HTTPError

from gtsm_metadata import GtsmMetadata, fdsn2bottlename

# from tiledbarray import StrainTiledbArray
from timeseries import Timeseries


class Strainmeter:
    def __init__(
        self,
        network: str,
        fcid: str,
        parameters: dict = None,
        load_metadata: bool = False,
        instrument_type: str = "GTSM",
        local_archive: str = None,
    ):
        self.network = network
        self.fcid = fcid
        self.parameters = parameters
        self.instrument_type = instrument_type
        logger.info(
            f"Building strainmeter object for {instrument_type} {network} {fcid}"
        )
        if self.instrument_type == "GTSM":
            if load_metadata:
                logger.info("Loading metadata from UNAVCO")
                self.metadata = GtsmMetadata(self.network, self.fcid)
        else:
            logger.error("No valid metadata specified.")
            self.metadata = None
        self.ts = {}

        self.stationdir = f"results/{network}_{fcid}"

        if parameters.get("YamlBaseName", None):
            basename = parameters["YamlBaseName"]
        else:
            basename = ""
        self.workdir = f"{self.stationdir}/{basename}_{datetime.datetime.now().isoformat(timespec='seconds')}"
        os.makedirs(self.workdir, exist_ok=True)

        self.plotdir = f"{self.workdir}/plots"
        os.makedirs(self.plotdir, exist_ok=True)

        self.local_archive = f"{self.stationdir}/data"
        os.makedirs(self.local_archive, exist_ok=True)

        # datadir = '.'
        # if parameters.get('SaveParameters', {}).get('DataDir', None):
        #     datadir = parameters['SaveParameters']['DataDir']
        # os.makedirs(datadir, exist_ok=True)
        # self.local_archive = datadir

    def set_local_archive(self, local_archive="."):
        # update where local data is written. defaults to working directory.
        self.local_archive = local_archive
        os.makedirs(self.local_archive, exist_ok=True)

    def set_plot_dir(self, plotdir="."):
        # update where local data is written. defaults to working directory.
        self.plotdir = plotdir
        os.makedirs(self.plotdir, exist_ok=True)

    def set_use_ch(self, use_ch):
        self.use_ch = use_ch
        self.set_gauge_weights()

    def set_gauge_weights(self):
        self.gauge_weights = []
        for i in ["CH0", "CH1", "CH2", "CH3"]:
            if i in self.use_ch:
                self.gauge_weights.append(1)
            else:
                self.gauge_weights.append(0)
        logger.info(f"Gauge Weights: {self.gauge_weights}")

    def list_ts(self):

        # remove any empty timeseries resulting from failed queries
        filtered = {k: v for k, v in self.ts.items() if v is not None}
        self.ts.clear()
        self.ts.update(filtered)
        outputstring = ""
        for name in self.ts:
            ts = self.ts[name]
            if len(ts.data):
                outputstring += f"{name:30} | Channels: {str(ts.columns):40} "
                outputstring += (
                    f"\n{'':30} | TimeRange: {ts.data.index[0]} - {ts.data.index[-1]} "
                )
                outputstring += f"\n{'':30} | Period: {ts.period:9}s | Samples: {len(ts.data):10} | Gaps: {ts.gap_percentage:4}% "
                outputstring += f"\n{'':30} | Series: {ts.series:10} | Units: {ts.units:12} | Level: {ts.level:4}\n"
        logger.info(
            f"{len(self.ts)} timeseries for {self.network} {self.fcid}:\n{outputstring}"
        )

    def _print_plot_save(self, ts: Timeseries, step_params: dict):
        if hasattr(ts, "data"):
            if "Print" not in step_params:
                if self.parameters.get("Print", None):
                    logger.info(f"{step_params.get('Name','')}\n{ts.data}")
            elif step_params.get("Print", None):
                logger.info(f"\n{ts.data}")
            if "Plot" not in step_params:
                if self.parameters.get("Plot", None):
                    self._plot_ts(ts, step_params)
            elif step_params.get("Plot", None):
                self._plot_ts(ts, step_params)
            if "Save" not in step_params:
                if self.parameters.get("Save", None):
                    self._save_ts(ts, step_params)
            elif step_params.get("Save", None):
                self._save_ts(ts, step_params)

    def _plot_ts(
        self, ts: Timeseries, step_params: dict,
    ):

        if step_params.get("Name", None):
            default_name = f'{self.network}_{self.fcid}_{step_params["Name"]}'
        else:
            default_name = f"{self.network}_{self.fcid}"

        if "Type" in step_params.get("PlotParameters", {}):
            plot_type = step_params["PlotParameters"]["Type"]
        elif "Type" in self.parameters.get("PlotParameters", {}):
            plot_type = self.parameters["PlotParameters"]["Type"]
        else:
            plot_type = "scatter"

        if "Zero" not in step_params.get("PlotParameters", {}):
            zero = self.parameters.get("PlotParameters", {}).get("Zero", False)
        elif step_params.get("PlotParameters", {}).get("Zero", None):
            zero = self.step_params["PlotParameters"]["Zero"]
        else:
            zero = False

        if "Detrend" in step_params.get("PlotParameters", {}):
            detrend = step_params["PlotParameters"]["Detrend"]
        elif "Detrend" in self.parameters.get("PlotParameters", {}):
            detrend = self.parameters["PlotParameters"]["Detrend"]
        else:
            detrend = None

        if "Title" in step_params.get("PlotParameters", {}):
            title = step_params["PlotParameters"]["Title"]
        # elif 'Title' in self.parameters.get('PlotParameters', {}):
        #    title = self.parameters['PlotParameters']['Title']
        else:
            title = default_name

        if "SaveAs" in step_params.get("PlotParameters", {}):
            plotname = step_params["PlotParameters"]["SaveAs"]
        # elif 'SaveAs' in self.parameters.get('PlotParameters', {}):
        #    plotname = self.parameters['PlotParameters']['SaveAs']
        else:
            plotname = f"{default_name}.png"
        save_as = f"{self.plotdir}/{plotname}"

        ts.plot(title=title, zero=zero, type=plot_type, save_as=save_as)

    def _save_ts(
        self, ts: Timeseries, step_params: dict,
    ):

        save_as_type = None
        if self.parameters.get("SaveParameters", {}).get("Type", None):
            save_as_type = self.parameters["SaveParameters"]["Type"]
        if step_params.get("SaveParameters", {}).get("Type", None):
            save_as_type = self.parameters["SaveParameters"]["Type"]

        if step_params.get("Name", None):
            default_name = f'{self.network}_{self.fcid}_{step_params["Name"]}'
        else:
            default_name = f"{self.network}_{self.fcid}"
        start_str = datetime.datetime.strftime(ts.data.index[0], "%Y-%m-%d")
        end_str = datetime.datetime.strftime(ts.data.index[-1], "%Y-%m-%d")
        filename = f"{default_name}_{start_str}_{end_str}.csv"

        if save_as_type == "csv":
            try:
                if "Compression" in step_params.get("SaveParameters", {}):
                    compression = step_params["SaveParameters"]["Compression"]
                elif "Compression" in self.parameters.get("SaveParameters", {}):
                    compression = self.parameters["SaveParameters"]["Compression"]
                else:
                    compression = None
                if "SaveAs" in step_params.get("SaveParameters", {}):
                    filename = step_params["SaveParameters"]["SaveAs"]
                if compression == "gzip":
                    if not filename.endswith(".gz"):
                        filename += ".gz"
                ts.save_csv(
                    datadir=self.local_archive,
                    filename=filename,
                    compression=compression,
                )

            except Exception as e:
                logger.error(f"Unable to save data to csv")

    def load_metadata(self, step_params: dict):
        """
        Processing step to load metadata
        :param step_params: dict. Required key is Source, Optional key is Print.
        :return:
        """
        if step_params.get("Source", {}).get("Format", None) == "unavco-ascii":
            self.metadata = GtsmMetadata(self.network, self.fcid)
            if "Print" not in step_params:
                if self.parameters.get("Print", None):
                    self.metadata.show()
            elif step_params.get("Print", None):
                self.metadata.show()
        else:
            logger.error(
                'Unable to load metadata.  Current implementation requires Source.Format to be "unavco-ascii".'
            )

    def load_data(self, step_params):
        """
        Processing step to load data.
        Source Formats currently implemented are miniseed (from fdsn dataselect-ws) and local csv.
        :param step_params: dict, defines what data to load
        :return: Timeseries, timeseries object containing specified data
        """
        if step_params.get("Source", {}).get("Format", None) == "miniseed":
            try:
                ts = self._load_miniseed(
                    location=step_params["Source"]["LocationCode"],
                    channel=step_params["Source"]["ChannelCode"],
                    start=step_params["Source"]["Start"],
                    end=step_params["Source"]["End"],
                    units=step_params.get("Source", {}).get("Units", None),
                )
            except Exception as e:
                logger.exception("Unable to load miniseed data")

        elif step_params.get("Source", {}).get("Format", None) == "csv":
            try:
                if step_params.get("Source", {}).get("Series", None):
                    series = step_params["Source"]["Series"]
                else:
                    series = None
                if step_params.get("Source", {}).get("Units", None):
                    units = step_params["Source"]["Units"]
                else:
                    units = None
                if step_params.get("Source", {}).get("PeriodS", None):
                    period = step_params["Source"]["PeriodS"]
                else:
                    logger.error("Error loading miniseed, Source PeriodS not set.")
                    period = None
                if step_params.get("Source", {}).get("Level", None):
                    level = step_params["Source"]["Level"]
                else:
                    # if not specified, default to 0
                    level = "0"
                ts = self._load_csv(
                    filename=step_params["Source"]["Filename"],
                    series=series,
                    units=units,
                    period=period,
                    level=level,
                )
            except Exception as e:
                logger.exception("Unable to load csv data")
        if step_params.get("ConvertNans", None):
            source = step_params["ConvertNans"]["From"]
            target = step_params["ConvertNans"]["To"]
            logger.info(f"Converting nans from {source} to {target}")
            ts.data = ts.data.replace(source, target)
            ts.check_for_gaps()

        if step_params.get("ConvertUnits", None):
            if step_params.get("ConvertUnits", {}).get("ScaleFactor", None):
                ts.data = ts.data * step_params["ConvertUnits"]["ScaleFactor"]
            if step_params.get("ConvertUnits", {}).get("Offset", None):
                ts.data = ts.data + step_params["ConvertUnits"]["Offset"]
            if step_params.get("ConvertUnits", {}).get("Units", None):
                ts.units = step_params["ConvertUnits"]["Units"]
        self._print_plot_save(ts, step_params)
        return ts

    def _load_miniseed(
        self,
        location: str,
        channel: str,
        start: datetime.datetime,
        end: datetime.datetime,
        units: str = None,
    ):
        st = self._download_mseed(loc=location, cha=channel, start=start, end=end)
        df = self._mseed2pandas(st)
        if len(df):
            period = (df.index[1] - df.index[0]).seconds
            # handle sample rate < 1
            if period == 0:
                period = (df.index[1] - df.index[0]).microseconds / 1e6
            elif period >= 1:
                # change timestamp (index) precision to seconds)
                df.index = df.index.round("1s")
            # print(period)
            return Timeseries(
                data=df, series="raw", units=units, period=period, level="0",
            )
        else:
            return None

    def _download_mseed(
        self, loc: str, cha: str, start: datetime.datetime, end: datetime.datetime
    ):
        """

        :param loc: FDSN Location code, accepts wildcards
        :param cha: FDSN Channel code, accepts wildcards
        :param start:
        :param end:
        :return:
        """
        start_str = datetime.datetime.strftime(start, "%Y-%m-%dT%H:%M:%S.%f")
        end_str = datetime.datetime.strftime(end, "%Y-%m-%dT%H:%M:%S.%f")
        logger.info(
            f"loading {loc} {cha} from {start_str} to {end_str} from IRIS DMC miniseed"
        )
        url = (
            f"https://service.iris.edu/fdsnws/dataselect/1/query?net={self.network}"
            f"&sta={self.fcid}&loc={loc}&cha={cha}&starttime={start_str}&endtime={end_str}"
            f"&format=miniseed&nodata=404"
        )
        # print(url)
        try:
            st = read(url)
        except HTTPError as e:
            logger.error(f"No data found for specified query {url}")
            st = Stream()
        return st

    def _mseed2pandas(self, st: Stream):
        # print(st)
        df = pd.DataFrame()
        dfs = {}
        for i, tr in enumerate(st):
            channel = fdsn2bottlename(tr.stats.channel)
            start = tr.stats["starttime"]
            stop = tr.stats["endtime"]
            step = datetime.timedelta(seconds=tr.stats["delta"])
            logger.info(
                f"Trace {i+1}. {start}:{stop} mapping {tr.stats.channel} to {channel}"
            )
            time_buffer = np.arange(start, stop + step, step, dtype=np.datetime64)
            df2 = pd.DataFrame(index=time_buffer)
            df2[channel] = tr.data
            if channel not in dfs.keys():
                dfs[channel] = df2
            else:
                dfs[channel] = pd.concat([dfs[channel], df2])
        for i, key in enumerate(dfs.keys()):
            df[key] = dfs[key]
        df.index.name = "time"
        return df

    def _load_csv(
        self,
        filename: str,
        sep: str = ",",
        series: str = "",
        units: str = "",
        period: str = None,
        level: str = "",
        use_channels: list = [],
    ):
        if self.local_archive:
            filename = f"{self.local_archive}/{filename}"
        logger.info(f"loading from {filename}")
        df = pd.read_csv(
            filename,
            sep=sep,
            header=0,
            index_col="time",
            parse_dates=True,
            compression="infer",
        )
        if len(df):
            # period = (df.index[1] - df.index[0]).seconds
            # logger.info(f"Using channels {use_channels}")
            return Timeseries(
                data=df, series=series, units=units, period=period, level=level,
            )
        else:
            return None

    def decimate(self, ts: Timeseries, step_params: dict):
        """
        Processing step to decimate data to lower sample rates.
        Only implemented for 1s -> 300s currently.
        :param ts: Timeseries object, containing data to decimate
        :param step_params: Dict, must contain required params PeriodS and Method
        :return:
        """
        # todo: add filter
        target_period = step_params.get("PeriodS", None)
        method = step_params.get("Method", None)
        # if ts.period == 1 and target_period == 300 and method == 'default':
        if (target_period / ts.period) == 300 and method == "default":
            limit = step_params.get("InterpolationLimit", None)
            if limit:
                limit = int(limit)
            method = step_params.get("InterpolationMethod", "linear")
            ts2 = self._decimate_1s_counts_to_300s(ts, method, limit)
            self._print_plot_save(ts2, step_params)
            return ts2
        elif (target_period / ts.period) == 3600 and method == "default":
            limit = step_params.get("InterpolationLimit", None)
            if limit:
                limit = int(limit)
            method = step_params.get("InterpolationMethod", "linear")
            ts2 = self._decimate_1s_counts_to_300s(ts, method, limit)
            ts3 = self._decimate_to_hourly(ts2)
            self._print_plot_save(ts3, step_params)
            return ts3
        elif method == "default":
            logger.error(
                "Decimation method default is only implemented for 1s -> 300s or 1s -> 3600.  Unable to decimate"
            )
            sys.exit(1)
        else:
            logger.error(
                "Only decimation method default is currently implemented.  Unable to decimate"
            )
            sys.exit(1)

    #   def decimate_pressure(self,
    #                         ts: Timeseries,
    #                         step_params: dict):
    #       #todo: test that filter works properly
    #       nans = ts.data.isna()
    #       if step_params.get('InterpolationMethod', None):
    #           interp_method = step_params['InterpolationMethod']
    #       else:
    #           logger.info('using default interpolation method "linear"')
    #           interp_method = 'linear'
    #       #logger.info(f"ts.data:\n{ts.data}")
    #       df = ts.data.interpolate(method=interp_method, limit_direction="both")
    #       #logger.info(f"interpolated:\n{df}")
    #       df2 = df - df.mean()
    #       #logger.info(f"demeaned:\n{df2}")
    #       source_period = ts.period
    #       if step_params.get('Filters', {}).get('Butterworth', None):
    #           try:
    #               filter_type = step_params['Filters']['Butterworth']['Type']
    #           except Exception as e:
    #               logger.error("Missing required butterworth filter parameter 'Type'")
    #           try:
    #               filter_order = step_params['Filters']['Butterworth']['Order']
    #           except Exception as e:
    #               logger.error("Missing required butterworth filter parameter 'Order'")
    #           try:
    #               filter_cutoff_s = step_params['Filters']['Butterworth']['CutoffS']
    #           except Exception as e:
    #               logger.error("Missing required butterworth filter parameter 'CutoffS'")
    #           df3 = self._butterworth_filter(df2, source_period, filter_type, filter_order, filter_cutoff_s)
    #       else:
    #           logger.info("No pressure filtering specified.")
    #           df3 = df2
    #       #print("df3: ", df3)
    #       #print("nans:", nans.loc[nans['atmp'] == True])
    #       #print("ts.quality_df:", ts.quality_df[ts.quality_df['atmp'] == 'm'])
    #       ts2 = Timeseries(data=df3,
    #                          quality_df=ts.quality_df,
    #                          series=ts.series,
    #                          units=ts.units,
    #                          level=ts.level,
    #                          period=ts.period,
    #                          )
    #       target_period = step_params.get('PeriodS', None)
    #       if target_period == 3600:
    #           ts3 = self._decimate_to_hourly(ts2)
    #           self._print_plot_save(ts3, step_params)
    #           #print(ts3.quality_df.value_counts())
    #           return ts3
    #       elif target_period == 300:
    #           ts3 = self._decimate_to_hourly(ts2)
    #           self._print_plot_save(ts3, step_params)
    #           #print(ts3.quality_df.value_counts())
    #           return ts3
    #       else:
    #           logger.error('Unable to decimate pressure data, only PeriodS=3600 implemented.')
    #           sys.exit(1)

    def decimate_pressure(self, ts: Timeseries, step_params: dict):
        # todo: test that filter works properly
        nans = ts.data.isna()
        if step_params.get("InterpolationMethod", None):
            interp_method = step_params["InterpolationMethod"]
        else:
            logger.info('using default interpolation method "linear"')
            interp_method = "linear"
        # logger.info(f"ts.data:\n{ts.data}")
        df = ts.data.interpolate(method=interp_method, limit_direction="both")
        # logger.info(f"interpolated:\n{df}")
        df2 = df - df.mean()
        # logger.info(f"demeaned:\n{df2}")
        source_period = ts.period
        if step_params.get("Filters", {}).get("Butterworth", None):
            try:
                filter_type = step_params["Filters"]["Butterworth"]["Type"]
            except Exception as e:
                logger.error("Missing required butterworth filter parameter 'Type'")
            try:
                filter_order = step_params["Filters"]["Butterworth"]["Order"]
            except Exception as e:
                logger.error("Missing required butterworth filter parameter 'Order'")
            try:
                filter_cutoff_s = step_params["Filters"]["Butterworth"]["CutoffS"]
            except Exception as e:
                logger.error("Missing required butterworth filter parameter 'CutoffS'")
            df3 = self._butterworth_filter(
                df2, source_period, filter_type, filter_order, filter_cutoff_s
            )
        else:
            logger.info("No pressure filtering specified.")
            df3 = df2
        # print("df3: ", df3)
        # print("nans:", nans.loc[nans['atmp'] == True])
        # print("ts.quality_df:", ts.quality_df[ts.quality_df['atmp'] == 'm'])
        ts2 = Timeseries(
            data=df3,
            quality_df=ts.quality_df,
            series=ts.series,
            units=ts.units,
            level=ts.level,
            period=ts.period,
        )
        target_period = step_params.get("PeriodS", None)
        if target_period == 3600:
            ts3 = self._decimate_to_hourly(ts2)
            self._print_plot_save(ts3, step_params)
            # print(ts3.quality_df.value_counts())
            return ts3
        elif target_period == 300:
            ts3 = self._decimate_to_300s(ts2)
            self._print_plot_save(ts3, step_params)
            # print(ts3.quality_df.value_counts())
            return ts3
        else:
            logger.error(
                "Unable to decimate pressure data, only PeriodS=3600 or 300 implemented."
            )
            sys.exit(1)

    def _butterworth_filter(
        self,
        df: pd.DataFrame,
        period: float,
        filter_type: str,
        filter_order: int,
        filter_cutoff_s: float,
    ):
        fc = 1 / filter_cutoff_s
        fs = 1 / period
        [bn, an] = signal.butter(filter_order, fc / (1 / 2 * fs), btype=filter_type)
        df2 = pd.DataFrame(index=df.index)
        for ch in df.columns:
            df2[ch] = signal.filtfilt(bn, an, df[ch])
        return df2

    def _decimate_to_hourly(
        self, ts: Timeseries,
    ):
        """

        :param ts: Timeseries
        :return: Timeseries, decimated to hourly
        """
        logger.info(
            f"Decimating {ts.period}s data to hourly using values where minutes=0."
        )
        return Timeseries(
            data=ts.data[ts.data.index.minute == 0],
            quality_df=ts.quality_df[ts.quality_df.index.minute == 0],
            series=ts.series,
            units=ts.units,
            level=ts.level,
            period=3600,
        )

    def _decimate_to_300s(
        self, ts: Timeseries,
    ):
        """

        :param ts: Timeseries
        :return: Timeseries, decimated to hourly
        """
        logger.info(
            f"Decimating {ts.period}s data to 300s using resample at 5 minute periods."
        )
        data = ts.data.resample("5T", label="left").nearest()
        qdf = ts.quality_df.resample("5T", label="left").nearest()
        return Timeseries(
            data=data,
            quality_df=qdf,
            series=ts.series,
            units=ts.units,
            level=ts.level,
            period=300,
        )

    def _decimate_1s_counts_to_300s(self, ts: Timeseries, method: str, limit: int):

        # df2 = ts.data.replace(999999, np.nan).interpolate(method='linear', limit_direction='both')

        logger.info(
            f"Interpolating 1s to 300s data using method={method} and limit={limit}"
        )
        df2 = ts.remove_999999s(
            interpolate=True, method=method, limit_direction="both", limit=limit
        )

        # zero the data to the first value prior to filtering
        initial_values = df2.iloc[0]
        df3 = df2 - initial_values

        # perform a 5 stage filter-and-decimate
        # 1s -> 2s -> 4s -> 12s -> 60s -> 300s
        # FIR filter coefficients from Agnew and Hodgkinson 2007

        wtsdby2d = [
            0.0983262,
            0.2977611,
            0.4086973,
            0.3138961,
            0.0494246,
            -0.1507778,
            -0.1123764,
            0.0376576,
            0.0996838,
            0.0154992,
            -0.0666489,
            -0.0346632,
            0.0322767,
            0.0399294,
            -0.0097461,
            -0.0341585,
            -0.0039241,
            0.0246776,
            0.0099725,
            -0.0157879,
            -0.0099098,
            0.0078510,
            0.0081126,
            -0.0026986,
            -0.0061424,
            0.0007108,
            0.0039659,
            -0.0006209,
            -0.0017117,
            0.0007240,
        ]

        wtsdby3c = [
            0.0373766,
            0.1165151,
            0.2385729,
            0.3083302,
            0.2887327,
            0.1597948,
            0.0058244,
            -0.0973639,
            -0.1051034,
            -0.0358455,
            0.0359044,
            0.0632477,
            0.0302351,
            -0.0168856,
            -0.0356758,
            -0.0190635,
            0.0126188,
            0.0159705,
            0.0082144,
            -0.0087978,
            -0.0037289,
            -0.0017068,
            0.0028335,
        ]

        wtsdby5b = [
            0.0218528,
            0.0458359,
            0.0908603,
            0.1359777,
            0.1830881,
            0.1993418,
            0.1957624,
            0.1561194,
            0.0994146,
            0.0346412,
            -0.0236544,
            -0.0580081,
            -0.0703257,
            -0.0555546,
            -0.0287709,
            0.0032613,
            0.0267938,
            0.0358952,
            0.0311186,
            0.0134283,
            -0.0028524,
            -0.0170042,
            -0.0176765,
            -0.0123123,
            -0.0036798,
            0.0057730,
            0.0059817,
            0.0083501,
            0.0000581,
            0.0005724,
            -0.0033127,
            0.0004411,
            -0.0030766,
            0.0016604,
        ]

        channels = ts.columns  # ['CH0', 'CH1', 'CH2', 'CH3']

        stage1 = pd.DataFrame(index=df3.index)
        for ch in channels:
            data = df3[ch].values
            stage1[ch] = lfilter(wtsdby2d, 1.0, data)
        stage1d = stage1.iloc[::2]

        stage2 = pd.DataFrame(index=stage1d.index)
        for ch in channels:
            data = stage1d[ch].values
            stage2[ch] = lfilter(wtsdby2d, 1.0, data)
        stage2d = stage2.iloc[::2]

        stage3 = pd.DataFrame(index=stage2d.index)
        for ch in channels:
            data = stage2d[ch].values
            stage3[ch] = lfilter(wtsdby3c, 1.0, data)
        stage3d = stage3.iloc[::3]

        stage4 = pd.DataFrame(index=stage3d.index)
        for ch in channels:
            data = stage3d[ch].values
            stage4[ch] = lfilter(wtsdby5b, 1.0, data)
        stage4d = stage4.iloc[::5]

        stage5 = pd.DataFrame(index=stage4d.index)
        for ch in channels:
            data = stage4d[ch].values
            stage5[ch] = lfilter(wtsdby5b, 1.0, data)
        stage5d = stage5.iloc[::5]

        # add back in the initial values
        decimated_data = (stage5d + initial_values).astype(int)

        return Timeseries(
            data=decimated_data,
            series=ts.series,
            units=ts.units,
            level=ts.level,
            period=300,
        )

    def linearize(self, ts: Timeseries, step_params: dict):
        """
        Processing step to convert digital counts to microstrain based on geometry of GTSM gauges
        :param ts: Timeseries object, containing data to convert to microstrain
        :param step_params: Dict, optional boolean params Plot, Print, Save
        :return: Timeseries object, in units of microstrain
        """
        # build a series of reference strains.  if using metadata from XML the /1e8 is already included.
        reference_strains = pd.Series(dtype="float64")
        for ch in ts.columns:
            reference_strains[ch] = self.metadata.linearization[ch]
        # remove any 999999 values in data, ok to leave as Nan rather than interpolate.
        if ts.nines:
            logger.info(f"Found {ts.nines} 999999s, replacing with nans")
            df = ts.remove_999999s(interpolate=False)
        else:
            df = ts.data
        linearized_data = (
            (
                ((df / 100000000) / (1 - (df / 100000000)))
                - (
                    (reference_strains / 100000000)
                    / (1 - (reference_strains / 100000000))
                )
            )
            * (self.metadata.gap / self.metadata.diameter)
            * 1000000
        )

        ts2 = Timeseries(
            data=linearized_data,
            series="strain",
            units="microstrain",
            level="1",
            period=ts.period,
        )
        self._print_plot_save(ts2, step_params)
        return ts2

    def apply_calibration_matrix(self, ts: Timeseries, step_params: dict):
        """
        Processing step to convert gauge strains into areal and shear strains
        :param ts: Timeseries object containing gauge data in microstrain
        :param step_params: dict, required param CalibrationMatrix, optional boolean params Plot, Print, Save
        :return: Timeseries object, in units of microstrain
        """
        # calculate areal and shear strains from gauge strains
        if step_params.get("CalibrationMatrix", None):
            if step_params["CalibrationMatrix"] == "lab":
                logger.info("Using lab calibrated strain matrix")
                strain_matrix = self.metadata.strain_matrices["lab_strain_matrix"]
            elif step_params["CalibrationMatrix"] == "er2010":
                strain_matrix = self.metadata.strain_matrices["er2010_strain_matrix"]
                logger.info("Using er2010 strain matrix")
            elif step_params["CalibrationMatrix"] == "ch_prelim":
                strain_matrix = self.metadata.strain_matrices["ch_prelim_strain_matrix"]
                logger.info("Using ch_prelim strain matrix")
            else:
                # if step_params['CalibrationMatrix'] != 'lab':
                logger.info(
                    f"Unrecognized CalibrationMatrix {step_params['CalibrationMatrix']}"
                )
                strain_matrix = self.metadata.strain_matrices["lab_strain_matrix"]
                logger.info("Using default lab strain matrix")

            # logger.info(strain_matrix)
            # todo: implement UseChannels to arbitrary matrices
            # self.use_ch

            regional_strain_df = np.matmul(
                strain_matrix, ts.data[ts.columns].transpose()
            ).transpose()
            regional_strain_df = regional_strain_df.rename(
                columns={0: "Eee+Enn", 1: "Eee-Enn", 2: "2Ene"}
            )
            ts2 = Timeseries(
                data=regional_strain_df,
                series="strain",
                units="microstrain",
                level="2a",
                period=ts.period,
            )

            self._print_plot_save(ts2, step_params)
            return ts2

    def upsample(self, ts: Timeseries, step_params: dict):
        # g_flags = ts.data.index
        # channels = ts.data.columns
        try:
            freq = f"{step_params['PeriodS']}S"
            new_index = pd.date_range(
                step_params["Source"]["Start"], step_params["Source"]["End"], freq=freq
            )
            limit = step_params.get("InterpolationLimit", None)
            if limit:
                limit = int(limit)
            method = step_params.get("InterpolationMethod", "linear")
            # ts2 = self.decimate_1s_counts_to_300s(ts, method, limit)
            data_df = ts.data.reindex(new_index).interpolate(
                method=method, limit_direction="both", limit=limit
            )
            quality_df = ts.quality_df.reindex(data_df.index)
            quality_df = quality_df.fillna("i")
            level = "1"
            ts2 = Timeseries(
                data=data_df,
                quality_df=quality_df,
                series="interp",
                units=ts.units,
                level=level,
                period=step_params["PeriodS"],
            )
            self._print_plot_save(ts2, step_params)
            return ts2
        except Exception as e:
            logger.error(e)

    def generate_pressure_correction(self, ts: Timeseries, step_params: dict):
        response = self.metadata.atmp_response
        logger.info(response)
        if step_params.get("Corrections", None):
            tmp_params = {
                "Corrections": step_params["Corrections"],
                "Print": False,
                "Plot": False,
                "Save": False,
            }
            ts2 = self.apply_corrections(ts, tmp_params)
        else:
            ts2 = ts

        data_df = pd.DataFrame(index=ts2.data.index)
        quality_df = pd.DataFrame(index=ts2.data.index)
        for key in response:
            data_df[key] = ts2.data * float(response[key])
            quality_df[key] = ts2.quality_df
        # print(data_df)
        # print(quality_df)
        ts3 = Timeseries(
            data=data_df,
            quality_df=quality_df,
            series="atmp_c",
            units="microstrain",
            period=ts.period,
            level="2a",
        )
        self._print_plot_save(ts2, step_params)
        return ts3

    def generate_tide_correction(self, ts: Timeseries, step_params: dict):
        # load tide info from xml
        tidal_params = self.metadata.tidal_params
        # load the longitude
        longitude = self.metadata.longitude

        hartid = "docker run -i --rm ghcr.io/earthscope/spotl hartid"
        start = ts.data.index[0]
        datestring = (
            str(start.year).zfill(4)
            + " "
            + str(start.month).zfill(2)
            + " "
            + str(start.day).zfill(2)
            + " "
            + str(start.hour).zfill(2)
            + " "
            + str(start.minute).zfill(2)
            + " "
            + str(start.second).zfill(2)
        )

        nterms = str(len(ts.data))
        samp = int(ts.period)
        cmd = f"{hartid} {datestring} {nterms} {samp}"
        # cmd = hartid + " " + datestring + " " + nterms + " " + samp

        channels = ts.columns
        tides = set()
        for key in tidal_params:
            tides.add(key[1])
        cmds = {}
        # for i, ch in enumerate(gauges):
        for ch in channels:
            inputfile = f"printf 'l\n{longitude}\n"
            for tide in tides:
                inputfile += f" {tidal_params[(ch, tide, 'doodson')]} {tidal_params[(ch, tide, 'amp')].ljust(7, '0')} {tidal_params[(ch, tide, 'phz')].ljust(8, '0')}\n"
            inputfile += "-1'"
            cmds[ch] = inputfile + " | " + cmd

        df = pd.DataFrame(index=ts.data.index)
        for ch in channels:
            output = subprocess.check_output(cmds[ch], shell=True).decode("utf-8")
            df[ch] = np.fromstring(output, dtype=float, sep="\n")
        df = df * 1e-3
        ts2 = Timeseries(
            data=df, series="tide_c", units="microstrain", period=ts.period, level="2a",
        )
        self._print_plot_save(ts2, step_params)
        return ts2

    def generate_trend_correction(self, ts: Timeseries, step_params: dict):

        if step_params.get("Type", None):
            if step_params["Type"] == "linear":
                df_trend_c = self._linear_trend_correction(ts, step_params)
            elif step_params["Type"] == "median":
                df_trend_c = self._median_trend_correction(ts, step_params)
            else:
                logger.error(
                    f"Detrend type {step_params['Type']} not supported.  try 'linear' or 'median'"
                )
                sys.exit(1)
            ts2 = Timeseries(
                data=df_trend_c,
                series="trend_c",
                units="microstrain",
                period=ts.period,
                level="2a",
            )
            self._print_plot_save(ts2, step_params)
            return ts2

        else:
            logger.error("Missing required parameter 'Type'")
            sys.exit(1)

    # def _linear_trend_correction(self,
    #                              ts: Timeseries,
    #                              step_params: dict):
    #     df_trend_c = pd.DataFrame(data=ts.data.index)
    #     for ch in ts.columns:
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(df_trend_c.index, ts.data[ch])
    #         df_trend_c[ch] = df_trend_c.index * slope
    #     return df_trend_c[ts.columns].set_index(df_trend_c['time'])
    #
    def _linear_trend_correction(self, ts: Timeseries, step_params: dict):
        if step_params.get("Source", {}).get("DateRange", None):
            trend_start = step_params["Source"]["DateRange"]["Start"]
            trend_end = step_params["Source"]["DateRange"]["End"]
            logger.info(
                "Calculating linear trend based on {trend_start} to {trend_end}"
            )
        else:
            trend_start = self.parameters["DateRange"]["Start"]
            trend_end = self.parameters["DateRange"]["End"]

        logger.info(f"Trend Start: {trend_start}")
        logger.info(f"Trend Start: {trend_end}")
        df_trend_c = pd.DataFrame(data=ts.data.index)
        windowed_df = ts.data.copy()[trend_start:trend_end].reset_index()
        for ch in ts.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                windowed_df.index, windowed_df[ch]
            )
            df_trend_c[ch] = df_trend_c.index * slope
        print("df_trend_c", df_trend_c)
        return df_trend_c[ts.columns].set_index(df_trend_c["time"])

        # df_trend_c = pd.DataFrame(data=ts.data.index)
        # for ch in ts.columns:
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(df_trend_c.index, ts.data[ch])
        #     df_trend_c[ch] = df_trend_c.index * slope
        # return df_trend_c[ts.columns].set_index(df_trend_c['time'])
        #

        # temp_df = ts.data.copy()[trend_start: trend_end]
        # df_windowed = pd.DataFrame(data=temp_df.index)
        # df_trend_c = pd.DataFrame(index=ts.data.index)
        # for ch in ts.columns:
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(
        #                                     df_windowed.index,
        #                                     temp_df[ch])
        #     # slope, intercept, r_value, p_value, std_err = stats.linregress(
        #     #     df_trend_c.index,
        #     #     ts.data[ch])
        #     df_trend_c[ch] = df_trend_c.index * slope
        # return df_trend_c#[ts.columns]#.set_index(df_trend_c['time'])

    def _median_trend_correction(self, ts: Timeseries, step_params: dict):
        if step_params.get("Source", {}).get("DateRange", None):
            trend_start = step_params["Source"]["DateRange"]["Start"]
            trend_end = step_params["Source"]["DateRange"]["End"]
            logger.info(
                f"Calculating median trend based on {trend_start} to {trend_end}"
            )
        else:
            trend_start = ts.data.index[0]
            trend_end = ts.data.index[-1]
        tdf = ts.data.copy()[trend_start:trend_end]
        # print(tdf)
        # convert index to seconds
        timestamps = tdf.index
        full_index_seconds = ts.data.index.astype(np.int64) // 10 ** 9
        tdf.index = tdf.index.astype(np.int64) // 10 ** 9
        tdiff = (29 * 24 * 60 * 60 + 12 * 60 * 60 + 44 * 60) / 30  # seconds
        # Medians
        df1 = tdf[tdf.index <= tdf.index[-1] - tdiff]
        df2 = tdf[tdf.index >= tdf.index[0] + tdiff]

        # print(f"df1:\n{pd.to_datetime(df1.index, unit='s')[:24]}")
        # print(f"df2:\n{pd.to_datetime(df2.index, unit='s')[-24:]}")

        trend_c = pd.DataFrame(index=ts.data.index)

        for gauge in tdf.columns:
            med = (df1[gauge].values - df2[gauge].values) / (df1.index - df2.index)
            medmad_std_dev1 = (
                stats.median_abs_deviation(med) * 1.4826
            )  # See Blewitt et al. 2016 and Wilcox 2005
            tmp = med[med < (np.median(med) + medmad_std_dev1 * 2)]
            med_2sig = tmp[tmp > (np.median(med) - medmad_std_dev1 * 2)]
            medmad_std_dev = stats.median_abs_deviation(med_2sig) * 1.4826
            m = np.median(med_2sig)
            std_err = 1.2533 * medmad_std_dev / np.sqrt(len(med_2sig) / 4)
            text = f"{gauge} Median and Standard Error (following Blewitt et al. 2016)): {m}, {std_err}"
            logger.info(text)
            # todo: write text to results file
            trend_c[gauge] = (full_index_seconds - full_index_seconds[0]) * m
        return trend_c

    def calculate_offsets(self, ts: Timeseries, step_params: dict):
        # todo: LimitMultiplier from params
        if step_params.get("LimitMultiplier", None):
            lim_mult = int(step_params["LimitMultiplier"])
        else:
            logger.info("Using default limit multiplier of 10.")
            lim_mult = 10
        if step_params.get("Corrections", None):
            tmp_params = {
                "Corrections": step_params["Corrections"],
                "Print": False,
                "Plot": False,
                "Save": False,
            }
            ts2 = self.apply_corrections(ts, tmp_params)
        else:
            ts2 = ts

        first_diffs = ts2.data.diff()
        # Use 75% of 1st differences to estimate an offset cutoff
        drop = round(len(first_diffs) * 0.25)
        offset_limit = []
        df_offsets = pd.DataFrame(index=first_diffs.index)
        for ch in ts2.columns:
            # Offset limit is 10x the average absolute value of first differences
            # within 2 st_dev of the first differences
            offset_limit.append(
                np.mean(abs(first_diffs[ch].sort_values().iloc[0:-drop])) * lim_mult
            )

            # CH edit. Calculate offsets from the detrended series
            # Justification: if the offset is calculated from the original series,
            # there may be an overcorrection that is noticeable, especially with large trends
            df_offsets[ch] = first_diffs[first_diffs[ch].abs() > offset_limit[-1]][ch]

        # make a dataframe of running total of offsets
        df_cumsum = df_offsets.fillna(0).cumsum()
        logger.info(f"Using offset limit of {offset_limit}")
        # todo: write to results file
        ts3 = Timeseries(
            data=df_cumsum,
            series="offset_c",
            units=ts.units,
            period=ts.period,
            level="2a",
        )
        self._print_plot_save(ts3, step_params)
        return ts3

    def calculate_observed_tides_pressure(self, strain_ts, pressure_ts, step_params):
        """
        Runs hourly strain and pressure data through baytap08.  Converts to nanostrain prior to input.
        :param strain_ts: Uncorrected hourly gauge data in microstrain
        :param pressure_ts: Hourly pressure data in hPa
        :param step_params:
        :return: nested dictionary containing baytap results for each channel
        """
        start = strain_ts.data.index[0]
        end = strain_ts.data.index[-1]
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        start_str_short = start.strftime("%Y-%m-%d")
        end_str_short = end.strftime("%Y-%m-%d")
        if step_params.get("Source", {}).get("Strain", {}).get("Corrections", None):
            logger.info(f"Applying corrections to strain data")
            tmp_params = {
                "Corrections": step_params["Source"]["Strain"]["Corrections"],
                "Print": False,
                "Plot": False,
                "Save": False,
            }
            ts2 = self.apply_corrections(strain_ts, tmp_params)
            for i, correction in enumerate(
                step_params["Source"]["Strain"]["Corrections"]
            ):
                if i == 0:
                    correction_list = correction
                else:
                    correction_list += f"_{correction}"

        else:
            ts2 = strain_ts
            correction_list = "uncorrected"

        if step_params.get("BaytapParameters", {}).get("Dmin", None):
            dmin = step_params["BaytapParameters"]["Dmin"]
        else:
            dmin = 0.001
            logger.info("Using default dmin = 0.001")

        if step_params.get("BaytapParameters", {}).get("Igrp", None):
            igrp = step_params["BaytapParameters"]["Igrp"]
        else:
            igrp = 5
            logger.info("Using default igrp = 5")

        strain_df = ts2.data.copy()
        pressure_df = pressure_ts.data.copy()

        # demean
        logger.info(f"demeaning strain and pressure data")
        for ch in strain_df.columns:
            strain_df[ch] = strain_df[ch] - strain_df[ch].mean()
        pressure_df = pressure_df - pressure_df.mean()
        logger.info(f"converting to nanostrain")
        strain_df = strain_df * 1e3

        # replace any nan values with 999999
        # todo: use original 999999 indexes?
        if step_params.get("ConvertNans", {}).get("To", None):
            if step_params["ConvertNans"]["To"] == 999999:
                logger.info(f"Converting NAN values to 999999 for baytap08 input")
                strain_df = strain_df.fillna(value=999999)
                pressure_df = pressure_df.fillna(value=999999)
            else:
                logger.error("ConvertNans for baytap08 input requires To: 999999")

        # reindex with seconds instead of timestamp
        strain_df.index = strain_df.index.astype(np.int64) / 1e9
        pressure_df.index = pressure_df.index.astype(np.int64) / 1e9
        # print(strain_df)

        # Sometimes the time indices don't work out to be same length
        # so we must reset the pressure index and fill with 999999
        pressure_df1 = pd.DataFrame(index=strain_df.index)
        pressure_df = pressure_df1.merge(
            pressure_df, how="left", left_index=True, right_index=True
        ).fillna(999999)
        # print(pressure_df)

        ### Data input file prep, strain and pressure
        # Make a new folder for baytap files if it doesn't exist
        baytap_dir = f"{self.workdir}/baytap08"
        os.makedirs(baytap_dir, exist_ok=True)
        # baytap_dir = self.local_archive + '/baytap08'
        # os.makedirs(baytap_dir, exist_ok=True)

        # Strain files
        strain_df["timestamp"] = strain_df.index
        for ch in strain_df.columns:  # single gauges
            fname = f"{self.network}_{self.fcid}.{ch}.{start_str_short}.{end_str_short}.baytapf.txt"
            strain_df[ch].to_csv(f"{baytap_dir}/{fname}", index=False, header=[f"{ch}"])

        # Pressure file
        pname = f"{self.network}_{self.fcid}.{start_str_short}.{end_str_short}.atmp.baytapf.txt"
        pressure_df.to_csv(f"{baytap_dir}/{pname}", index=False, header=["atmp"])

        ##########
        ## Control file parameters not already set
        yr = str(start.year)
        mo = str(start.month).zfill(2)
        day = str(start.day).zfill(2)
        hr = str(start.hour).zfill(2)
        ndata = span = shift = int(len(strain_df))
        # Sample period, in hours
        samp = ((end.timestamp() - start.timestamp()) / 60 / 60) / (ndata - 1)
        # print(yr, mo, day, hr, ndata, samp)

        # Control.txt write
        with open(baytap_dir + "/control.txt", "w") as file:
            file.write("&param \nkind=7, \n")
            file.write(f"span={span} , shift={shift} , \n")
            file.write(f"dmin={dmin}, \n")
            file.write("lpout=0, filout=1, \n")
            file.write(f"iaug=1, lagp=0,\n")
            file.write("maxitr=50,\n")
            file.write(f"igrp={igrp}, \n")
            file.write(f"ndata={ndata},\n")
            file.write("inform=3, \n")
            file.write(f"year={yr},mon={mo},day={day},hr={hr},delta={samp},\n")
            file.write(
                f"lat={self.metadata.latitude},long={self.metadata.longitude},ht={self.metadata.elevation},grav=0.0\n"
            )
            file.write("rlim=999990.D0,spectw=3,\n")
            file.write("&end \n")
            file.write(f"{self.fcid} Strain {str(start_str)} to {str(end_str)}\n")
            file.write("----\n")
            file.write(f"{self.fcid: <40} STATION NAME\n")
            file.write(f'{"PBO GTSM21": <40} INSTRUMENT NAME\n')
            file.write("----\n")
            file.write(f'{"Strain (counts or nstrain)": <40} UNIT OF TIDAL DATA\n')
            file.write(
                f'{"Barometric Pressure (mbar)": <40} TITLE OF ASSOSIATED DATASET\n'
            )
        file.close()

        ## Runs baytap for all gauges
        # abs_path = os.getcwd() + baytap_dir[1:]
        abs_path = f"{os.getcwd()}/{baytap_dir}"
        # logger.info(abs_path)
        logger.info(f"{self.network}_{self.fcid} files prepped")
        for ch in strain_ts.columns:
            logger.info(f"Running baytap08 for {ch}")
            fname = f"{self.network}_{self.fcid}.{ch}.{start_str_short}.{end_str_short}.baytapf.txt"
            subprocess.check_output(
                f"cat {baytap_dir}/{fname} | docker run --rm -i --mount type=bind,src='{abs_path}',"
                f"target='/opt/baytapf' ghcr.io/earthscope/baytap08 baytap08 baytapf/control.txt "
                f"baytapf/{self.network}_{self.fcid}.{ch}.{correction_list}.{start_str_short}.{end_str_short}.results "
                f"baytapf/{pname} > {baytap_dir}/{self.network}_{self.fcid}.{ch}.{start_str_short}.{end_str_short}.tmp",
                shell=True,
            )
            # replaces D with E for scientific notation, python friendly
            subprocess.check_output(
                f"sed 's/D/E/g' "
                f"{baytap_dir}/{self.network}_{self.fcid}.{ch}.{start_str_short}.{end_str_short}.tmp > "
                f"{baytap_dir}/{self.network}_{self.fcid}.{ch}.{correction_list}.{start_str_short}.{end_str_short}.decomp",
                shell=True,
            )
            logger.info(
                f"Writing to "
                f"{baytap_dir}/{self.network}_{self.fcid}.{ch}.{correction_list}.{start_str_short}.{end_str_short}.results"
                f"{baytap_dir}/{self.network}_{self.fcid}.{ch}.{correction_list}.{start_str_short}.{end_str_short}.decomp"
            )
        logger.info("Baytap08 analysis complete")

        logger.info("Parsing baytap08 results files")

        res = {}
        res["respc"] = []
        res["respc_err"] = []
        res["m2amp"] = []
        res["m2amp_err"] = []
        res["m2phs"] = []
        res["m2phs_err"] = []
        res["o1amp"] = []
        res["o1amp_err"] = []
        res["o1phs"] = []
        res["o1phs_err"] = []

        for ch in strain_ts.data.columns:
            file = f"{baytap_dir}/{self.network}_{self.fcid}.{ch}.{correction_list}.{start_str_short}.{end_str_short}.results"
            with open(file, "r") as f:
                for line in f:
                    if "respc" in line:
                        respc = float(line.split()[-1].replace("D", "e"))
                        res["respc"].append(respc)
                        line = next(f)
                        respc_err = float(line.split()[-1].replace("D", "e"))
                        res["respc_err"].append(respc_err)
                    elif ": M2" in line:
                        res["m2amp"].append(float(line.split()[8]))
                        res["m2phs"].append(float(line.split()[6]))
                        res["m2amp_err"].append(float(line.split()[9]))
                        res["m2phs_err"].append(float(line.split()[7]))
                    elif ": O1" in line:
                        res["o1amp"].append(float(line.split()[8]))
                        res["o1phs"].append(float(line.split()[6]))
                        res["o1amp_err"].append(float(line.split()[9]))
                        res["o1phs_err"].append(float(line.split()[7]))
        logger.info(f"Results:\n {res}")
        results_file = f"{baytap_dir}/{self.network}_{self.fcid}_{correction_list}.{start_str_short}.{end_str_short}_results.json"
        with open(results_file, "w") as res_file:
            res_file.write(json.dumps(res))

        self.obs_tides = np.array(
            [
                [
                    res["m2amp"][0] * np.cos(res["m2phs"][0] * np.pi / 180),
                    res["m2amp"][0] * np.sin(res["m2phs"][0] * np.pi / 180),
                    res["o1amp"][0] * np.cos(res["o1phs"][0] * np.pi / 180),
                    res["o1amp"][0] * np.sin(res["o1phs"][0] * np.pi / 180),
                ],
                [
                    res["m2amp"][1] * np.cos(res["m2phs"][1] * np.pi / 180),
                    res["m2amp"][1] * np.sin(res["m2phs"][1] * np.pi / 180),
                    res["o1amp"][1] * np.cos(res["o1phs"][1] * np.pi / 180),
                    res["o1amp"][1] * np.sin(res["o1phs"][1] * np.pi / 180),
                ],
                [
                    res["m2amp"][2] * np.cos(res["m2phs"][2] * np.pi / 180),
                    res["m2amp"][2] * np.sin(res["m2phs"][2] * np.pi / 180),
                    res["o1amp"][2] * np.cos(res["o1phs"][2] * np.pi / 180),
                    res["o1amp"][2] * np.sin(res["o1phs"][2] * np.pi / 180),
                ],
                [
                    res["m2amp"][3] * np.cos(res["m2phs"][3] * np.pi / 180),
                    res["m2amp"][3] * np.sin(res["m2phs"][3] * np.pi / 180),
                    res["o1amp"][3] * np.cos(res["o1phs"][3] * np.pi / 180),
                    res["o1amp"][3] * np.sin(res["o1phs"][3] * np.pi / 180),
                ],
            ]
        )

        logger.info(f"Observed Tides:\n {self.obs_tides}")
        obs_tides_file = f"{baytap_dir}/{self.network}_{self.fcid}_{correction_list}.{start_str_short}.{end_str_short}_obs_tides.txt"
        header = "m2-r m2-i o1-r o1-i"
        np.savetxt(obs_tides_file, self.obs_tides, header=header)
        self.obs_tides_name = step_params.get("Name", "baytap_results")
        # #convert index back to timestamp
        # strain_df.index = pd.to_datetime(strain_df.index, unit='s')
        # print(strain_df)
        #
        # ts2 = Timeseries(data=strain_df,
        #                  series='baytap08_input',
        #                  units='microstrain',
        #                  period=strain_ts.period,
        #                  level='2a',
        #                  )
        # title="Baytap08 Input"
        # zero=False
        # plot_type='scatter'
        # save_as=f'{self.plotdir}/{self.network}_{self.fcid}_baytap08_input.png'
        # ts2.plot(title=title, zero=zero, type=plot_type, save_as=save_as)

    def calculate_modeled_tides(self, step_params: dict):

        spotl = "ghcr.io/earthscope/spotl"
        fcid = self.metadata.fcid
        lat = self.metadata.latitude
        lon = self.metadata.longitude
        ht = self.metadata.elevation
        if step_params.get("Inputs", {}).get("GlobalOceanModel", None):
            glob_oc = step_params["Inputs"]["GlobalOceanModel"]
        else:
            glob_oc = "osu.tpxo72.2010"

        if step_params.get("Inputs", {}).get("RegionalOceanModel", None):
            reg_oc = step_params["Inputs"]["RegionalOceanModel"]
        else:
            reg_oc = "gefu"

        if step_params.get("Inputs", {}).get("GreensFunction", None):
            greenf = step_params["Inputs"]["GreensFunction"]
        else:
            greenf = "contap"

        # Set directories
        spotdir = f"{os.getcwd()}/{self.workdir}/spotl"
        os.makedirs(spotdir, exist_ok=True)
        # spotdir = os.getcwd() + '/data/spotl'
        # if os.path.exists(spotdir):
        #    subprocess.call(['rm', '-rf', spotdir])

        # Run polymake for regional models of interest
        # Kathleen already had one made - using it for this purpose.
        # COMMENT NEXT 2 LINES FOR POLYS ALREADY IN DOCKER CONTAINER working DIRECTORY
        # AND CHANGE DELETE PATH PRECEDING poly.{reg_oc} IN ALL LINES BELOW
        command = (
            f"docker run --rm -i -w /opt/spotl/working/ --mount type=bind,src='{spotdir}',"
            f"target='/opt/spotl/work' {spotl} polymake << EOF > {spotdir}/poly.{reg_oc} \n- {reg_oc} \nEOF"
        )
        subprocess.check_output(command, shell=True)
        # M2 ocean load for the ocean model but exclude the area in specified polygon
        command = (
            f"docker run --rm -i -w /opt/spotl/working/ --mount type=bind,src='{spotdir}',"
            f"target='/opt/spotl/work' {spotl} nloadf {fcid} {lat} {lon} {ht} "
            f"m2.{glob_oc} green.{greenf}.std l ../work/poly.{reg_oc} - > {spotdir}/ex1m2.f1"
        )
        subprocess.check_output(command, shell=True)
        # M2 ocean load for the regional ocean model in the area in specified polygon
        command = (
            f"docker run --rm -i -w /opt/spotl/working/ --mount type=bind,src='{spotdir}',"
            f"target='/opt/spotl/work' {spotl} nloadf {fcid} {lat} {lon} {ht} "
            f"m2.{reg_oc} green.{greenf}.std l ../work/poly.{reg_oc} + > {spotdir}/ex1m2.f2"
        )
        subprocess.check_output(command, shell=True)
        #  Add the M2 loads computed above together
        command = (
            f"cat {spotdir}/ex1m2.f1 {spotdir}/ex1m2.f2 | docker run --rm -i -w /opt/spotl/working/ "
            f"--mount type=bind,src='{spotdir}',target='/opt/spotl/work' {spotl} loadcomb c >  {spotdir}/tide.m2"
        )
        subprocess.check_output(command, shell=True)
        # O1 ocean load for the ocean model but exclude the area in specified polygon
        command = (
            f"docker run --rm -i -w /opt/spotl/working/ --mount type=bind,src='{spotdir}',"
            f"target='/opt/spotl/work' {spotl} nloadf {fcid} {lat} {lon} {ht} "
            f"o1.{glob_oc} green.{greenf}.std l ../work/poly.{reg_oc} - > {spotdir}/ex1o1.f1"
        )
        subprocess.check_output(command, shell=True)
        # O1 ocean load for the regional ocean model in the area in specified polygon
        command = (
            f"docker run --rm -i -w /opt/spotl/working/ --mount type=bind,src='{spotdir}',"
            f"target='/opt/spotl/work' {spotl} nloadf {fcid} {lat} {lon} {ht} "
            f"o1.{reg_oc} green.{greenf}.std l ../work/poly.{reg_oc} + > {spotdir}/ex1o1.f2"
        )
        subprocess.check_output(command, shell=True)
        # Add the O1 loads computed above together
        command = (
            f"cat {spotdir}/ex1o1.f1 {spotdir}/ex1o1.f2 | docker run --rm -i -w /opt/spotl/working/ "
            f"--mount type=bind,src='{spotdir}',target='/opt/spotl/work' "
            f"{spotl} loadcomb c >  {spotdir}/tide.o1"
        )
        subprocess.check_output(command, shell=True)
        # Compute solid earth wides and combine with above ocean loads
        command = (
            f"cat  {spotdir}/tide.m2 | docker run --rm -i -w /opt/spotl/working/ "
            f"--mount type=bind,src='{spotdir}',target='/opt/spotl/work' "
            f"{spotl} loadcomb t >  {spotdir}/m2.tide.total"
        )
        subprocess.check_output(command, shell=True)
        command = (
            f"cat {spotdir}/tide.o1 | docker run --rm -i -w /opt/spotl/working/ "
            f"--mount type=bind,src='{spotdir}',target='/opt/spotl/work' "
            f"{spotl} loadcomb t >  {spotdir}/o1.tide.total"
        )
        subprocess.check_output(command, shell=True)
        # Find the amps and phases, compute complex numbers:
        awk_print_str = "{print $2,$3,$4,$5,$6,$7}"
        command = f"grep '^s' {spotdir}/m2.tide.total | awk '{awk_print_str}'"
        # print(command)
        l = subprocess.check_output(command, shell=True)
        Eamp, Ephase, Namp, Nphase, ENamp, ENphase = [float(i) for i in l.split()]
        m2E, m2N, m2EN = (
            complex(
                Eamp * np.cos(Ephase * np.pi / 180), Eamp * np.sin(Ephase * np.pi / 180)
            ),
            complex(
                Namp * np.cos(Nphase * np.pi / 180), Namp * np.sin(Nphase * np.pi / 180)
            ),
            complex(
                ENamp * np.cos(ENphase * np.pi / 180),
                ENamp * np.sin(ENphase * np.pi / 180),
            ),
        )
        command = f"grep '^s' {spotdir}/o1.tide.total | awk '{awk_print_str}'"
        # print(command)
        l = subprocess.check_output(command, shell=True)
        Eamp, Ephase, Namp, Nphase, ENamp, ENphase = [float(i) for i in l.split()]
        o1E, o1N, o1EN = (
            complex(
                Eamp * np.cos(Ephase * np.pi / 180), Eamp * np.sin(Ephase * np.pi / 180)
            ),
            complex(
                Namp * np.cos(Nphase * np.pi / 180), Namp * np.sin(Nphase * np.pi / 180)
            ),
            complex(
                ENamp * np.cos(ENphase * np.pi / 180),
                ENamp * np.sin(ENphase * np.pi / 180),
            ),
        )
        # Combine into areal and shear (differential and engineering) real and imaginary parts
        arealm2, diffm2, engm2 = m2E + m2N, m2E - m2N, 2 * m2EN
        arealo1, diffo1, engo1 = o1E + o1N, o1E - o1N, 2 * o1EN
        # print(arealm2, diffm2, engm2, arealo1, diffo1, engo1)
        self.pred_tides = np.array(
            [
                [
                    np.real(arealm2),
                    np.imag(arealm2),
                    np.real(arealo1),
                    np.imag(arealo1),
                ],
                [np.real(diffm2), np.imag(diffm2), np.real(diffo1), np.imag(diffo1)],
                [np.real(engm2), np.imag(engm2), np.real(engo1), np.imag(engo1)],
            ]
        )
        logger.info(
            f"M2 areal amp, phase: {np.absolute(arealm2)}, {np.angle(arealm2, deg=True)}"
        )
        logger.info(
            f"M2 diff amp, phase: {np.absolute(diffm2)}, {np.angle(diffm2, deg=True)}"
        )
        logger.info(
            f"M2 eng amp, phase: {np.absolute(engm2)}, {np.angle(engm2, deg=True)}"
        )
        logger.info(
            f"O1 areal amp, phase: {np.absolute(arealo1)}, {np.angle(arealo1, deg=True)}"
        )
        logger.info(
            f"O1 diff amp, phase: {np.absolute(diffo1)}, {np.angle(diffo1, deg=True)}"
        )
        logger.info(
            f"O1 eng amp, phase: {np.absolute(engo1)}, {np.angle(engo1, deg=True)}"
        )
        logger.info(f"Predicted Tides:\n {self.pred_tides}")

        pred_tides_file = f"{spotdir}/{self.network}_{self.fcid}_pred_tides.txt"
        header = "m2-r m2-i o1-r o1-i"
        np.savetxt(pred_tides_file, self.pred_tides, header=header)
        self.pred_tides_name = step_params.get("Name", "spotl_results")

    def calculate_orientation_matrix(self, step_params: dict):
        if (
            step_params.get("Source", {}).get("ObservedTides", None)
            == self.obs_tides_name
        ):
            obs_tides = self.obs_tides
        else:
            logger.error("Missing or unknown step parameter for Source: ObservedTides")
        if (
            step_params.get("Source", {}).get("PredictedTides", None)
            == self.pred_tides_name
        ):
            pred_tides = self.pred_tides
        else:
            logger.error("Missing or unknown step parameter for Source: PredictedTides")

        # Calculate the orentation matrix from the coupling matrix
        self.coupling_mat = np.linalg.lstsq(pred_tides.T, obs_tides.T, rcond=-1)[0].T
        res2 = np.linalg.lstsq(pred_tides.T, obs_tides.T, rcond=-1)[1]
        self.orient_mat = np.linalg.pinv(self.coupling_mat)
        logger.info(f"Coupling Matrix:\n{self.coupling_mat}")
        logger.info(f"Tidal orientation Matrix:\n{self.orient_mat}")
        logger.info(
            f"Residuals (obs - modeled):\n{obs_tides - np.dot(self.coupling_mat, pred_tides)}"
        )
        # Sum of squared residuals,
        # divided by number of data (4 per gauge)
        logger.info(f"RMSE, per gauge:\n{np.sqrt(res2 / 4)}")
        logger.info(f"RMSE:\n{np.sqrt(sum(res2) / 16)}")
        # Manufacturers Matrix
        lab_orient_mat = self.metadata.strain_matrices["lab_strain_matrix"]
        lab_coupling_mat = np.linalg.pinv(lab_orient_mat)
        logger.info(f"Lab coupling matrix:\n{lab_coupling_mat}")
        logger.info(f"Lab orientation matrix:\n{lab_orient_mat}")
        # RMSI from hodgkinson et al 2013, but with lab matrix instead of fully constrained matrix
        logger.info(
            f"RMSI between tidal and lab calibrations, if close, rmsi ~= 0\n"
            f"{np.sqrt(sum(sum((np.dot(self.orient_mat, lab_coupling_mat) - np.identity(3)) ** 2)) / 9)}"
        )

        # logger.info('Coupling coefficients assuming CH0 azimuth:')
        # ch0_az = 90 - self.metadata.orientation  # CCW positive from East
        # ch1_az = ch0_az + 60;
        # ch2_az = ch0_az + 120;
        # ch3_az = ch0_az + 150
        # cosines = np.array(
        #     [np.cos(2 * ch0_az * np.pi / 180), np.cos(2 * ch1_az * np.pi / 180), np.cos(2 * ch2_az * np.pi / 180),
        #      np.cos(2 * ch3_az * np.pi / 180)])
        # sines = np.array(
        #     [np.sin(2 * ch0_az * np.pi / 180), np.sin(2 * ch1_az * np.pi / 180), np.sin(2 * ch2_az * np.pi / 180),
        #      np.sin(2 * ch3_az * np.pi / 180)])
        # logger.info(f'Areal: {self.coupling_mat[:, 0] * 2}')
        # logger.info(f'Differential: {self.coupling_mat[:, 1] * 2 / cosines}')
        # logger.info(f'Shear: {self.coupling_mat[:, 2] * 2 / sines}')
        # logger.info(f'Installation CH0 azimuth:\n{self.metadata.orientation} or {self.metadata.orientation - 180}')
        # logger.info(f'CH 0 Azimuth assuming d=3.0 (should be double checked):')
        # differential = [(90 - np.arccos(self.coupling_mat[0, 1] * 2 / 3) * 180 / np.pi),
        #                 (90 - np.arccos(self.coupling_mat[1, 1] * 2 / 3) * 180 / np.pi) - 60,
        #                 (90 - np.arccos(self.coupling_mat[2, 1] * 2 / 3) * 180 / np.pi) - 120,
        #                 (90 - np.arccos(self.coupling_mat[3, 1] * 2 / 3) * 180 / np.pi) - 150]
        # logger.info(f'Differential:\n{differential} or \n{differential + 180}')
        # engineering = [(90 - np.arcsin(self.coupling_mat[0, 2] * 2 / 3) * 180 / np.pi),
        #                (90 - np.arcsin(self.coupling_mat[1, 2] * 2 / 3) * 180 / np.pi) - 60,
        #                (90 - np.arcsin(self.coupling_mat[2, 2] * 2 / 3) * 180 / np.pi) - 120,
        #                (90 - np.arcsin(self.coupling_mat[3, 2] * 2 / 3) * 180 / np.pi) - 150]
        # logger.info(f'Engineering:\n{engineering} or \n{engineering + 180}')

    def apply_corrections(self, ts: Timeseries, step_params: dict):
        data_df = ts.data.copy()
        if step_params.get("Corrections", None):
            for correction in step_params["Corrections"]:
                data_df = data_df - self.ts[correction].data
        ts2 = Timeseries(
            data=data_df,
            quality_df=ts.quality_df,
            series="corrected",
            units="microstrain",
            period=ts.period,
            level="2a",
        )
        self._print_plot_save(ts2, step_params)
        return ts2

    def plot_snr(self, ts: Timeseries, step_params: dict):
        if step_params.get("Name", None):
            default_name = f'{self.network}_{self.fcid}_{step_params["Name"]}'
        else:
            default_name = f"{self.network}_{self.fcid}"
        if "SaveAs" in step_params.get("PlotParameters", {}):
            plotname = step_params["PlotParameters"]["SaveAs"]
        else:
            plotname = f"{default_name}.png"
        save_as = f"{self.plotdir}/{plotname}"

        df_lowpass = self._snr_lowpass(ts.data)
        df_downsample = self._snr_downsample(df_lowpass)
        df_highpass = self._snr_highpass(df_downsample)
        self._plot_snr(df_highpass, save_as)

    def _snr_lowpass(self, df: pd.DataFrame):
        """
        2hr lowpass filter for tidal SNR analysis
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """

        [bn, an] = signal.butter(5, 600 / (2 * 3600), btype="low")
        df_lowpass = pd.DataFrame(index=df.index)
        for channel in df.columns:
            df_lowpass[channel] = signal.filtfilt(
                bn, an, df[channel]
            )  # zero phase filter the interpolated data
        return df_lowpass

    def _snr_downsample(self, df: pd.DataFrame):
        """
        downsample to hourly
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        return df.loc[df.index._data.minute == 0]

    def _snr_highpass(self, df: pd.DataFrame):
        """
        7 day highpass filter for tidal SNR analysis
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        days = 7
        fc = 1 / (days * 24 * 3600)
        fs = 1 / 3600
        [bn, an] = signal.butter(5, fc / (fs / 2), btype="high")
        df_highpass = pd.DataFrame(index=df.index)
        for channel in df.columns:
            df_highpass[channel] = signal.filtfilt(bn, an, df[channel])
        return df_highpass

    def _plot_snr(self, df: pd.DataFrame, save_as: str):
        m2 = 1.933593750000
        m2upper = 1.968750000000
        m2lower = 1.875
        o1 = 0.925781250000
        o1upper = 0.960937500000
        o1lower = 0.867187500000

        # for pwelch window is 2 months, overlap is 50%, should be 2 windows for 3 months of data
        # win = 744*2
        win = 744 * 2
        over = win / 2

        o1_snr = []
        m2_snr = []
        x = np.arange(
            0, 12, 12 / 1024
        )  # interpolate to 1024 frequency values between 0 and 12 hours
        nlm2 = np.argmax(x == m2lower)
        num2 = np.argmax(x == m2upper)
        nm2 = np.argmax(x == m2)

        nlo1 = np.argmax(x == o1lower)
        nuo1 = np.argmax(x == o1upper)
        no1 = np.argmax(x == o1)

        for channel in df.columns:
            # fs=24 is 24 samples per day, which leads to x-axis in cycles per day
            f, power = signal.welch(
                x=df[channel].values, fs=24, nperseg=win, noverlap=over
            )
            interp_f = interpolate.interp1d(f, power)
            y = interp_f(x)
            m2_snr.append(y[nm2] / ((y[num2] + y[nlm2]) / 2))
            o1_snr.append(y[no1] / ((y[nuo1] + y[nlo1]) / 2))
            # plot_m2(channel, x,y,nm2,nlm2,nlm2)
            # plot_o1(channel, x,y,no1,nlo1, nuo1)
            self._plot_m2_o1(
                self.fcid,
                channel,
                x,
                y,
                m2,
                m2upper,
                m2lower,
                o1,
                o1upper,
                o1lower,
                nm2,
                nlm2,
                num2,
                no1,
                nlo1,
                nuo1,
                df.index[0],
                df.index[-1],
                save_as.replace(".png", f"{channel}.png"),
            )

    def _plot_m2_o1(
        self,
        fcid,
        channel,
        x,
        y,
        m2,
        m2upper,
        m2lower,
        o1,
        o1upper,
        o1lower,
        nm2,
        nlm2,
        num2,
        no1,
        nlo1,
        nuo1,
        start,
        end,
        save_as,
    ):
        plt.figure(figsize=(12, 8))
        plt.plot(x, y)
        plt.scatter(x, y, s=10)
        plt.scatter(
            x[nm2], y[nm2], color="red"
        )  # , label="m2_signal: " + str(round(y[nm2], 6)))
        plt.scatter(x[nlm2], y[nlm2], color="green")
        plt.scatter(x[num2], y[num2], color="green")
        plt.scatter(
            x[nm2], (y[nlm2] + y[num2]) / 2, color="green",
        )
        # label="m2_noise: " + str(round((y[nlm2] + y[num2]) / 2, 6)))
        m2_snr = round(y[nm2] / ((y[num2] + y[nlm2]) / 2), 2)
        plt.axvline(m2, color="red", label="m2_snr: " + str(m2_snr))
        plt.axvline(m2lower, color="green")
        plt.axvline(m2upper, color="green")
        plt.scatter(
            x[no1], y[no1], color="orange"
        )  # , label="o1_signal: " + str(round(y[no1], 6)))
        plt.scatter(x[nlo1], y[nlo1], color="green")
        plt.scatter(x[nuo1], y[nuo1], color="green")
        plt.scatter(
            x[no1], (y[nlo1] + y[nuo1]) / 2, color="green",
        )
        # label="o1_noise: " + str(round((y[nlo1] + y[nuo1]) / 2, 6)))
        o1_snr = round(y[no1] / ((y[nuo1] + y[nlo1]) / 2), 2)
        plt.axvline(o1, color="orange", label="o1_snr: " + str(o1_snr))
        plt.axvline(o1lower, color="green")
        plt.axvline(o1upper, color="green")
        plt.yscale("log")
        plt.xlim(0.82, 2.1)
        plt.ylim(1e-10, 1e-1)
        plt.xlabel("Cycles per day")
        plt.ylabel("Power")
        plt.legend()
        plt.title(f"{fcid} {channel} {start} to {end}")
        plt.savefig(save_as)
        logger.info(f"m2_snr: {m2_snr}  o1_snr:{o1_snr}")

    def compare_plots(self, step_params: dict):

        series_list = step_params.get("SeriesList", [])
        num_series = len(series_list)
        if step_params.get("Type", None):
            type = step_params["Type"]
        else:
            type = self.parameters["PlotParameters"]["Type"]
        if step_params.get("Zero", None):
            zero = step_params["Zero"]
        else:
            zero = self.parameters["PlotParameters"]["Zero"]
        if num_series:
            colors = ["black", "blue", "green", "red"]
            for series_num in range(num_series):
                series_dict = list(series_list[series_num].values())[0]
                label = f"{series_dict.get('Name', '')}"
                ts = self.ts[series_dict.get("Source", {}).get("Name", None)]
                logger.info(
                    f"Starting with timeseries {series_dict.get('Source', {}).get('Name',None)}"
                )
                df = ts.data.copy()
                if series_num == 0:
                    fig, axs = plt.subplots(
                        len(ts.columns), 1, figsize=(12, 10), squeeze=False
                    )
                correction_list = series_dict.get("Corrections", None)
                if correction_list:
                    for correction in correction_list:
                        logger.info(f"applying correction {self.ts[correction].series}")
                        if series_dict.get("CalibrationMatrix", None):
                            logger.info(
                                f"applying {series_dict['CalibrationMatrix']} calibration matrix to correction."
                            )
                            series_dict["Print"] = series_dict["Plot"] = series_dict[
                                "Save"
                            ] = False
                            df -= self.apply_calibration_matrix(
                                self.ts[correction], series_dict
                            ).data
                        else:
                            df -= self.ts[correction].data

                for i, ch in enumerate(df.columns):
                    axs[i][0].set_title(ch)
                    if zero:
                        df -= df.loc[df.first_valid_index()]
                    if type == "line":
                        axs[i][0].plot(df[ch], color=colors[series_num], label=label)
                    elif type == "scatter":
                        axs[i][0].scatter(
                            df.index, df[ch], color=colors[series_num], s=2, label=label
                        )
                    else:
                        logger.error("Plot type must be either 'line' or 'scatter'")
                    if ts.units:
                        axs[i][0].set_ylabel(ts.units)
                    axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
                    axs[i][0].legend()

            if step_params.get("Name", None):
                fig.suptitle(f"{self.network}_{self.fcid}_{step_params['Name']}")
                plot_name = f"{self.network}_{self.fcid}_{step_params['Name']}.png"
            else:
                fig.suptitle(f"{self.network}_{self.fcid}")
                plot_name = f"{self.network}_{self.fcid}_plot_comparison.png"

            fig.tight_layout()

            logger.info(f"Saving plot to {plot_name}")
            plt.savefig(f"{self.plotdir}/{plot_name}")

        else:
            logger.error("No valid timeseries to plot")

        #
        # if step_params.get('Source',{}).get('Name1',None):
        #     ts1 = self.ts[step_params['Source']['Name1']]
        # if step_params.get('Source', {}).get('Name2', None):
        #     ts2 = self.ts[step_params['Source']['Name2']]
        #
        # figg, axs = plt.subplots(len(self.columns), 1, figsize=(12, 10), squeeze=False)
        # if step_params.get('Name',None):
        #     fig.suptitle(step_params['Name'])
        # else:
        #     fig.suptitle(f"{self.metadata.fcid}")
        # # else:
        # #     if self.period < 1:
        # #         sample_rate = f"{int(1/self.period)}hz"
        # #     else:
        # #         sample_rate = f"{self.period}s"
        # #     fig.suptitle(f"{self.metadata.network}_{self.metadata.fcid}_{sample_rate}_{self.series} ")
        # if remove_9s:
        #     df = self.remove_999999s(interpolate=False)
        # else:
        #     df = self.data.copy()
        # if zero:
        #     df = df - df.iloc[0]
        # if detrend:
        #     if detrend == 'linear':
        #         for ch in self.columns:
        #             df[ch] = signal.detrend(df[ch], type='linear')
        #     else:
        #         logger.error('Only linear detrend implemented')
        # for i, ch in enumerate(self.columns):
        #     if type == 'line':
        #         axs[i][0].plot(df[ch], color='black', label=ch)
        #     elif type == 'scatter':
        #         axs[i][0].scatter(df.index, df[ch], color='black', s=2, label=ch)
        #     else:
        #         logger.error("Plot type must be either 'line' or 'scatter'")
        #     if self.units:
        #         axs[i][0].set_ylabel(self.units)
        #     if ymin or ymax:
        #         axs[i][0].set_ylim(ymin, ymax)
        #     axs[i][0].ticklabel_format(axis='y', useOffset=False, style='plain')
        #     axs[i][0].legend()
        # fig.tight_layout()
        # if save_as:
        #     logger.info(f"Saving plot to {save_as}")
        #     plt.savefig(save_as)

    # def compare_snr(self,
    #                 ts1: Timeseries,
    #                 ts2: Timeseries,
    #                 step_params: dict
    #                 ):
    #     '''
    #     Given two timeseries, plot snr for both on the same plots (1 plot per channel)
    #     :param ts1:
    #     :param ts2:
    #     :param step_params:
    #     :return:
    #     '''
    #     if step_params.get('Name', None):
    #         default_name = f'{self.network}_{self.fcid}_{step_params["Name"]}'
    #     else:
    #         default_name = f'{self.network}_{self.fcid}'
    #     if 'SaveAs' in step_params.get('PlotParameters', {}):
    #         plotname = step_params['PlotParameters']['SaveAs']
    #     else:
    #         plotname = f"{default_name}.png"
    #     save_as = f"{self.plotdir}/{plotname}"
    #
    #     df_lowpass1 = self._snr_lowpass(ts1.data)
    #     df_downsample1 = self._snr_downsample(df_lowpass1)
    #     df_highpass1 = self._snr_highpass(df_downsample1)
    #
    #     df_lowpass2 = self._snr_lowpass(ts2.data)
    #     df_downsample2 = self._snr_downsample(df_lowpass2)
    #     df_highpass2 = self._snr_highpass(df_downsample2)
    #
    #

    # def upsample(self,
    #              ts: Timeseries,
    #              target_period: float,
    #              start: datetime.datetime,
    #              end: datetime.datetime):
    #     # method for upsampling a timeseres to higher sample rate
    #     # original data marked as quality 'g', interpolated data marked as quality 'i'
    #     # returns new Timeseries
    #
    #     g_flags = ts.data.index
    #     channels = ts.data.columns
    #     #print(self.data)
    #     freq = f'{target_period}S'
    #     data_df = ts.data.reindex(index=pd.date_range(start, end, freq=freq)).interpolate()
    #     quality_df = ts.quality_df.reindex(data_df.index)
    #     quality_df = quality_df.fillna('i')
    #     level = "1"
    #     return Timeseries(data=data_df,
    #                       quality_df=quality_df,
    #                       series=ts.series,
    #                       units=ts.units,
    #                       level=level,
    #                       period=target_period,
    #                       )
    #     #print(self.data)
    #     #print(self.quality_df)

    # def load_atmp_data(self,
    #                    step_params: dict):
    #     ts = self.load_data(step_params)
    #
    #
    #     # todo: implement scale factor from metadata
    #     #       parse through dates and ensure using correct scale factor
    #     # channel_code = step_params.get('Source', {}).get('ChannelCode', None)
    #     # response_metadata = inv_client.get_stations(network=self.network,
    #     #                                     station=self.fcid,
    #     #                                     channel=channel_code,
    #     #                                     level='response')
    #     #
    #     # scale_factor = response_metadata[0][0][0].response.instrument_sensitivity.value
    #     # if channel_code == 'RDO':
    #     #     if step_params.get('Source', {}).get('Units', None) == 'hPa':
    #     #         ts.data = ts.data * scale_factor * 10
    #     #     else:
    #     #         ts.data = ts.data * scale_factor
    #     return ts

    # def load_300s_atmp(self,
    #                    start: datetime.datetime,
    #                    end: datetime.datetime,
    #                    source='GTSM',
    #                    demean=False):
    #
    #     if source == 'GTSM':
    #         # start from 30m GTSM low rate pressure data
    #         try:
    #             logger.info('Loading 30m pressure data from miniseed')
    #             st = self.download_mseed(loc="T*", cha="RDO", start=start, end=end)
    #             df = self.mseed2pandas(st)
    #             # read scale factor (conversion to kpa), then multiply by 10 because we actually want mbar
    #             baro_meta = inv_client.get_stations(network=self.network, station=self.fcid, channel="RDO",
    #                                                 level='response')
    #             scale_factor = baro_meta[0][0][0].response.instrument_sensitivity.value
    #             df = df * scale_factor * 10
    #             # demean the data
    #             # Need to subtract 10 x pmedian (addLinearStrain.m) and then add offsets (edits/fcid.atm.edits)
    #             if demean:
    #                 # these values are hardcoded for B073 currently.  TODO: add programatically
    #                 df = df - 96.364 * 10 + 50.78 - 42.60
    #             print(df)
    #             ts = Timeseries(data=df,
    #                             series='raw',
    #                             units='mbar',
    #                             period=1800,
    #                             level='0',
    #                             )
    #             return self.upsample(ts, target_period=300, start=start, end=end)
    #
    #         except Exception as e:
    #             logger.exception('Unable to load GTSM pressure data')
    #
    #     elif source == 'Setra':
    #         # start from 1s Setra high rate pressure data
    #         # not fully implemented yet.  need conversion to hPa, differentiate HA vs regular
    #         # look up response based on time in metadata
    #         try:
    #             logger.info('Loading 1s miniseed pressure data from Setra Barometer')
    #             st = self.download_mseed(loc="*", cha="LDO", start=start, end=end)
    #             print(st)
    #             df = self.mseed2pandas(st)
    #             print(df)
    #             baro_meta = inv_client.get_stations(network=self.network, station=self.fcid, channel="LDO",
    #                                                 level='response')
    #             print(baro_meta[0][0])
    #         except Exception as e:
    #             logger.exception('Unable to load Setra pressure data')
    #
    #     elif source == 'Metpack':
    #         # start from 1s Metpack high rate pressure data
    #         try:
    #             logger.info('Loading 1s ascii pressure data from Metpack')
    #         except Exception as e:
    #             logger.exception('Unable to load Metpack pressure data')
    #     else:
    #         logger.error('Invalid atmp source.  Implemented options are ["GTSM", "Setra", "Metpack"]')

    # def load_1s_s3_tdb(self,
    #                    start: datetime.datetime,
    #                    end: datetime.datetime,
    #                    bucket: str):
    #     session_edid = get_session_edid(self.network, self.fcid, session="Hour")
    #     uri = f's3://{bucket}/{session_edid}.tdb'
    #     data_types = ['CH0', 'CH1', 'CH2', 'CH3']
    #     timeseries = ['raw']
    #     attrs = ['data']
    #     self.array_1s = StrainTiledbArray(uri=uri, period=1, location='S3')
    #     df = self.array_1s.read_to_df(data_types, timeseries, attrs, start, end)
    #     if len(df):
    #         return Timeseries(data=df,
    #                           metadata=self.metadata,
    #                           series='raw',
    #                           units='counts',
    #                           period=1,
    #                           level='0',
    #                           )
    #     else:
    #         return None

    # def load_local_tdb(self,
    #                    uri: str,
    #                    start: datetime.datetime,
    #                    end: datetime.datetime,
    #                    units: str,
    #                    period: float,
    #                    data_types: list,
    #                    series: str,
    #                    attrs: list
    #                    ):
    #     if self.local_archive:
    #         uri = f"{self.local_archive}/{uri}"
    #     array = StrainTiledbArray(uri=uri, period=period, location='local')
    #     #make any strings into 1 item lists
    #     data_types = [data_types] if isinstance(data_types, str) else data_types
    #     #timeseries = [series] if isinstance(series, str) else series
    #     attrs = [attrs] if isinstance(attrs, str) else attrs
    #
    #     if 'data' in attrs:
    #         data_df = array.read_to_df(data_types, timeseries, attrs='data', start_dt=start, end_dt=end)
    #     else:
    #         data_df = None
    #     if 'quality' in attrs:
    #         quality_df = array.read_to_df(data_types, timeseries, attrs='quality', start_dt=start, end_dt=end)
    #     else:
    #         quality_df = None
    #     if 'level' in attrs:
    #         level_df = array.read_to_df(data_types, timeseries, attrs='level', start_dt=start, end_dt=end)
    #         level = level_df[data_types[0]].iloc[0]
    #     else:
    #         level_df = None
    #         level = array.read_to_df(data_types=[data_types[0]],
    #                                  timeseries=series,
    #                                  attrs=['level'],
    #                                  start_dt=start,
    #                                  end_dt=end).values[0][0]
    #     if 'version' in attrs:
    #         version_df = array.read_to_df(data_types, timeseries, attrs='version', start_dt=start, end_dt=end)
    #     else:
    #         version_df = None
    #
    #     return Timeseries(data=data_df,
    #                       quality_df=quality_df,
    #                       level_df=level_df,
    #                       version_df=version_df,
    #                       metadata=self.metadata,
    #                       series=series,
    #                       units=units,
    #                       period=period,
    #                       level=level,
    #                       )
    #
    #
