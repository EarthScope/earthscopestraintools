import numpy as np
import pandas as pd
import datetime

from obspy import read, Stream
from obspy.clients.fdsn import Client
from earthscopestraintools.timeseries import Timeseries

inv_client = Client("IRIS")
from requests.exceptions import HTTPError


import logging


logger = logging.getLogger(__name__)


def load_mseed_to_df(
    net: str,
    sta: str,
    loc: str,
    cha: str,
    start: str,
    end: str,
    save_as: str = None,
    print_url: bool = False,
    print_traces: bool = True
):
    """
    Load miniseed data from fdsnws-dataselect a pandas dataframe, with time as the index and
    each channel as a column.  Uses obspy as an intermediate step.

    :param net: FDSN 2 character network code, accepts wildcards
    :type net: str
    :param sta: FDSN 4 character station code, accepts wildcards
    :type sta: str
    :param loc: FDSN 2 character location code, accepts wildcards
    :type loc: str
    :param cha: FDSN 3 character channel code, accepts wildcards
    :type cha: str
    :param start: start time of query as ISO formatted string
    :type start: str
    :param end: end time of query as ISO formatted string
    :type end: str
    :param save_as: filename to save miniseed data, defaults to None
    :type save_as: str, optional
    :param print_url: show dataselect url, defaults to False
    :type print_url: bool, optional
    :param print_traces: show traces loaded into obspy, defaults to True
    :type print_traces: bool, optional
    :return: data loaded into pandas dataframe
    :rtype: pd.DataFrame
    """
    st = download_mseed(
        net=net, sta=sta, loc=loc, cha=cha, start=start, end=end, print_url=print_url
    )
    if save_as:
        st.write(filename=save_as)
    df = mseed2pandas(st, print_traces=print_traces)
    return df


def load_mseed_file_to_df(filename: str):
    """read miniseed file from local disk into pandas dataframe, with time as the index and
    each channel as a column.  Uses obspy as an intermediate step.
 

    :param filename: miniseed filename
    :type filename: str
    :return: data loaded into pandas dataframe
    :rtype: pd.DataFrame
    """
    st = read(filename)
    df = mseed2pandas(st)
    return df


def save_mseed_file(
    net: str, sta: str, loc: str, cha: str, start: str, end: str, filename: str
):
    """Load miniseed data from fdsnws-dataselect into obspy, and save as miniseed.

    :param net: FDSN 2 character network code, accepts wildcards
    :type net: str
    :param sta: FDSN 4 character station code, accepts wildcards
    :type sta: str
    :param loc: FDSN 2 character location code, accepts wildcards
    :type loc: str
    :param cha: FDSN 3 character channel code, accepts wildcards
    :type cha: str
    :param start: start time of query as ISO formatted string
    :type start: str
    :param end: end time of query as ISO formatted string
    :type end: str
    :param filename: filename to save miniseed data
    :type filename: str
    """
    st = download_mseed(net=net, sta=sta, loc=loc, cha=cha, start=start, end=end)
    st.write(filename)


def download_mseed(
    net: str,
    sta: str,
    loc: str,
    cha: str,
    start: str,
    end: str,
    print_url: bool = False,
):
    """Load miniseed data from fdsnws-dataselect into obspy Stream object.

    :param net: FDSN 2 character network code, accepts wildcards
    :type net: str
    :param sta: FDSN 4 character station code, accepts wildcards
    :type sta: str
    :param loc: FDSN 2 character location code, accepts wildcards
    :type loc: str
    :param cha: FDSN 3 character channel code, accepts wildcards
    :type cha: str
    :param start: start time of query as ISO formatted string
    :type start: str
    :param end: end time of query as ISO formatted string
    :type end: str
    :param print_url: show dataselect url, defaults to False
    :type print_url: bool, optional
    :return: data as Stream object
    :rtype: obspy.Core.Stream
    """
    logger.info(
        f"{net} {sta} Loading {loc} {cha} from {start} to {end} from Earthscope DMC miniseed"
    )
    url = (
        f"https://service.iris.edu/fdsnws/dataselect/1/query?net={net}"
        f"&sta={sta}&loc={loc}&cha={cha}&starttime={start}&endtime={end}"
        f"&format=miniseed&nodata=404"
    )
    if print_url:
        logger.info(f"{net} {sta} Reading from {url}")
    try:
        st = read(url)
    except HTTPError as e:
        logger.error(f"{net} {sta} No data found for specified query {url}")
        st = Stream()
    return st


def mseed2pandas(st: Stream, print_traces=True):
    """Restructure data from obspy.Core.Stream to pandas.DataFrame with time as the index and
    each channel as a column.

    :param st: Stream object containing one or more Traces
    :type st: obspy.Core.Stream
    :param print_traces: show traces loaded into obspy, defaults to True
    :type print_traces: bool, optional
    :return: data loaded into pandas dataframe
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame()
    dfs = {}
    for i, tr in enumerate(st):
        channel = fdsn2bottlename(tr.stats.channel)
        start = tr.stats["starttime"]
        stop = tr.stats["endtime"]
        step = datetime.timedelta(seconds=tr.stats["delta"])
        if print_traces:
            logger.info(
                f"    Trace {i + 1}. {start}:{stop} mapping {tr.stats.channel} to {channel}"
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


def fdsn2bottlename(channel):
    """convert FDSN channel code into bottlename

    :param channel: FDSN channel
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
        "RRO": "rain",
        "RK1": "LoggerDegC",
        "RKD": "DownholeDegC",
        "RE1": "BatteryVolts",
        "REO": "SolarAmps",
        "RE2": "SystemAmps",
        "RK2": "PowerBoxDegC",
        "LDD": "PorePressure",
        "LKD": "PoreDegC",
        "VEP": "Q330Volts"
    }

    return codes[channel]


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
    """Load miniseed data from fdsnws-dataselect into Timeseries object.  

    :param network: FDSN 2 character network code, accepts wildcards
    :type network: str
    :param station: FDSN 4 character station code, accepts wildcards
    :type station: str
    :param location: FDSN 2 character location code, accepts wildcards
    :type location: str
    :param channel: FDSN 3 character channel code, accepts wildcards
    :type channel: str
    :param start: start time of query as ISO formatted string
    :type start: str
    :param end: end time of query as ISO formatted string
    :type end: str
    :param name: name of timeseries, including station name.  useful for showing stats. , defaults to None
    :type name: str, optional
    :param period: sample period of data, defaults to None
    :type period: float, optional
    :param series: description of timeseries, ie 'raw', 'microstrain', 'atmp_c', 'tide_c', 'offset_c', 'trend_c', defaults to "raw"
    :type series: str, optional
    :param units: units of timeseries, defaults to ""
    :type units: str, optional
    :param to_nan: option to convert 999999 gap fill values to numpy.nan, defaults to True
    :type to_nan: bool, optional
    :param scale_factor: scale factor to apply to miniseed data, defaults to None
    :type scale_factor: float, optional
    :return: Timeseries object containing data loaded from miniseed
    :rtype: earthscopestraintools.timeseries.Timeseries
    """
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
