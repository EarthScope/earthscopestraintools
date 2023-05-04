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
):
    st = download_mseed(
        net=net, sta=sta, loc=loc, cha=cha, start=start, end=end, print_url=print_url
    )
    if save_as:
        st.write(filename=save_as)
    df = mseed2pandas(st)
    return df


def load_mseed_file_to_df(filename: str):
    st = read(filename)
    df = mseed2pandas(st)
    return df


def save_mseed_file(
    net: str, sta: str, loc: str, cha: str, start: str, end: str, filename: str
):
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
    """

    :param loc: FDSN Location code, accepts wildcards
    :param cha: FDSN Channel code, accepts wildcards
    :param start:
    :param end:
    :return:
    """
    logger.info(
        f"Loading {loc} {cha} from {start} to {end} from Earthscope DMC miniseed"
    )
    url = (
        f"https://service.iris.edu/fdsnws/dataselect/1/query?net={net}"
        f"&sta={sta}&loc={loc}&cha={cha}&starttime={start}&endtime={end}"
        f"&format=miniseed&nodata=404"
    )
    if print_url:
        logger.info(f"Reading from {url}")
    try:
        st = read(url)
    except HTTPError as e:
        logger.error(f"No data found for specified query {url}")
        st = Stream()
    return st


def mseed2pandas(st: Stream):
    df = pd.DataFrame()
    dfs = {}
    for i, tr in enumerate(st):
        channel = fdsn2bottlename(tr.stats.channel)
        start = tr.stats["starttime"]
        stop = tr.stats["endtime"]
        step = datetime.timedelta(seconds=tr.stats["delta"])
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
    """
    convert location and channel into bottlename
    :param channel: str
    :return: str
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
