# write tarred tiledb arrays from tiledb arrays
# slice by start and end time
# use in lambda with web service to deliver data in tiledb

import tiledb
import numpy as np
import pandas as pd

# import datetime
# import sys, os
import shutil
import json

# from straintiledbarray import StrainTiledbArray

from earthscopestraintools.edid import get_station_edid
from earthscopestraintools.tiledbtools import (
    ProcessedStrainReader,
    ProcessedStrainWriter,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
workdir = "arrays"

#
# def to_date(datetime64):
#     ts = pd.to_datetime(str(datetime64))
#     d = ts.strftime("%Y%m%d")
#     return d


def read_date_range(network, station, start_str, end_str):
    edid = get_station_edid(network, station)
    uri = f"{workdir}/{edid}_level2.tdb"
    logger.info(f"Array uri: {uri}")
    reader = ProcessedStrainReader(uri)
    data_types = reader.array.get_data_types()
    timeseries = reader.array.get_timeseries()
    attrs = ["data", "quality", "level", "version"]
    df = reader.to_df(
        data_types=data_types,
        timeseries=timeseries,
        attrs=attrs,
        start_str=start_str,
        end_str=end_str,
        reindex=False,
    )
    return df


def convert_time_to_unix_ms(df):
    df = df.reset_index()
    df["time"] = df["time"].astype(int) / 10 ** 6
    df["time"] = df["time"].astype(np.int64)
    return df


def export_date_range(
    network, station, start_str, end_str, write_it=True, print_it=False
):
    df = read_date_range(network, station, start_str, end_str)
    df = convert_time_to_unix_ms(df)
    if print_it:
        logger.info(f"\n{df}")

    if write_it:
        uri = f"{workdir}/{network}_{station}_level2_{start_str}_{end_str}.tdb"
        writer = ProcessedStrainWriter(uri=uri)
        writer.array.delete()
        writer.array.create(schema_type="3D", schema_source="s3")
        writer.write_df_to_tiledb(df)
