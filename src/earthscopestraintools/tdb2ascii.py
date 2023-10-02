# write ascii files from tiledb arrays
# slice by start and end time

import tarfile
import tiledb
import numpy as np
import pandas as pd
import os
import json
import datetime
from earthscopestraintools.tiledbtools import ProcessedStrainReader
from earthscopestraintools.datasources_api_interact import get_station_edid

# from earthscopestraintools.straintiledbarray import StrainTiledbArray
# from edid import get_station_edid

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

workdir = "arrays"
outputdir = "ascii_output"
DATASOURCES_API_URL = "https://datasources-api.prod.earthscope.org"

def to_date(datetime64):
    ts = pd.to_datetime(str(datetime64))
    d = ts.strftime("%Y%m%d")
    return d


def read_date_range(network, station, start_str, end_str):
    edid = get_station_edid(api_url=DATASOURCES_API_URL, namespace="BNUM", station=station)
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


def write_ascii(df, station, date_range_string, write_it=True, print_it=False):
    filenames = []
    for data_type in ["CH0", "CH1", "CH2", "CH3", "Eee+Enn", "Eee-Enn", "2Ene"]:
        logger.info(f"generating ascii file for {data_type}")
        for timeseries in ["microstrain", "offset_c", "tide_c", "trend_c", "atmp_c"]:
            tmp_df = (
                df.sort_index()
                .loc[data_type, timeseries]
                .rename(columns={"data": timeseries})
            )
            if timeseries == "microstrain":
                tmp_df = tmp_df.rename(
                    columns={
                        "microstrain": f"{data_type}(mstrain)",
                        "quality": "strain_quality",
                    }
                )
                new_df = tmp_df
                new_df["strain"] = data_type
            elif timeseries == "offset_c":
                new_df["s_offset"] = tmp_df["offset_c"]
            elif timeseries == "trend_c":
                new_df["detrend_c"] = tmp_df["trend_c"]
            elif timeseries == "tide_c":
                new_df["tide_c"] = tmp_df["tide_c"]
            elif timeseries == "atmp_c":
                new_df["atmp_c"] = tmp_df["atmp_c"]
                new_df["atmp_c_quality"] = tmp_df["quality"]

        new_df["atmp"] = df["data"].sort_index().loc["atmp", "hpa"]
        # new_df["MJD"] = df["data"].loc["time_index", "mjd"]

        new_df = new_df.reset_index().rename(columns={"time": "date"})
        new_df["doy"] = new_df["date"].dt.dayofyear
        new_df["MJD"] = pd.DatetimeIndex(new_df["date"]).to_julian_date() - 2400000.5
        # logger.info(f"\n{new_df}")

        ordered_columns = [
            "strain",
            "date",
            "doy",
            "MJD",
            f"{data_type}(mstrain)",
            "s_offset",
            "strain_quality",
            "tide_c",
            "detrend_c",
            "atmp_c",
            "atmp_c_quality",
            "level",
            "version",
            "atmp",
        ]
        new_df = new_df[ordered_columns]
        output = f"{outputdir}/{station}.{date_range_string}.{data_type}.txt.gz"
        filenames.append(output)
        if print_it:
            logger.info(f"\n{new_df}")
        if write_it:
            new_df.to_csv(output, sep="\t", index=False)

    if write_it:
        # Create Tar file
        tarfilename = f"{outputdir}/{station}.bsm.level2.{date_range_string}.tar"
        logger.info(f"writing to {tarfilename}")
        tar = tarfile.open(tarfilename, "w")

        for f in filenames:
            tar.add(f, arcname=f.split("/")[-1])

        tar.close()


def write_date_range(
    network, station, start_str, end_str, write_it=True, print_it=False
):
    df = read_date_range(network, station, start_str, end_str)
    logger.info(f"\n{df}")
    date_range_string = f"{start_str}-{end_str}"
    os.makedirs(outputdir, exist_ok=True)
    write_ascii(df, station, date_range_string, write_it=write_it, print_it=print_it)
