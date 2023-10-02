# ETL from ascii level2 files to tiledb
# write tiledb arrays locally, then aws s3 sync
import tarfile
import tiledb
import pandas as pd
import datetime
import sys, os
import shutil

# from edid import find_station_edid
#from earthscopestraintools.datasources_api_interact import get_station_edid
import json
from io import BytesIO
import requests
import configparser

from earthscopestraintools.tiledbtools import (
    ProcessedStrainWriter,
    ProcessedStrainReader,
)


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
workdir = ""


# def write_df_to_tiledb(df, array):
#     # print(df)
#     mode = "append"
#     tiledb.from_pandas(
#         uri=array.uri,
#         dataframe=df,
#         index_dims=["data_type", "timeseries", "time"],
#         mode=mode,
#         ctx=array.ctx,
#     )
#
#     # update the string dimension metadata
#     data_type = df["data_type"].unique()
#     timeseries = df["timeseries"].unique()
#     if type(data_type) == str:
#         data_type = [data_type]
#     if type(timeseries) == str:
#         timeseries = [timeseries]
#     with tiledb.open(array.uri, "r", ctx=array.ctx) as A:
#         try:
#             dimension_json = A.meta["dimensions"]
#         except KeyError:
#             dimension_json = '{"data_types":[], "timeseries":[]}'
#
#         dimension_dict = json.loads(dimension_json)
#         # print(dimension_dict)
#         for item in data_type:
#             if item not in dimension_dict["data_types"]:
#                 dimension_dict["data_types"].append(item)
#         for item in timeseries:
#             if item not in dimension_dict["timeseries"]:
#                 dimension_dict["timeseries"].append(item)
#
#         # print(dimension_dict)
#         with tiledb.open(array.uri, "w", ctx=array.ctx) as A:
#             A.meta["dimensions"] = json.dumps(dimension_dict)
#


def loop_through_ts(filebase, file, writer):
    df = pd.read_csv(filebase + "/" + file, delimiter="\s+")
    label = df["strain"][0] + "(mstrain)"
    df = df.rename(
        columns={
            "strain": "data_type",
            label: "microstrain",
            "s_offset": "offset_c",
            "detrend_c": "trend_c",
            "MJD": "mjd",
            "date": "time",
        }
    )
    df["data_type"] = df["data_type"].str.replace("gauge", "CH")
    # display(df)
    cols = ["data_type", "timeseries", "time", "data", "quality", "level", "version"]
    # convert time string to unix ms
    df["time"] = (pd.to_datetime(df["time"]).astype(int) / 10**6).astype(int)

    timeseries = "microstrain"
    tmp_df = df[["data_type", "time", timeseries, "strain_quality", "level", "version"]]
    tmp_df = tmp_df.rename(columns={timeseries: "data", "strain_quality": "quality"})
    tmp_df["timeseries"] = timeseries
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    writer.write_df_to_tiledb(tmp_df)

    timeseries = "offset_c"
    tmp_df = df[["data_type", "time", timeseries, "level", "version"]]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df["timeseries"] = timeseries
    tmp_df["quality"] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    writer.write_df_to_tiledb(tmp_df)

    timeseries = "tide_c"
    tmp_df = df[["data_type", "time", timeseries, "level", "version"]]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df["timeseries"] = timeseries
    tmp_df["quality"] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    writer.write_df_to_tiledb(tmp_df)

    timeseries = "trend_c"
    tmp_df = df[["data_type", "time", timeseries, "level", "version"]]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df["timeseries"] = timeseries
    tmp_df["quality"] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    writer.write_df_to_tiledb(tmp_df)

    timeseries = "atmp_c"
    tmp_df = df[["data_type", "time", timeseries, "atmp_c_quality", "level", "version"]]
    tmp_df = tmp_df.rename(columns={timeseries: "data", "atmp_c_quality": "quality"})
    tmp_df["timeseries"] = timeseries
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    writer.write_df_to_tiledb(tmp_df)

    if df["data_type"][0] == "CH0":
        tmp_df = df[["data_type", "time", "atmp", "level", "version"]]
        tmp_df = tmp_df.rename(columns={"atmp": "data"})
        tmp_df = tmp_df.assign(data_type="atmp")
        tmp_df["timeseries"] = "hpa"
        tmp_df["quality"] = " "
        tmp_df = tmp_df[cols]
        logger.info(f"{timeseries}: {len(tmp_df)} samples")
        writer.write_df_to_tiledb(tmp_df)


def etl_yearly_ascii_file(
    network, station, year, delete_array=False, workdir="", print_it=False
):
    os.makedirs(workdir, exist_ok=True)
    # edid = get_station_edid(station)
    # uri = f"{workdir}/{edid}_level2.tdb"
    uri = f"{workdir}/{network}_{station}_l2_etl.tdb"
    logger.info(f"Array uri: {uri}")
    writer = ProcessedStrainWriter(uri)
    if delete_array:
        writer.array.delete()

    # create new array if needed.  note: array_exists only works locally not in s3.
    if not tiledb.array_exists(writer.array.uri):
        writer.array.create(schema_type="3d", schema_source="s3")
        writer.array.set_array_meta(network=network, station=station, period=300)

    filebase = station + "." + year + ".bsm.level2"
    url = "http://bsm.unavco.org/bsm/level2/" + station + "/" + filebase + ".tar"
    response = requests.get(url, stream=True)
    print(url)
    tar = tarfile.open(fileobj=BytesIO(response.raw.read()), mode="r")
    tar.extractall()
    files = os.listdir(filebase)
    for file in files:
        logger.info(file)
        loop_through_ts(filebase, file, writer)
    shutil.rmtree(filebase)

    writer.array.consolidate_array_meta()
    writer.array.vacuum_array_meta()
    writer.array.consolidate_fragment_meta()
    writer.array.vacuum_fragment_meta()
    writer.array.consolidate_fragments()
    writer.array.vacuum_fragments()
    if print_it:
        reader = ProcessedStrainReader(uri)
        logger.info(f"Network: {reader.array.get_network()}")
        logger.info(f"Station: {reader.array.get_station()}")
        logger.info(f"Period: {reader.array.get_period()}")
        start_str = f"{year}-01-01T00:00:00"
        end_str = f"{int(year)+1}-01-01T00:00:00"
        df = reader.to_df(
            data_types=["CH0", "CH1", "CH2", "CH3", "Eee+Enn", "Eee-Enn", "2Ene"],
            timeseries="microstrain",
            attrs="data",
            start_str=start_str,
            end_str=end_str,
        )
        logger.info(f"\n{df}")


# if __name__ == "__main__":
#     workdir = "arrays"
#     network = sys.argv[1]
#     station = sys.argv[2]
#     year = sys.argv[3]
#     etl_yearly_ascii_file(network, station, year, workdir=workdir)
# years = ["2005","2006","2007","2008","2009",
#          "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
#          "2020","2021","2022"]
# for year in years:
#     etl_yearly_ascii_file(network, station, year)
#
