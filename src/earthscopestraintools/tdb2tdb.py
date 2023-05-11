# write tarred tiledb arrays from tiledb arrays
# slice by start and end time
# use in lambda with web service to deliver data in tiledb
# only works for processed strain data currently

import numpy as np
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


def read_date_range(uri, start_str, end_str):
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
    df["time"] = df["time"].astype(int) / 10**6
    df["time"] = df["time"].astype(np.int64)
    return df


def export_date_range(
    uri, start_str, end_str, save_as=None, write_it=True, print_it=False
):
    reader = ProcessedStrainReader(uri=uri)
    network = reader.array.get_network()
    station = reader.array.get_station()
    period = reader.array.get_period()

    df = read_date_range(uri, start_str, end_str)
    df = convert_time_to_unix_ms(df)

    if write_it:
        if save_as:
            uri2 = save_as
        else:
            uri2 = f"{workdir}/{network}_{station}_level2_{start_str}_{end_str}.tdb"
        writer = ProcessedStrainWriter(uri=uri2)
        writer.array.delete()
        writer.array.create(schema_type="3D", schema_source="s3")
        writer.array.set_array_meta(network=network, station=station, period=period)
        writer.write_df_to_tiledb(df)

    if print_it:
        reader = ProcessedStrainReader(uri=uri2)
        logger.info(f"Network: {reader.array.get_network()}")
        logger.info(f"Station: {reader.array.get_station()}")
        logger.info(f"Period: {reader.array.get_period()}")
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
        logger.info(f"\n{df}")
