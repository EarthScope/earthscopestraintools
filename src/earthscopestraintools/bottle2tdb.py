from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from earthscopestraintools.bottletar import GtsmBottleTar
import tiledb
from earthscopestraintools.tiledbtools import (
    StrainArray,
    RawStrainWriter,
    RawStrainReader,
)
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def build_strain_buffer(gbt: GtsmBottleTar, session: str):
    '''
    Testing the doc?
    '''
    logger.info(
        f"{gbt.file_metadata['filename']}: loading strain bottles into dataframe"
    )
    bottle_dfs = []
    logger.info(
        f"{gbt.file_metadata['filename']}: contains {len(gbt.bottle_list)} bottle files"
    )
    for i, name in enumerate(gbt.bottle_list):
        bottle = gbt.load_bottle(name)
        bottle.parse_filename()
        if bottle.file_metadata["channel"] in ["CH0", "CH1", "CH2", "CH3"]:
            # logger.info(bottle.file_metadata["channel"])
            bottle.read_header()
            channel = bottle.file_metadata["channel"]
            timestamps = bottle.get_unix_ms_timestamps()
            data = bottle.read_data()
            bottle.file.close()
            d = {
                "channel": channel,
                "time": timestamps,
                "data": data,
            }
            bottle_df = pd.DataFrame(data=d)
            bottle_dfs.append(bottle_df)
    tiledb_buffer = pd.concat(bottle_dfs, axis=0).reset_index(drop=True)
    tiledb_buffer["data"] = tiledb_buffer["data"].astype(np.int32)
    return tiledb_buffer


def build_ancillary_buffer(gbt: GtsmBottleTar, session: str):
    logger.info(
        f"{gbt.file_metadata['filename']}: loading ancillary bottles into dataframe"
    )
    bottle_dfs = []
    for i, name in enumerate(gbt.bottle_list):
        bottle = gbt.load_bottle(name)
        bottle.parse_filename()
        if bottle.file_metadata["channel"] not in ["CH0", "CH1", "CH2", "CH3"]:
            logger.info(bottle.file_metadata["channel"])
            bottle.read_header()
            channel = bottle.file_metadata["channel"]
            timestamps = bottle.get_unix_ms_timestamps()
            data = bottle.read_data()
            bottle.file.close()
            d = {
                "channel": channel,
                "time": timestamps,
                "data": data,
            }
            bottle_df = pd.DataFrame(data=d)
            bottle_dfs.append(bottle_df)
    tiledb_buffer = pd.concat(bottle_dfs, axis=0).reset_index(drop=True)
    tiledb_buffer["data"] = tiledb_buffer["data"].astype(np.float64)
    return tiledb_buffer


def write_buffer(uri, buffer: pd.DataFrame, cleanup_meta=False):
    writer = RawStrainWriter(uri)
    writer.write_df_to_tiledb(buffer)
    if cleanup_meta:
        writer.array.cleanup_meta()


def check_result(buffer, uri):
    try:
        start = int(buffer.time.iloc[0])
        end = int(buffer.time.iloc[-1])
        channels = list(buffer["channel"].unique())
        period = (float(buffer.time.iloc[1]) - float(buffer.time.iloc[0])) / 1000
        reader = RawStrainReader(uri, period=period)
        result_df = reader.to_df(channels=channels, start_ts=start, end_ts=end)
        logger.info(f"\n{result_df}")
    except Exception as e:
        logger.exception(f"{type(e)}")


def bottle2tdb(
    filepath: str,
    strain_uri: str,
    ancillary_uri: str,
    session: str,
    write_it=True,
    print_it=False,
    check_it=False,
):
    gbt = GtsmBottleTar(filepath, session)
    strain_buffer = build_strain_buffer(gbt, session)
    if session.casefold() == "Day".casefold():
        ancillary_buffer = build_ancillary_buffer(gbt, session)
    gbt.delete_bottles_from_disk()

    if write_it:
        try:
            logger.info(f"{filepath}: Writing to {strain_uri}")
            write_buffer(strain_uri, strain_buffer)
        except tiledb.TileDBError as e:
            logger.error(e)
            array = StrainArray(uri=strain_uri)
            array.create(schema_type="2D_INT", schema_source="s3")
            if session.casefold() == "Min".casefold():
                period = 0.05
            elif session.casefold() == "Hour".casefold():
                period = 1
            elif session.casefold() == "Day".casefold():
                period = 600
            else:
                logger.error(
                    f"Invalid session {session}, cannot determine period for array metadata"
                )
            array.set_array_meta(
                station=gbt.file_metadata["fcid"], period=period
            )  # todo: set network meta?
            logger.info(f"{filepath}: Writing to {strain_uri}")
            write_buffer(strain_uri, strain_buffer)

        if session.casefold() == "Day".casefold():
            try:
                logger.info(f"{filepath}: Writing to {ancillary_uri}")
                write_buffer(ancillary_uri, ancillary_buffer)
            except tiledb.TileDBError as e:
                logger.error(e)
                ancillary_array = StrainArray(uri=ancillary_uri)
                ancillary_array.create(schema_type="2D_FLOAT", schema_source="s3")
                ancillary_array.set_array_meta(
                    station=gbt.file_metadata["fcid"], period=1800
                )  # todo: set network meta?
                logger.info(f"{filepath}: Writing to {ancillary_uri}")
                write_buffer(ancillary_uri, ancillary_buffer)

    if print_it:
        logger.info(f"Strain Buffer:\n{strain_buffer}")
        if session.casefold() == "Day".casefold():
            logger.info(f"Ancillary Buffer:\n{ancillary_buffer}")

    if check_it:
        logger.info(f"Reading data back from array")
        logger.info(f"Network: {array.get_network()}")
        logger.info(f"Station: {array.get_station()}")
        logger.info(f"Period: {array.get_period()}")
        check_result(strain_buffer, strain_uri)
        if session.casefold() == "Day".casefold():
            logger.info(f"Network: {ancillary_array.get_network()}")
            logger.info(f"Station: {ancillary_array.get_station()}")
            logger.info(f"Period: {ancillary_array.get_period()}")
            check_result(ancillary_buffer, ancillary_uri)
