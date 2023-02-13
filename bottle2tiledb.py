from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from bottletar import GtsmBottleTar
import tiledb
from straintiledbarray import StrainTiledbArray, Writer
from edid import find_session_edid
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def build_tiledb_buffer(gbt: GtsmBottleTar):
    # builds a tiledb buffer dataframe from a given set of bottles
    # index as row number
    # columns are data_type, timeseries, time, data, level, quality, version (3 dim + 4 attr)


    logger.info(f"{gbt.file_metadata['filename']}: loading {len(gbt.bottle_list)} bottles into dataframe")
    bottle_dfs = []
    for i, name in enumerate(gbt.bottle_list):
        bottle = gbt.load_bottle(name)
        bottle.read_header()
        #logger.info(bottle.file_metadata)
        data_type = bottle.file_metadata['channel']
        timeseries = 'raw'
        #logger.info(f"{data_type}, {timeseries}")
        timestamps = bottle.get_unix_ms_timestamps()
        #timestamps = bottle.get_datetime_timestamps()

        data = bottle.read_data()
        # Important for Min_Archive session: close open bottles when done reading to free up memory
        bottle.file.close()
        level = '0'
        quality = 'g'
        version = int(datetime.now().strftime("%Y%j%H%M%S"))
        d = {'data_type': data_type,
             'timeseries': timeseries,
             'time': timestamps,
             'data': data,
             'level': level,
             'quality': quality,
             'version': version}
        bottle_df = pd.DataFrame(data=d)
        bottle_df.loc[bottle_df['data'] == 999999, 'quality'] = 'm'
        bottle_dfs.append(bottle_df)
    tiledb_buffer = pd.concat(bottle_dfs, axis=0).reset_index(
        drop=True)
    tiledb_buffer['data'] = tiledb_buffer['data'].astype(np.float64)
    tiledb_buffer['version'] = tiledb_buffer['version'].astype(np.int64)
    gbt.delete_bottles_from_disk()
    return tiledb_buffer


# def write_buffer_to_array(array, tiledb_buffer):
#     #writes a tiledb buffer dataframe to a given tiledb array
#     tiledb.from_pandas(uri=array.uri,
#                        dataframe=tiledb_buffer,
#                        index_dims=['data_type', 'timeseries', 'time'],
#                        mode='append',
#                        ctx=array.ctx
#                        )

def to_tiledb(gbt, array):
    #method called to trigger building buffer dataframe and writing to tiledb
    #if specified array does not exist, it will make one
    try:
        #num_bottles = len(self.bottles)
        num_bottles = len(gbt.bottle_list)
        if num_bottles:
            logger.info(f"{gbt.file_metadata['filename']}: {gbt.file_metadata['session']} tar contains {num_bottles} bottles.")
            #self.tiledb_buffer = self.build_tiledb_buffer(self.bottles)
            tiledb_buffer = build_tiledb_buffer(gbt)
            logger.info(f"{gbt.file_metadata['filename']}: Writing to {array.uri}")
            writer = Writer(array=array)
            writer.write_df_to_tiledb(tiledb_buffer)
            #write_buffer_to_array(tiledb_buffer, array)
            logger.info(f"{gbt.file_metadata['filename']}: Written to {array.uri}")
            array.consolidate_fragment_meta()
            array.vacuum_fragment_meta()
            array.consolidate_array_meta()
            array.vacuum_array_meta()
            gbt.file_metadata['start_time'] = tiledb_buffer['time'].iloc[0]
            gbt.file_metadata['end_time'] = tiledb_buffer['time'].sort_values().iloc[-1]
        else:
            logger.error(f"Error: GtsmLoggerFile object {gbt.file_metadata['filename']} contains no bottles")

    except tiledb.TileDBError as e:
        try:
            logger.error(e)
            logger.warning(f"Array {array.uri} does not exist, creating.")
            array.create()
            logger.info(f"Created {array.uri}")
            logger.info(f"{gbt.file_metadata['filename']}: Writing to {array.uri}")
            writer = Writer(array=array)
            writer.write_df_to_tiledb(tiledb_buffer)
            logger.info(f"{gbt.file_metadata['filename']}: Written to {array.uri}")
            array.consolidate_fragment_meta()
            array.vacuum_fragment_meta()
            array.consolidate_array_meta()
            array.vacuum_array_meta()
            gbt.file_metadata['start_time'] = tiledb_buffer['time'].iloc[0]
            gbt.file_metadata['end_time'] = tiledb_buffer['time'].sort_values().iloc[-1]

        except Exception as e:
            logger.exception(e)

    except Exception as e:
        logger.exception(e)
#
# def build_stream(network: str,
#                    station: str,
#                    gbt: GtsmBottleTar):
#     st = obspy.core.stream.Stream()
#     gbt.load_bottles()
#     for bottle in gbt.bottles:
#         stats = obspy.core.trace.Stats()
#         stats.station = station
#         stats.network = network
#         stats.location = bottle.file_metadata['seed_loc']
#         stats.channel = bottle.file_metadata['seed_ch']
#         #(stats.station, channel, year, dayofyear, hour, min) = btl.parse_filename()
#         #metadata = btl.read_header()
#         stats.delta = bottle.file_metadata['interval']
#         stats.npts = bottle.file_metadata['num_pts']
#         stats.starttime += timedelta(seconds=bottle.file_metadata['start'])
#
#         data = np.array(bottle.read_data(), dtype=np.int32)
#         tr = obspy.core.trace.Trace(data=data, header=stats)
#         st.append(tr)
#     st.merge()
#     logger.info(st)
#     gbt.delete_bottles_from_disk()
#     return st

if __name__ == '__main__':

    #function for running local tests
    t1 = datetime.now()

    #filename = 'B001.2022001Day.tgz'  #24 hr Day session (archive and logger format)
    #session = "Day"
    #filename = 'B001.2022001_01.tar' #24 Hour, Hour Session (archive format)
    #session = "Hour_Archive"
    #filename = 'B001.2022001_20.tar' #24 Hour, Min session (archive format)
    #session = "Min_Archive"
    filename = 'B0012200100.tgz'  #1 Hour, Hour Session (logger format)
    session = "Hour"
    #filename = 'B0012200100_20.tar' #1 Hour, Min Session (logger format)
    #session = "Min"
    network = "PB"
    station = "B001"
    edid = find_session_edid(network, station, session)
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    #uri = f"s3://tiledb-strain/{edid}.tdb"
    uri = f"arrays/{edid}.tdb"
    array = StrainTiledbArray(uri)
    to_tiledb(gbt, array)
    t2 = datetime.now()
    elapsed_time = t2 - t1
    logger.info(f'{gbt.file_metadata["filename"]}: Elapsed time {elapsed_time} seconds')
    #print(gbt.file_metadata['filename'], gbt.file_metadata['fcid'], gbt.file_metadata['session'])