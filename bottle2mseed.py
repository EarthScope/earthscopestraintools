from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import obspy
from bottletar import GtsmBottleTar
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def build_stream(network: str,
                   station: str,
                   gbt: GtsmBottleTar):
    st = obspy.core.stream.Stream()
    gbt.load_bottles()
    for bottle in gbt.bottles:
        stats = obspy.core.trace.Stats()
        stats.station = station
        stats.network = network
        stats.location = bottle.file_metadata['seed_loc']
        stats.channel = bottle.file_metadata['seed_ch']
        #(stats.station, channel, year, dayofyear, hour, min) = btl.parse_filename()
        #metadata = btl.read_header()
        stats.delta = bottle.file_metadata['interval']
        stats.npts = bottle.file_metadata['num_pts']
        stats.starttime += timedelta(seconds=bottle.file_metadata['start'])

        data = np.array(bottle.read_data(), dtype=np.int32)
        tr = obspy.core.trace.Trace(data=data, header=stats)
        st.append(tr)
    st.merge()
    logger.info(st)
    gbt.delete_bottles_from_disk()
    return st

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
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    st = build_stream(network, station, gbt)
    miniseed_filename = f"miniseed/{gbt.file_metadata['filebase']}.ms"
    st.write(miniseed_filename, format="MSEED")
    t2 = datetime.now()
    elapsed_time = t2 - t1
    logger.info(f'{gbt.file_metadata["filename"]}: Elapsed time {elapsed_time} seconds')
    #print(gbt.file_metadata['filename'], gbt.file_metadata['fcid'], gbt.file_metadata['session'])