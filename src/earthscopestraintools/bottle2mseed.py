from datetime import timedelta
import numpy as np
import obspy
from earthscopestraintools.bottletar import GtsmBottleTar
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def build_stream(network: str, station: str, gbt: GtsmBottleTar):
    st = obspy.core.stream.Stream()
    gbt.load_bottles()
    for bottle in gbt.bottles:
        stats = obspy.core.trace.Stats()
        stats.station = station
        stats.network = network
        stats.location = bottle.file_metadata["seed_loc"]
        stats.channel = bottle.file_metadata["seed_ch"]
        # (stats.station, channel, year, dayofyear, hour, min) = btl.parse_filename()
        # metadata = btl.read_header()
        stats.delta = bottle.file_metadata["interval"]
        stats.npts = bottle.file_metadata["num_pts"]
        stats.starttime += timedelta(seconds=bottle.file_metadata["start"])

        data = np.array(bottle.read_data(), dtype=np.int32)
        tr = obspy.core.trace.Trace(data=data, header=stats)
        st.append(tr)
    st.merge()
    logger.info(st.__str__(extended=True))
    gbt.delete_bottles_from_disk()
    return st


def bottle2mseed(network, station, filename, session):
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    st = build_stream(network, station, gbt)
    miniseed_filename = f"miniseed/{gbt.file_metadata['filebase']}.ms"
    st.write(miniseed_filename, format="MSEED")
