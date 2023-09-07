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

def build_stream(network: str, station: str, gbt: GtsmBottleTar, verbose=False, print_traces: bool = True):
    st = obspy.core.stream.Stream()
    try:
        gbt.load_bottles(verbose=verbose)
        try:
            for bottle in gbt.bottles:
                stats = obspy.core.trace.Stats()
                stats.station = station
                stats.network = network
                stats.location = bottle.file_metadata["seed_loc"]
                stats.channel = bottle.file_metadata["seed_ch"]
                stats.delta = bottle.file_metadata["interval"]
                bottle_num_pts = bottle.file_metadata["num_pts"]
                bottle_starttime = stats.starttime + timedelta(seconds=bottle.file_metadata["start"])
                #print(bottle.file_metadata["filename"])
                if bottle.file_metadata['seed_scale_factor'] == 10000:
                    #print("float", bottle.file_metadata['data_type'])
                    float_data = np.array(bottle.read_data(), dtype=np.float32) * bottle.file_metadata['seed_scale_factor']
                    int_data = np.array(bottle.read_data(), dtype=np.int32)
                    gap_indicies = np.where(int_data==999999)[0] 
                    np.put(float_data, gap_indicies, 999999)
                    data = np.int32(float_data)

                else:
                    #print("int", bottle.file_metadata['data_type'])
                    data = np.array(bottle.read_data(), dtype=np.int32)
                    gap_indicies = np.where(data==999999)[0] 
                
                
                #print(data)
                #print(f"{len(gap_indicies)} fill values at {gap_indicies}")
                start_indicies = []
                num_gaps = len(gap_indicies)
                if num_gaps:
                    for i, gap_index in enumerate(gap_indicies):
                        if i +1 < num_gaps:
                            if gap_index + 1 != gap_indicies[i+1]:
                                start_indicies.append(gap_index + 1)
                        else:
                            start_indicies.append(gap_index + 1)
                #print(f"{len(start_indicies)} start indicies at {start_indicies}")

                gap_times = {}
                for index in gap_indicies:
                    gap_times[index] = bottle_starttime + timedelta(seconds=index*stats.delta)
                #print(gap_times)

                start_times = [bottle_starttime]
                for index in start_indicies:
                    start_times.append(bottle_starttime + timedelta(seconds=index*stats.delta))
                #print(start_times)

                arrays = np.split(data, start_indicies)
                for i, array in enumerate(arrays):
                    tracedata = np.delete(array, np.where(array==999999)[0])
                    #print(i, tracedata)
                    stats.npts = len(tracedata)
                    if stats.npts > 0:
                        stats.starttime = start_times[i]
                        tr = obspy.core.trace.Trace(data=tracedata, header=stats)
                        st.append(tr)
            if print_traces:
                print(st.__str__(extended=True))
            return st
        except Exception as e:
            logger.error(f"{gbt.file_metadata['filename']}: {type(e)} unable to build stream from bottles")
            gbt.delete_bottles_from_disk()
            raise
    except Exception as e:
        logger.error(f"{gbt.file_metadata['filename']}: {type(e)} unable to load bottles")
        gbt.delete_bottles_from_disk()
        raise
    # gbt.delete_bottles_from_disk()
    # return None


# def build_stream(network: str, station: str, gbt: GtsmBottleTar):
#     st = obspy.core.stream.Stream()
#     gbt.load_bottles()
#     for bottle in gbt.bottles:
#         stats = obspy.core.trace.Stats()
#         stats.station = station
#         stats.network = network
#         stats.location = bottle.file_metadata["seed_loc"]
#         stats.channel = bottle.file_metadata["seed_ch"]
#         # (stats.station, channel, year, dayofyear, hour, min) = btl.parse_filename()
#         # metadata = btl.read_header()
#         stats.delta = bottle.file_metadata["interval"]
#         stats.npts = bottle.file_metadata["num_pts"]
#         stats.starttime += timedelta(seconds=bottle.file_metadata["start"])

#         data = np.array(bottle.read_data(), dtype=np.int32)
#         tr = obspy.core.trace.Trace(data=data, header=stats)
#         st.append(tr)
#     st.merge()
#     logger.info(st.__str__(extended=True))
#     gbt.delete_bottles_from_disk()
#     return st


def bottle2mseed(network, station, filename, session, verbose=False, print_traces=True, plot_traces=False):
    gbt = GtsmBottleTar(f"bottles/{filename}", session, verbose=verbose)
    try:
        st = build_stream(network, station, gbt, verbose=verbose, print_traces=print_traces)
        if plot_traces:
            st.plot()
        miniseed_filename = f"miniseed/{gbt.file_metadata['filebase']}.ms"
        st.write(miniseed_filename, format="MSEED", reclen=512)
        return st
    except Exception as e:
        logger.error(f"{gbt.file_metadata['filename']}: unable to translate to miniseed \n{e}")
