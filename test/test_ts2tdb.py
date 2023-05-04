from earthscopestraintools.tiledbtools import ProcessedStrainWriter, ProcessedStrainReader, RawStrainReader, RawStrainWriter
from earthscopestraintools.timeseries import Timeseries
from earthscopestraintools.edid import get_station_edid
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(message)s", level=logging.INFO
    )
    
if __name__ == '__main__':
    workdir = "arrays"
    network = 'PB'
    station = "B001"
    year = "2022"
    #station_edid = get_station_edid(station)
    #uri = f"{workdir}/{station_edid}_level2.tdb"
    uri = f"{workdir}/01GVDXYEEKY7T5XCCVX2BM51EM_level2.tdb"
    reader = ProcessedStrainReader(uri)
    start_str = f"{year}-01-01T00:00:00"
    end_str = f"{int(year) + 1}-01-01T00:00:00"
    ts = reader.to_ts(
        data_types=["CH0", "CH1", "CH2", "CH3"],#, "Eee+Enn", "Eee-Enn", "2Ene"],
        timeseries="microstrain",
        attrs=['data','quality','level','version'],
        start_str=start_str,
        end_str=end_str,
        units="microstrain",
        name=f"from {uri}"
    )
    ts.stats()
    ts2 = ts.remove_999999s()
    ts2.stats()
    uri2 = "arrays/test_write.tdb"
    writer = ProcessedStrainWriter(uri=uri2)
    writer.array.delete()
    writer.create()
    writer.ts_2_tiledb(ts2)

    reader2 = ProcessedStrainReader(uri2)
    start_str = f"{year}-01-01T00:00:00"
    end_str = f"{int(year) + 1}-01-01T00:00:00"
    ts3 = reader.to_ts(
        data_types=["CH0", "CH1", "CH2", "CH3"],  # , "Eee+Enn", "Eee-Enn", "2Ene"],
        timeseries="microstrain",
        attrs=['data', 'quality', 'level', 'version'],
        start_str=start_str,
        end_str=end_str,
        units="microstrain",
        name=f"from {uri2}"
    )
    ts3.stats()