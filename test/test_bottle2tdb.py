from earthscopestraintools.bottle2tdb import bottle2tdb
from earthscopestraintools.edid import get_station_edid, get_session_edid
from earthscopestraintools.tiledbtools import RawStrainReader
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

if __name__ == '__main__':

    t1 = datetime.now()

    network = "PB"
    station = "B001"
    station_edid = get_station_edid(station)
    ancillary_uri = f"arrays/{station_edid}_ancillary.tdb"
    write_it = True
    print_it = True
    check_it = True


    filepath = 'bottles/B001.2022001Day.tgz'  #24 hr Day session (archive and logger format)
    session = "Day"
    session_edid = get_session_edid(station, session)
    strain_uri = f"arrays/{session_edid}.tdb"
    bottle2tdb(filepath, strain_uri, ancillary_uri, session, write_it=write_it, print_it=print_it, check_it=check_it)

    filepath = 'bottles/B0012200100.tgz'  # 1 Hour, Hour Session (logger format)
    session = "Hour"
    session_edid = get_session_edid(station, session)
    strain_uri = f"arrays/{session_edid}.tdb"
    bottle2tdb(filepath, strain_uri, ancillary_uri, session, write_it=write_it, print_it=print_it, check_it=check_it)

    filepath = 'bottles/B0012200100_20.tar'  # 1 Hour, Min Session (logger format)
    session = "Min"
    session_edid = get_session_edid(station, session)
    strain_uri = f"arrays/{session_edid}.tdb"
    bottle2tdb(filepath, strain_uri, ancillary_uri, session, write_it=write_it, print_it=print_it, check_it=check_it)

    # filepath = 'bottles/B001.2022001_01.tar' #24 Hour, Hour Session (archive format)
    # session = "Hour"
    # session_edid = get_session_edid(network, station, session)
    # strain_uri = f"arrays/{session_edid}.tdb"
    # bottle2tdb(filepath, strain_uri, ancillary_uri, session, write_it=write_it, print_it=print_it, check_it=check_it)

    # filepath = 'bottles/B001.2022001_20.tar' #24 Hour, Min session (archive format)
    # session = "Min"
    # session_edid = get_session_edid(network, station, session)
    # strain_uri = f"arrays/{session_edid}.tdb"
    # bottle2tdb(filepath, strain_uri, ancillary_uri, session, write_it=write_it, print_it=print_it, check_it=check_it)

    t2 = datetime.now()
    elapsed_time = t2 - t1
    logger.info(f'test_bottle2tdb.py: Elapsed time {elapsed_time} seconds')