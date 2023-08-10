
from earthscopestraintools.edid import get_network_edid, get_station_edid, get_session_edid 
from earthscopestraintools.edid import get_network_name, get_all_stations_in_network, get_all_sessions_in_network, get_bnum_list
import logging
logger = logging.getLogger()
logging.basicConfig(
        format="%(message)s", level=logging.INFO
    )

if __name__ == '__main__':
    namespace = "FDSN"
    network = "PB"

    station_edids = get_all_stations_in_network(namespace, network)
    for station in station_edids:
        logger.info(f"{network} {station}: {station_edids[station]}")

    
    session_edids = get_all_sessions_in_network(namespace, network)
    for station in session_edids:
        logger.info(f"{network} {station}: {session_edids[station]}")
    
    station = "B001"
    network_edid = get_network_edid(namespace, network)
    logger.info(f"{network}: {network_edid}")

    station_edid = get_station_edid(station)
    logger.info(f"{station}: {station_edid}")
    session = "Day"
    session_edid = get_session_edid(station, session)
    logger.info(f"{station}.{session}: {session_edid}")
    session = "Hour"
    session_edid = get_session_edid(station, session)
    logger.info(f"{station}.{session}: {session_edid}")
    session = "Min"
    session_edid = get_session_edid(station, session)
    logger.info(f"{station}.{session}: {session_edid}")

    fdsn_network = get_network_name(station)
    logger.info(f"FDSN network name: {fdsn_network}")

    bnums = get_bnum_list(namespace="BSM", network="NOTA")
    logger.info(f"BSM:NOTA:{bnums}")