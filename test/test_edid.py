from earthscopestraintools.edid import get_network_edid, get_station_edid, get_session_edid, get_network_edids, get_network_name
import requests
import logging
logger = logging.getLogger()
logging.basicConfig(
        format="%(message)s", level=logging.INFO
    )

if __name__ == '__main__':
    namespace = "BSM"
    network = "NOTA"
    station = "B001"
    network_edid = get_network_edids(namespace, network)
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
