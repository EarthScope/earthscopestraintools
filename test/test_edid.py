from earthscopestraintools.edid import get_network_edid, get_station_edid, get_session_edid, get_network_edids

import logging
logger = logging.getLogger()
logging.basicConfig(
        format="%(message)s", level=logging.INFO
    )

if __name__ == '__main__':
    namespace = "BSM"
    network = "NOTA"
    network_edid = get_network_edids(namespace, network)
    logger.info(f"{network}: {network_edid}")

    station = "B001"
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