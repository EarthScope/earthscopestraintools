import sys
from es_datasources_client import Client, api
import httpx
import logging

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

client = Client(base_url="https://datasources-api.dev.earthscope.org")

def find_station_edid(network, station):
    try:
        r = api.station.sync.find_stations(
            client=client, network_name=f'FDSN-{network}', name=f'FDSN-{station}'
        )
        return r[0].edid
    except httpx.ConnectError:
        logger.error(f"Unable to connect to datasources-id api")
        exit(1)
    except KeyError:
        logger.error(f"Unable to find a valid edid for {network} {station}")
        exit(1)
    except Exception as e:
        logger.exception(f"Unable to load edid for {network} {station}")
        exit(1)

def find_session_edid(network, station, session):
    try:
        r = api.session.sync.find_sessions(
            client=client, network_name=f'FDSN-{network}', station_name=f'FDSN-{station}', name=f'DATAFLOW-{session}'
        )
        return r[0].edid
    except httpx.ConnectError:
        logger.error(f"Unable to connect to datasources-id api")
        exit(1)
    except KeyError:
        logger.error(f"Unable to find a valid edid for {network} {station}")
        exit(1)
    except Exception as e:
        logger.exception(f"Unable to load edid for {network} {station} {session}")
        exit(1)

if __name__ == '__main__':
    network = sys.argv[1]
    station = sys.argv[2]
    if len(sys.argv) == 3:
        logger.info(find_station_edid(network, station))
    elif len(sys.argv) == 4:
        session = sys.argv[3]
        logger.info(find_session_edid(network, station, session))