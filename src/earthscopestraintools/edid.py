# look up edid without dependecy on es_datasources_client
import os, sys
import requests
import json
import logging

logger = logging.getLogger(__name__)

API_BASE_URL = "https://datasources-api.dev.earthscope.org/"
STATION_EDID_PATH = os.path.join(API_BASE_URL, "station/find")
SESSION_EDID_PATH = os.path.join(API_BASE_URL, "session/find")


def get_station_edid(network_name: str, fcid: str):
    """ Returns a station edid """

    parameters = {
        "network_name": f"FDSN-{network_name}",
        "name": f"FDSN-{fcid}",
        "with_parents": False,
        "name_to_id_map": False,
    }
    try:
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {fcid} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {fcid} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {fcid} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {fcid} {err}")
        raise
    if len(r.json()):
        return r.json()[0]["edid"]
    else:
        return None


def get_session_edid(
    network_name: str,
    fcid: str,
    session: str,
    with_parents: bool = False,
    name_to_id_map: bool = False,
):
    """ Returns a session edid """

    parameters = {
        "network_name": f"FDSN-{network_name}",
        "station_name": f"FDSN-{fcid}",
        "name": f"DATAFLOW-{session}",
        "with_parents": with_parents,
        "name_to_id_map": name_to_id_map,
    }
    try:
        r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {fcid} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {fcid} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {fcid} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {fcid} {err}")
        raise
    if len(r.json()):
        return r.json()[0]["edid"]
    else:
        return None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    network_name = sys.argv[1]
    fcid = sys.argv[2]
    if len(sys.argv) == 3:
        edid = get_station_edid(network_name, fcid)
        logger.info(f"{network_name} {fcid}: {edid}")
    elif len(sys.argv) == 4:
        session = sys.argv[3]
        edid = get_session_edid(network_name, fcid, session)
        logger.info(f"{network_name} {fcid} {session}: {edid}")


# import sys
# from es_datasources_client import Client, api
# import httpx
# import logging
#
# logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s %(levelname)s: %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S',
# )
# logger = logging.getLogger(__name__)
#
# client = Client(base_url="https://datasources-api.dev.earthscope.org")
#
# def find_station_edid(network, station):
#     try:
#         r = api.station.sync.find_stations(
#             client=client, network_name=f'FDSN-{network}', name=f'FDSN-{station}'
#         )
#         return r[0].edid
#     except httpx.ConnectError:
#         logger.error(f"Unable to connect to datasources-id api")
#         exit(1)
#     except KeyError:
#         logger.error(f"Unable to find a valid edid for {network} {station}")
#         exit(1)
#     except Exception as e:
#         logger.exception(f"Unable to load edid for {network} {station}")
#         exit(1)
#
# def find_session_edid(network, station, session):
#     try:
#         r = api.session.sync.find_sessions(
#             client=client, network_name=f'FDSN-{network}', station_name=f'FDSN-{station}', name=f'DATAFLOW-{session}'
#         )
#         return r[0].edid
#     except httpx.ConnectError:
#         logger.error(f"Unable to connect to datasources-id api")
#         exit(1)
#     except KeyError:
#         logger.error(f"Unable to find a valid edid for {network} {station}")
#         exit(1)
#     except Exception as e:
#         logger.exception(f"Unable to load edid for {network} {station} {session}")
#         exit(1)
#
# if __name__ == '__main__':
#     network = sys.argv[1]
#     station = sys.argv[2]
#     if len(sys.argv) == 3:
#         logger.info(find_station_edid(network, station))
#     elif len(sys.argv) == 4:
#         session = sys.argv[3]
#         logger.info(find_session_edid(network, station, session))
