# look up edids without dependecy on es_datasources_client

import os, sys
import requests
import json
import logging

logger = logging.getLogger(__name__)

API_BASE_URL = "https://datasources-api.dev.earthscope.org/"

NETWORK_EDID_PATH = os.path.join(API_BASE_URL, "network")
STATION_EDID_PATH = os.path.join(API_BASE_URL, "station")
SESSION_EDID_PATH = os.path.join(API_BASE_URL, "session")


def get_network_edid(
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    id_map=False,
    name_map=False,
):
    """Returns a single network edid"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "id_map": id_map,
        "name_map": name_map,
    }
    try:
        r = requests.get(NETWORK_EDID_PATH, params=parameters, timeout=10)
        # print(r.url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {network} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {network} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {network} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {network} {err}")
        raise
    if len(r.json()):
        return r.json()[0]
    else:
        return None


def get_station_edid(
    station: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    id_map=False,
    name_map=False,
):
    """Returns a station edid in the BNUM station namespace"""

    parameters = {
        # "network_name": f"FDSN:{network_name}",
        "station_name": f"BNUM:{station}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "id_map": id_map,
        "name_map": name_map,
    }
    try:
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
        # print(r.url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {station} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {station} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {station} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {station} {err}")
        raise
    if len(r.json()):
        return r.json()[0]  # ["edid"]
    else:
        return None


def get_session_edid(
    station: str,
    session: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    id_map=False,
    name_map=False,
):
    """Returns a session edid in the BNUM station namespace"""

    parameters = {
        "station_name": f"BNUM:{station}",
        "session_name": f"DATAFLOW:{session}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "id_map": id_map,
        "name_map": name_map,
    }
    try:
        r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {station} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {station} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {station} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {station} {err}")
        raise
    if len(r.json()):
        return r.json()[0]  # ["edid"]
    else:
        return None


def get_network_edids(
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=False,
    with_parents=False,
    with_slugs=False,
    id_map=False,
    name_map=True,
):
    """Returns a dictionary of station:edid for a given namespace:network"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "id_map": id_map,
        "name_map": name_map,
    }
    try:
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
        # print(r.url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {network} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {network} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {network} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {network} {err}")
        raise
    if len(r.json()):
        if namespace == "FDSN":
            dict1 = {k: v for k, v in r.json().items() if f"{namespace}:{network}" in k}
            dict2 = {
                k.split(":")[-1].split("_")[-1]: v
                for k, v in dict1[f"{namespace}:{network}"].items()
                if namespace in k
            }
            return dict2
        elif namespace == "BSM":
            dict1 = {k: v for k, v in r.json().items() if f"{namespace}:{network}" in k}
            dict2 = {
                k.split(":")[-1]: v
                for k, v in dict1[f"{namespace}:{network}"].items()
                if "BNUM" in k
            }
            return dict2
    else:
        return None

def get_network_name(station: str):

    parameters = {
    "station_name": f"BNUM:{station}",
    "with_parents": True,
    }
    try:
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
        # print(r.url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f'Http error: {station} {json.loads(r.content)["detail"]}')
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {station} {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {station} {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Oops: Something Else: {station} {err}")
        raise
    if len(r.json()):
        networks = r.json()[0]['networks']
        for network in networks:
            for name in network['names']:
                if 'FDSN' in name:
                    fdsn_name = name.split(':')[-1]
                    return fdsn_name
        return None
    else:
        return None