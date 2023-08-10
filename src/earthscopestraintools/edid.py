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
):
    """Returns a single network edid"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
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
        return r.json()['items'][0]
    else:
        return None


def get_station_edid(
    station: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
):
    """Returns a station edid in the BNUM station namespace"""

    parameters = {
        # "network_name": f"FDSN:{network_name}",
        "station_name": f"BNUM:{station}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
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
        return r.json()['items'][0] 
    else:
        return None


def get_session_edid(
    station: str,
    session: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
):
    """Returns a session edid in the BNUM station namespace"""

    parameters = {
        "station_name": f"BNUM:{station}",
        "session_name": f"DATAFLOW:{session}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
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
        return r.json()['items'][0]  # ["edid"]
    else:
        return None


def get_all_stations_in_network(
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=False,
    with_parents=False,
    with_slugs=False,
    limit=50
):
    """Returns a dictionary of station:edid for a given namespace:network"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "limit": limit
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
        stations = {}
        more_pages = True
        while more_pages:
            more_pages = r.json()['has_next']
            #print(r.url)
            #print(r.json())
            for item in r.json()['items']:
                for name in item['names']:
                    if namespace == "BSM":
                        if "BNUM" in name:
                            stations[name.split(':')[-1].split("_")[-1]] = item['edid']
                    elif namespace == "FDSN":
                        if "FDSN" in name:
                            stations[name.split(':')[-1].split("_")[-1]] = item['edid']
            parameters['offset']=r.json()['offset'] + r.json()['limit']
            r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
            r.raise_for_status()
        return dict(sorted(stations.items()))
    else:
        return None
    
def get_all_sessions_in_network(
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=False,
    with_parents=True,
    with_slugs=False,
    limit=1000
):
    """Returns a nested dictionary of each session and its parent station/network for a given namespace:network"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "limit": limit
    }
    try:
        stations = {}
        #sessions = {}
        more_pages = True
        while more_pages:
            r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=10)
            #print(r.url)
            r.raise_for_status()
            more_pages = r.json()['has_next']
            if more_pages:
                logger.error(f'get_all_sessions_in_network request exceeded {limit} responses, returned only partial results')
            #print(r.json()['has_next'])
            for item in r.json()['items']:
                session_edid = item['edid']
                session_name = item['names'][0].split(':')[-1]
                for name in item['station']['names']:
                    if namespace == "BSM":
                        if "BNUM" in name:
                            station_name = name.split(':')[-1]
                            station_edid = item['station']['edid']
                            if station_name not in stations:
                                stations[station_name] = {"edid": station_edid}
                    elif namespace == "FDSN":
                        if "FDSN" in name:
                            station_name = name.split(':')[-1].split("_")[-1]
                            station_edid = item['station']['edid']
                            if station_name not in stations:
                                stations[station_name] = {"edid": station_edid}
                stations[station_name][session_name] = session_edid
            parameters['offset']=r.json()['offset'] + r.json()['limit']
            
        #print(stations)
        return dict(sorted(stations.items()))
        # r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=10)
        # # print(r.url)
        # r.raise_for_status()
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
        
def get_bnum_list(namespace="BSM", network="NOTA"):
    response = get_all_stations_in_network(namespace=namespace, network=network)
    try:
        return sorted(list(response.keys()))
    except:
        logger.error("No stations found")
        return []


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
        networks = r.json()['items'][0]["networks"]
        for network in networks:
            for name in network["names"]:
                if "FDSN" in name:
                    fdsn_name = name.split(":")[-1]
                    return fdsn_name
        return None
    else:
        return None
