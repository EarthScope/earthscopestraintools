# look up edids without dependecy on es_datasources_client

import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

def get_network_edid(
    api_url: str,
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    timeout: int = 10
):
    """Returns a single network edid"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
    }
    NETWORK_EDID_PATH = os.path.join(api_url, "network")
    try:
        # async with httpx.AsyncClient() as client:
        #     r = await client.get(NETWORK_EDID_PATH, 
        #                         params=parameters,
        #                         timeout=10)
        r = requests.get(NETWORK_EDID_PATH, params=parameters, timeout=timeout)
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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise

    if len(r.json()):
        return r.json()["items"][0]
    else:
        logger.warning("No items returned")
        return None


def get_station_edid(
    api_url: str,
    namespace: str,
    station: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    timeout: int = 10
):
    """Returns a station edid in the specified station namespace"""

    parameters = {
        "station_name": f"{namespace}:{station}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
    }
    STATION_EDID_PATH = os.path.join(api_url, "station")
    try:
        # async with httpx.AsyncClient() as client:
        #     r = await client.get(STATION_EDID_PATH, 
        #                         params=parameters,
        #                         timeout=10)
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=timeout)
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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise

    if len(r.json()):
        return r.json()["items"][0]
    else:
        logger.warning("No items returned")
        return None


def get_session_edid(
    api_url: str,
    namespace: str,
    station: str,
    session: str,
    with_children_count=False,
    with_edid_only=True,
    with_parents=False,
    with_slugs=False,
    timeout: int = 10
):
    """Returns a session edid in the specified station namespace"""

    parameters = {
        "station_name": f"{namespace}:{station}",
        "session_name": f"DATAFLOW:{session}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
    }
    SESSION_EDID_PATH = os.path.join(api_url, "session")
    try:
        # async with httpx.AsyncClient() as client:
        #     r = await client.get(SESSION_EDID_PATH, 
        #                         params=parameters,
        #                         timeout=10)
        r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=timeout)
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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise

    if len(r.json()):
        return r.json()["items"][0]
    else:
        logger.warning("No items returned")
        return None


def get_all_stations_in_network(
    api_url: str,
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=False,
    with_parents=False,
    with_slugs=False,
    limit=1000,
    timeout: int = 10
):
    """Returns a dictionary of station:edid for a given namespace:network"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "limit": limit,
    }
    STATION_EDID_PATH = os.path.join(api_url, "station")
    try:
        # async with httpx.AsyncClient() as client:
        #     r = await client.get(STATION_EDID_PATH, 
        #                         params=parameters,
        #                         timeout=10)
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=timeout)
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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise

    if len(r.json()):
        stations = {}
        more_pages = True
        while more_pages:
            more_pages = r.json()["has_next"]

            for item in r.json()["items"]:
                for name in item["names"]:
                    if namespace == "BSM":
                        if "BNUM" in name:
                            stations[name.split(":")[-1]] = item["edid"]
                    elif namespace == "FDSN":
                        if "FDSN" in name:
                            stations[name.split(":")[-1]] = item["edid"]
                    elif namespace == "PERM":
                        if "PNUM" in name:
                            stations[name.split(":")[-1]] = item["edid"]
            parameters["offset"] = r.json()["offset"] + r.json()["limit"]

            r = requests.get(STATION_EDID_PATH, params=parameters, timeout=10)
            r.raise_for_status()

        return dict(sorted(stations.items()))

    else:
        logger.warning("No items returned")
        return None


def get_all_sessions_in_network(
    api_url: str,
    namespace: str,
    network: str,
    with_children_count=False,
    with_edid_only=False,
    with_parents=True,
    with_slugs=False,
    limit=1000,
    timeout: int = 10
):
    """Returns a nested dictionary of each session and its parent station/network for a given namespace:network"""

    parameters = {
        "network_name": f"{namespace}:{network}",
        "with_children_count": with_children_count,
        "with_edid_only": with_edid_only,
        "with_parents": with_parents,
        "with_slugs": with_slugs,
        "limit": limit,
    }
    SESSION_EDID_PATH = os.path.join(api_url, "session")
    try:
        stations = {}
        more_pages = True

        while more_pages:
            # async with httpx.AsyncClient() as client:
            #     r = await client.get(SESSION_EDID_PATH, 
            #                     params=parameters,
            #                     timeout=10)
            r = requests.get(SESSION_EDID_PATH, params=parameters, timeout=timeout)
            #print(r.url)
            r.raise_for_status()
            more_pages = r.json()["has_next"]

            for item in r.json()["items"]:
                session_edid = item["edid"]
                session_name = item["names"][0].split(":")[-1]
                for name in item["station"]["names"]:
                    if namespace == "BSM":
                        if "BNUM" in name:
                            station_name = name.split(":")[-1]
                            station_edid = item["station"]["edid"]
                            if station_name not in stations:
                                stations[station_name] = {"edid": station_edid}

                    elif namespace == "FDSN":
                        if "FDSN" in name:
                            station_name = name.split(":")[-1].split("_")[-1]
                            station_edid = item["station"]["edid"]
                            if station_name not in stations:
                                stations[station_name] = {"edid": station_edid}

                    elif namespace == "PERM":
                        if "PNUM" in name:
                            station_name = name.split(":")[-1]
                            station_edid = item["station"]["edid"]
                            if station_name not in stations:
                                stations[station_name] = {"edid": station_edid}

                stations[station_name][session_name] = session_edid

            if more_pages:
                logger.info(
                    f"get_all_sessions_in_network request exceeded {limit} responses, "
                    f"returned only partial results, grabbing more.."
                )
            else:
                logger.info("Collected all results.")

            parameters["offset"] = r.json()["offset"] + r.json()["limit"]

        return dict(sorted(stations.items()))

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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise


def get_bnum_list(api_url: str, 
                  namespace="BSM", 
                  network="NOTA",
                  timeout: int = 10):
    #returns all bnums with GTSM instruments (explicitly removes MC dilatometers)
    response = get_all_stations_in_network(api_url=api_url, namespace=namespace, network=network, timeout=timeout)
    try:
        bnums = sorted(list(response.keys()))
        mc = ["AIRS", "OLV1", "OLV2", "TRNT"]
        return [x for x in bnums if x not in mc]

    except Exception:
        logger.error("No stations found")
        return []


def lookup_fdsn_network_from_bnum(api_url: str,
                                  station: str,
                                  timeout: int = 10):
    parameters = {
        "station_name": f"BNUM:{station}",
        "with_parents": True,
    }
    STATION_EDID_PATH = os.path.join(api_url, "station")
    try:
        # async with httpx.AsyncClient() as client:
        #     r = await client.get(STATION_EDID_PATH, 
        #                         params=parameters,
        #                         timeout=10)
        r = requests.get(STATION_EDID_PATH, params=parameters, timeout=timeout)
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

    except Exception as err:
        logger.error("Unknown error: {}".format(err))
        raise

    if len(r.json()):
        networks = r.json()["items"][0]["networks"]
        for network in networks:
            for name in network["names"]:
                if "FDSN" in name:
                    fdsn_name = name.split(":")[-1]
                    return fdsn_name
        return None
    else:
        logger.warning("No items returned")
        return None