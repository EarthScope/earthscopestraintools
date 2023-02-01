# script to clean up arrays after writing before syncing to s3

import tarfile
import tiledb
import pandas as pd
import sys, os
import shutil
import boto3
import json
from io import BytesIO
import requests
import configparser

from straintiledbarray import StrainTiledbArray
from es_datasources_client import Client, api

import logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger()
workdir = "arrays"

def find_station_edid(network, station):
    client = Client(base_url="https://datasources-api.dev.earthscope.org")
    r = api.station.sync.find_stations(
        client=client, network_name=f'FDSN-{network}', name=f'FDSN-{station}', name_to_id_map=True
    )
    #print(r)
    return r.additional_properties[f'FDSN-{network}'].additional_properties[f'FDSN-{station}']


if __name__ == '__main__':

    network = sys.argv[1]
    fcid = sys.argv[2]
    edid = find_station_edid(network, fcid)
    os.makedirs(workdir, exist_ok=True)
    uri = f"{workdir}/{edid}_level2.tdb"
    logger.info(f"Array uri: {uri}")
    array = StrainTiledbArray(uri, period=300, location='local')

    logger.info("Consolidating meta")
    array.consolidate_meta()
    logger.info("Vacuuming meta")
    array.vacuum_meta()
    # logger.info("Consolidating fragments")
    # array.consolidate_fragments()
    # logger.info("Vacuuming fragments")
    # array.vacuum_fragments()




