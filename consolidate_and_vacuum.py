# script to clean up arrays after writing before syncing to s3
import sys, os

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

    array.consolidate_array_meta()
    array.vacuum_array_meta()
    array.consolidate_fragment_meta()
    array.vacuum_fragment_meta()
    array.consolidate_fragments()
    array.vacuum_fragments()






