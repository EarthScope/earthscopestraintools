# write tarred tiledb arrays from tiledb arrays
# slice by start and end time
# use in lambda with web service to deliver data in tiledb

import tiledb
import numpy as np
import pandas as pd
# import datetime
# import sys, os
import shutil
import json

from straintiledbarray import StrainTiledbArray
from es_datasources_client import Client, api

import logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger()

def find_station_edid(network, station):
    client = Client(base_url="https://datasources-api.dev.earthscope.org")
    r = api.station.sync.find_stations(
        client=client, network_name=f'FDSN-{network}', name=f'FDSN-{station}', name_to_id_map=True
    )
    #print(r)
    return r.additional_properties[f'FDSN-{network}'].additional_properties[f'FDSN-{station}']

def to_date(datetime64):
    ts = pd.to_datetime(str(datetime64))
    d = ts.strftime('%Y%m%d')
    return d

def read_date_range(array, start, end):
    with tiledb.open(array.uri, 'r', ctx=array.ctx) as A:
        dims = json.loads(A.meta['dimensions'])
        data_types = dims['data_types']
        timeseries_list = dims['timeseries']
        index_col = ['data_type', 'timeseries', 'time']
        attrs = ['data', 'quality', 'level', 'version']
        df = A.query(index_col=index_col, attrs=attrs).df[:, :, start:end]#.sort_index()
    return df, data_types, timeseries_list

def write_new_tdb(df, array):
    data_types = list(df.index.get_level_values(0).unique())
    timeseries_list = list(df.index.get_level_values(1).unique())
    dimension_dict = {"data_types":data_types, "timeseries":timeseries_list}
    array.delete()
    array.create()
    with tiledb.open(array.uri, 'w', ctx=array.ctx) as A:
        A.meta["dimensions"] = json.dumps(dimension_dict)
    tiledb.from_pandas(uri=array.uri,
                       dataframe=df,
                       index_dims=['data_type', 'timeseries', 'time'],
                       mode='append',
                       ctx=array.ctx
                       )
    shutil.make_archive(array.uri, 'zip', array.uri)
    shutil.rmtree(array.uri)
    logger.info("Export complete")

if __name__ == '__main__':
    fcid = "B007"
    net = "PB"
    start = np.datetime64("2019-01-01T00:00:00")
    end = np.datetime64("2021-01-01T00:00:00")

    edid = find_station_edid(net, fcid)
    workdir = "arrays"
    uri = f"{workdir}/{edid}_level2.tdb"
    logger.info(f"Array uri: {uri}")
    array = StrainTiledbArray(uri, period=300, location='local')

    df, data_types, timeseries_list = read_date_range(array, start, end)
    #logger.info(df)

    uri2 = f"{workdir}/{net}_{fcid}_level2_{start}.tdb"
    logger.info(f"Array uri: {uri2}")
    array2 = StrainTiledbArray(uri2, period=300, location='local')
    write_new_tdb(df, array2)
