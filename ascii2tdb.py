# ETL from ascii level2 files to tiledb
# write tiledb arrays locally, then aws s3 cp

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
workdir = "arrays"

def find_station_edid(network, station):
    client = Client(base_url="https://datasources-api.dev.earthscope.org")
    r = api.station.sync.find_stations(
        client=client, network_name=network, name=station, name_to_id_map=True
    )
    return r.additional_properties[network].additional_properties[station]

def write_df_to_tiledb(df, array):
    # print(df)
    mode = "append"
    tiledb.from_pandas(uri=array.uri,
                       dataframe=df,
                       index_dims=['data_type', 'timeseries', 'time'],
                       mode=mode,
                       ctx=array.ctx
                       )

    # update the string dimension metadata
    data_type = df['data_type'].unique()
    timeseries = df['timeseries'].unique()
    if type(data_type) == str:
        data_type = [data_type]
    if type(timeseries) == str:
        timeseries = [timeseries]
    with tiledb.open(array.uri, 'r', ctx=array.ctx) as A:
        try:
            dimension_json = A.meta["dimensions"]
        except KeyError:
            dimension_json = '{"data_types":[], "timeseries":[]}'

        dimension_dict = json.loads(dimension_json)
        # print(dimension_dict)
        for item in data_type:
            if item not in dimension_dict['data_types']:
                dimension_dict['data_types'].append(item)
        for item in timeseries:
            if item not in dimension_dict['timeseries']:
                dimension_dict['timeseries'].append(item)

        # print(dimension_dict)
        with tiledb.open(array.uri, 'w', ctx=array.ctx) as A:
            A.meta["dimensions"] = json.dumps(dimension_dict)

def loop_through_ts(filebase, file, array):
    df = pd.read_csv(filebase + '/' + file, delimiter='\s+')
    label = df['strain'][0] + "(mstrain)"
    df = df.rename(
        columns={"strain": "data_type", label: "microstrain", "s_offset": "offset_c", "MJD": "mjd", "date": "time"})
    df['data_type'] = df['data_type'].str.replace('gauge', 'CH')
    # display(df)
    cols = ['data_type', 'timeseries', 'time', 'data', 'quality', 'level', 'version']

    timeseries = "microstrain"
    tmp_df = df[['data_type', 'time', timeseries, 'strain_quality', 'level', 'version']]
    tmp_df = tmp_df.rename(columns={timeseries: "data", "strain_quality": "quality"})
    tmp_df['timeseries'] = timeseries
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    write_df_to_tiledb(tmp_df, array)

    timeseries = 'offset_c'
    tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df['timeseries'] = timeseries
    tmp_df['quality'] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    write_df_to_tiledb(tmp_df, array)

    timeseries = 'tide_c'
    tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df['timeseries'] = timeseries
    tmp_df['quality'] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    write_df_to_tiledb(tmp_df, array)

    timeseries = 'detrend_c'
    tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
    tmp_df = tmp_df.rename(columns={timeseries: "data"})
    tmp_df['timeseries'] = timeseries
    tmp_df['quality'] = " "
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    write_df_to_tiledb(tmp_df, array)

    timeseries = 'atmp_c'
    tmp_df = df[['data_type', 'time', timeseries, 'atmp_c_quality', 'level', 'version']]
    tmp_df = tmp_df.rename(columns={timeseries: "data", 'atmp_c_quality': 'quality'})
    tmp_df['timeseries'] = timeseries
    tmp_df = tmp_df[cols]
    logger.info(f"{timeseries}: {len(tmp_df)} samples")
    write_df_to_tiledb(tmp_df, array)

    if df['data_type'][0] == 'CH0':
        timeseries = 'atmp'
        tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
        tmp_df = tmp_df.assign(data_type='pressure')
        tmp_df = tmp_df.rename(columns={timeseries: "data"})
        tmp_df['timeseries'] = timeseries
        tmp_df['quality'] = " "
        tmp_df = tmp_df[cols]
        logger.info(f"{timeseries}: {len(tmp_df)} samples")
        write_df_to_tiledb(tmp_df, array)

        timeseries = 'doy'
        # do not apply scale factor, already an int
        tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
        tmp_df = tmp_df.assign(data_type='time_index')
        tmp_df = tmp_df.rename(columns={timeseries: "data"})
        tmp_df['timeseries'] = timeseries
        tmp_df['quality'] = " "
        tmp_df = tmp_df[cols]
        logger.info(f"{timeseries}: {len(tmp_df)} samples")
        write_df_to_tiledb(tmp_df, array)

        timeseries = 'mjd'
        tmp_df = df[['data_type', 'time', timeseries, 'level', 'version']]
        tmp_df = tmp_df.assign(data_type='time_index')
        tmp_df = tmp_df.rename(columns={timeseries: "data"})
        tmp_df['timeseries'] = timeseries
        tmp_df['quality'] = " "
        tmp_df = tmp_df[cols]
        logger.info(f"{timeseries}: {len(tmp_df)} samples")
        write_df_to_tiledb(tmp_df, array)

def etl_yearly_ascii_file(fcid, array, year):
    filebase = fcid + "." + year + ".bsm.level2"
    url = "http://bsm.unavco.org/bsm/level2/" + fcid + "/" + filebase + ".tar"
    response = requests.get(url, stream=True)
    tar = tarfile.open(fileobj=BytesIO(response.raw.read()), mode='r')
    tar.extractall()
    files = os.listdir(filebase)
    for file in files:
        logger.info(file)
        loop_through_ts(filebase, file, array)
    #shutil.rmtree(filebase)


if __name__ == '__main__':
    logger = logging.getLogger()
    fcid = "B018"
    net = "PB"
    edid = find_station_edid(net, fcid)
    os.makedirs(workdir, exist_ok=True)
    uri = f"{workdir}/{edid}_level2.tdb"
    logger.info(f"Array uri: {uri}")
    array = StrainTiledbArray(uri, period=300, location='local')

    # delete any existing array at that uri
    array.delete()

    # create new array.  array_exists only works locally not in s3.
    if not tiledb.array_exists(array.uri):
        array.create()

    # years = ["2006","2007","2008","2009",
    #         "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
    #         "2020","2021","2022"]

    years = ["2022"]
    for year in years:
       etl_yearly_ascii_file(fcid, array, year)

    logger.info("Consolidating meta")
    array.consolidate_meta()
    logger.info("Vacuuming meta")
    array.vacuum_meta()
    logger.info("Consolidating fragments")
    array.consolidate_fragments()
    logger.info("Vacuuming fragments")
    array.vacuum_fragments()