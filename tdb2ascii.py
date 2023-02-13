# write ascii files from tiledb arrays
# slice by start and end time

import tarfile
import tiledb
import numpy as np
import pandas as pd
import os
import json
import datetime

from straintiledbarray import StrainTiledbArray
from edid import find_station_edid

import logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

workdir = "arrays"
outputdir = "ascii_output"

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
        start_ts = start.astype('int')*1000
        end_ts = end.astype('int')*1000
        df = A.query(index_col=index_col, attrs=attrs).df[:, :, start_ts:end_ts].sort_index()
        # convert unix ms to pd.Timestamp
        df.index = df.index.set_levels(pd.to_datetime(df.index.levels[2], unit='ms'), level=2)
    return df, data_types, timeseries_list

def write_ascii(df, fcid, date_range_string):
    filenames = []
    for data_type in ['CH0', 'CH1', 'CH2', 'CH3', 'Eee+Enn', 'Eee-Enn', '2Ene']:
        logger.info(f"writing ascii file for {data_type}")
        for timeseries in ['microstrain', 'offset_c', 'tide_c', 'trend_c', 'atmp_c']:
            tmp_df = df.loc[data_type, timeseries].rename(columns={'data': timeseries})
            if timeseries == 'microstrain':
                tmp_df = tmp_df.rename(columns={"microstrain": f"{data_type}(mstrain)",
                                                'quality': 'strain_quality'})
                new_df = tmp_df
                new_df['strain'] = data_type
            elif timeseries == 'offset_c':
                new_df["s_offset"] = tmp_df['offset_c']
            elif timeseries == 'trend_c':
                new_df["detrend_c"] = tmp_df['trend_c']
            elif timeseries == 'tide_c':
                new_df["tide_c"] = tmp_df['tide_c']
            elif timeseries == 'atmp_c':
                new_df["atmp_c"] = tmp_df['atmp_c']
                new_df["atmp_c_quality"] = tmp_df['quality']

        new_df['atmp'] = df['data'].loc['pressure', 'atmp']
        new_df['MJD'] = df['data'].loc['time_index', 'mjd']
        new_df['doy'] = df['data'].loc['time_index', 'doy'].astype(int)

        new_df = new_df.reset_index().rename(columns={'time': 'date'})


        ordered_columns = ['strain', 'date', 'doy', 'MJD', f"{data_type}(mstrain)", 's_offset', 'strain_quality',
                           'tide_c', 'detrend_c', 'atmp_c', 'atmp_c_quality', 'level', 'version', 'atmp']
        new_df = new_df[ordered_columns]
        output = f"{outputdir}/{fcid}.{date_range_string}.{data_type}.txt.gz"
        filenames.append(output)
        new_df.to_csv(output, sep='\t', index=False)

    #Create Tar file
    tarfilename = f"{outputdir}/{fcid}.bsm.level2.{date_range_string}.tar"
    logger.info(f"writing to {tarfilename}")
    tar = tarfile.open(tarfilename, 'w')

    for f in filenames:
        tar.add(f, arcname = f.split('/')[-1])

    tar.close()

if __name__ == '__main__':
    fcid = "B005"
    net = "PB"
    start = np.datetime64("2022-01-01T00:00:00")
    end = np.datetime64("2022-02-01T00:00:00")
    date_range_string = f"{to_date(start)}-{to_date(end)}"
    edid = find_station_edid(net, fcid)
    uri = f"{workdir}/{edid}_level2.tdb"
    logger.info(f"Array uri: {uri}")
    array = StrainTiledbArray(uri, period=300, location='local')

    df, data_types, timeseries_list = read_date_range(array, start, end)
    logger.info(df)
    os.makedirs(outputdir, exist_ok=True)
    write_ascii(df, fcid, date_range_string)