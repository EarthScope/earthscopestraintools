import os
import dateutil
import requests
import tarfile
import shutil
import pandas as pd
from earthscopestraintools.timeseries import Timeseries
import logging

logger = logging.getLogger(__name__)

def load_l2_ascii(station, start, end, strain_type='gauge', show_stats=False, download=True, remove=True):
    """method to download level 2 ascii strain data for a given date range and read into Timeseries objects.  
    Returns a dictionary of Timeseries keyed as 'microstrain', 'offset_c', 'tide_c', 'trend_c', 'atmp_c', and 'atmp'

    :param station: Four character station id
    :type station: str
    :param start: start time of date range
    :type start: str
    :param end: end time of desired date range
    :type end: str
    :param strain_type: whether to load 'gauge' or 'regional' strains, defaults to 'gauge'
    :type strain_type: str, optional
    :param show_stats: option to print gap analysis during loading, defaults to False
    :type show_stats: bool, optional
    :param download: option to download the ascii files, use False if already have files in working directory, defaults to True
    :type download: bool, optional
    :param remove: option to delete the ascii files after loading, defaults to True
    :type remove: bool, optional
    :return: a dictionary containing 6 Timeseries objects.  microstrain, 4 corrections, and atmp (pressure)
    :rtype: dict of Timeseries
    """
    if download:
        download_l2_ascii(station=station, start=start, end=end)
    start_dt = dateutil.parser.parse(start)
    end_dt = dateutil.parser.parse(end)
    start_year = start_dt.year
    end_year = end_dt.year
    if start_year != end_year:
        years = range(start_year, end_year+1, 1)
    else:
        years = [start_year]
    for year in years:
        if year == years[0]:
            l2 = read_l2_ascii(station, year, strain_type=strain_type, show_stats=show_stats)
        else:
            ts_dict = read_l2_ascii(station, year, strain_type=strain_type, show_stats=show_stats)
            l2['microstrain'].append(ts_dict['microstrain'], in_place=True, show_stats=show_stats)
            l2['offset_c'].append(ts_dict['offset_c'], in_place=True, show_stats=show_stats)
            l2['tide_c'].append(ts_dict['tide_c'], in_place=True, show_stats=show_stats)
            l2['trend_c'].append(ts_dict['trend_c'], in_place=True, show_stats=show_stats)
            l2['atmp_c'].append(ts_dict['atmp_c'], in_place=True, show_stats=show_stats)
            l2['atmp'].append(ts_dict['atmp'], in_place=True, show_stats=show_stats)

    l2['microstrain'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)
    l2['offset_c'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)
    l2['tide_c'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)
    l2['trend_c'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)
    l2['atmp_c'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)
    l2['atmp'].truncate(new_start=start, new_end=end, in_place=True, show_stats=show_stats)    
   
    if remove:
        remove_l2_ascii(station=station, start=start, end=end)
    return l2


def read_l2_ascii(station: str,
                  year: str,
                  strain_type='gauge',
                  show_stats=True):
    """method that reads a single year of l2 data from disk into a dictionary of Timeseries

    :param station: Four character station id
    :type station: str
    :param year: Four character year YYYY
    :type year: str
    :param strain_type: whether to load 'gauge' or 'regional' strains, defaults to 'gauge'
    :type strain_type: str, optional
    :param show_stats: option to print gap analysis during loading, defaults to True
    :type show_stats: bool, optional
    :return: a dictionary containing 6 Timeseries objects.  microstrain, 4 corrections, and atmp (pressure)
    :rtype: dict of Timeseries
    """
    
    if strain_type == 'gauge':
        strains = ['Gauge0', 'Gauge1', 'Gauge2', 'Gauge3']
        channels = ['CH0','CH1','CH2','CH3']
        microstrains = ['gauge0(mstrain)','gauge1(mstrain)','gauge2(mstrain)','gauge3(mstrain)']
    elif strain_type == 'regional':
        strains = ['Eee+Enn', 'Eee-Enn', '2Ene']
        channels = strains 
        microstrains = ['Eee+Enn(mstrain)', 'Eee-Enn(mstrain)', '2Ene(mstrain)']
    microstrain_df = pd.DataFrame()
    offset_c_df = pd.DataFrame()
    tide_c_df = pd.DataFrame()
    trend_c_df = pd.DataFrame()
    atmp_c_df = pd.DataFrame()
    strain_quality_df = pd.DataFrame()
    atmp_c_quality_df = pd.DataFrame()
    version_df = pd.DataFrame()
    atmp_df = pd.DataFrame()
    level_df = pd.DataFrame()
    for i, strain in enumerate(strains):
        filebase = station+'.'+str(year)+'.bsm.level2'
        file=station+'.'+str(year)+'.xml.'+strain+'.txt.gz'
        df=pd.read_csv(filebase+'/'+file, compression='gzip', sep='\s+', index_col = 'date', parse_dates=True)
        microstrain_df[channels[i]]=df[microstrains[i]]
        strain_quality_df[channels[i]]=df['strain_quality']
        offset_c_df[channels[i]]=df['s_offset']
        tide_c_df[channels[i]]=df['tide_c']
        trend_c_df[channels[i]]=df['detrend_c']
        atmp_c_df[channels[i]]=df['atmp_c']
        atmp_c_quality_df[channels[i]]=df['atmp_c_quality']
        version_df[channels[i]]=df['version']
        level_df[channels[i]]=df['level']
    atmp_df['atmp'] = df['atmp']
    
    series = 'microstrain'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    microstrain = Timeseries(
            data=microstrain_df,
            quality_df=strain_quality_df,
            series=series,
            units="microstrain",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        microstrain.stats()

    series = 'offset_c'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    offset_c = Timeseries(
            data=offset_c_df,
            series=series,
            units="microstrain",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        offset_c.stats()

    series = 'tide_c'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    tide_c = Timeseries(
            data=tide_c_df,
            series=series,
            units="microstrain",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        tide_c.stats()

    series = 'trend_c'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    trend_c = Timeseries(
            data=trend_c_df,
            series=series,
            units="microstrain",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        trend_c.stats()

    series = 'atmp_c'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    atmp_c = Timeseries(
            data=atmp_c_df,
            series=series,
            units="microstrain",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        atmp_c.stats()

    series = 'atmp'
    logger.info(f'Loading {year} {strain_type} {series} into Timeseries')
    atmp = Timeseries(
            data=atmp_df,
            series=series,
            units="hPa",
            level='2a',
            name=f"{station}.{strain_type}.{series}",
            show_stats=show_stats
        ).remove_fill_values(fill_value=999999, interpolate=False, show_stats=show_stats)
    if show_stats:
        atmp.stats()
    return {
            "microstrain": microstrain, 
            "offset_c": offset_c, 
            "tide_c": tide_c, 
            "trend_c": trend_c,
            "atmp_c": atmp_c, 
            "atmp": atmp
            }
    

def download_l2_ascii(station: str,
                  start: str,
                  end: str):
    """method to download level 2 ascii tar files from EarthScope.  Untars into a folder and deletes the tarball.

    :param station: Four character station id
    :type station: str
    :param start: start time of date range
    :type start: str
    :param end: end time of desired date range
    :type end: str
    """
    start_dt = dateutil.parser.parse(start)
    end_dt = dateutil.parser.parse(end)
    start_year = start_dt.year
    end_year = end_dt.year
    if start_year != end_year:
        years = range(start_year, end_year+1, 1)
    else:
        years = [start_year]
    for year in years:
        filebase = station+'.'+str(year)+'.bsm.level2'
        url = 'http://bsm.unavco.org/bsm/level2/'+station+'/'+filebase+'.tar'

        #download tar
        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filebase+'.tar', 'wb') as f:
                f.write(response.raw.read())

        #unpack tar into tar.gz

        my_tar = tarfile.open(filebase+'.tar')
        my_tar.extractall() # specify which folder to extract to
        my_tar.close()

        #remove tar file
        os.remove(filebase+'.tar')

def remove_l2_ascii(station: str,
                  start: str,
                  end: str):
    """method to delete level 2 ascii folders from disk.  .

    :param station: Four character station id
    :type station: str
    :param start: start time of date range
    :type start: str
    :param end: end time of desired date range
    :type end: str
    """
    
    start_dt = dateutil.parser.parse(start)
    end_dt = dateutil.parser.parse(end)
    start_year = start_dt.year
    end_year = end_dt.year
    if start_year != end_year:
        years = range(start_year, end_year+1, 1)
    else:
        years = [start_year]
    for year in years:
        filebase = station+'.'+str(year)+'.bsm.level2'
        shutil.rmtree(filebase)
        
