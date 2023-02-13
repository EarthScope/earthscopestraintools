import os
import shutil
import tiledb
import tarfile
import numpy as np
import pandas as pd
import datetime
from bottle import Bottle
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class GtsmBottleTar:
    '''
    This class is designed to unpack the following 5 cases of GTSM tar file.
    1. 'Day' session.  24hrs, low rate data.  tgz of bottles.  Logger and DMC Archive format.
    2. 'Hour' session. 1hr, 1s data.  tgz of bottles.  Logger format
    3. 'Min' session.  1hr, 20hz data.  tar of 60 tgz of bottles.  Logger format
    4. 'Hour_Archive' session.  24hr, 1s data.  tar of 24 tgz of bottles.  Archive format
    5. 'Min_Archive' session.  24hr, 20hz data.  tar of 24 tar of 60 tgz of bottles.  Archive format

    Bottles are unpacked and temporarily written to filebase/bottles.  These bottles are then loaded into
    a pandas DataFrame and deleted.  The DataFrame is used as the buffer to write the data to tiledb.

    '''

    def __init__(self, filename, session=None, fileobj=None):
        #requires filename, which is path to local object or original filename of fileobj
        self.file_metadata = {}
        self.file_metadata['filename'] = filename
        self.file_metadata['fcid'] = filename.split('/')[-1][0:4]
        self.file_metadata['filebase'] = filename.split('/')[-1][:-4]
        self.file_metadata['session'] = session
        if fileobj:
            self.fileobj = fileobj
        else:
            self.fileobj = None
        self.untar_files()
        self.bottle_list = self.list_bottle_dir()
        self.bottle_list.sort()
        #self.load_bottles()
        #self.delete_bottles_from_disk()


    def untar_files(self):
        #unpack tars as complex as  tar(tar(tgz(bottle)))
        logger.info(f"{self.file_metadata['filename']}: unpacking tar file")
        path1 = f"{self.file_metadata['filebase']}/level1"
        path2 = f"{self.file_metadata['filebase']}/level2"
        self.bottle_path = f"{self.file_metadata['filebase']}/bottles"
        if self.file_metadata['filename'].endswith('tar'): #contains more tars or tgzs.  Min, Min_Archive, Hour_Archive session
            with tarfile.open(self.file_metadata['filename'], "r") as tar:
                names = tar.getnames()
                tar.extractall(path=path1)
                for name in names:
                    if name.endswith('tar'):  #contains more tars or tgzs. Min_Archive session
                        with tarfile.open(f"{path1}/{name}", "r") as tar2:
                            names2 = tar2.getnames()
                            tar2.extractall(path=path2)
                            for name2 in names2:
                                if name2.endswith('tgz'): #only contains bottles. Min_Archive session
                                    with tarfile.open(f"{path2}/{name2}", "r:gz") as tgz2:
                                        tgz2.extractall(path=self.bottle_path)
                                else:
                                    logger.error(f"{name2} expected to be a tgz but is not.")
                            shutil.rmtree(path2)
                    elif name.endswith('tgz'): #only contains bottles.  Min or Hour_Archive session
                        with tarfile.open(f"{path1}/{name}", "r:gz") as tgz:
                            tgz.extractall(path=self.bottle_path)
                shutil.rmtree(path1)

        elif self.file_metadata['filename'].endswith('tgz'): #only contains bottles.  Day or Hour session
            with tarfile.open(self.file_metadata['filename'], "r:gz") as tgz:
                tgz.extractall(path=self.bottle_path)

    def list_bottle_dir(self):
        return os.listdir(self.bottle_path)

    def load_bottles(self):
        #opens and adds bottlefiles to self.bottles
        self.bottles = []
        bottle_list = self.list_bottle_dir()
        bottle_list.sort()
        for bottlename in bottle_list:
            btl = Bottle(f"{self.bottle_path}/{bottlename}")
            btl.read_header()
            self.bottles.append(btl)

    def load_bottle(self, bottlename):
        #open and returns a bottle
        return Bottle(f"{self.bottle_path}/{bottlename}")

    def delete_bottles_from_disk(self):
        #clean up everything extracted from original file
        shutil.rmtree(self.file_metadata['filebase'])

    def list_bottles(self):
        for bottle in self.bottles:
            print(bottle.file_metadata['filename'])

    def get_bottle_names(self):
        bottle_names = []
        for bottle in self.bottles:
            bottle_names.append(bottle.file_metadata['filename'])
        return bottle_names

    # def build_tiledb_buffer(self):
    #     # builds a 3d tiledb buffer dataframe from a given set of bottles
    #     # index as row number
    #     # columns are data_type, timeseries, time, data, level, quality, version (3 dim + 4 attr)
    # 
    # 
    #     logger.info(f"{self.file_metadata['filename']}: loading {len(self.bottle_list)} bottles into dataframe")
    #     bottle_dfs = []
    #     for i, name in enumerate(self.bottle_list):
    #         bottle = self.load_bottle(name)
    #         bottle.read_header()
    #         #logger.info(bottle.file_metadata)
    #         data_type = bottle.file_metadata['channel']
    #         timeseries = 'raw'
    #         logger.info(f"{data_type}, {timeseries}")
    #         timestamps = bottle.get_unix_ms_timestamps()
    #         #timestamps = bottle.get_datetime_timestamps()
    # 
    #         data = bottle.read_data()
    #         # Important for Min_Archive session: close open bottles when done reading to free up memory
    #         bottle.file.close()
    #         level = '0'
    #         quality = 'g'
    #         version = int(datetime.datetime.now().strftime("%Y%j%H%M%S"))
    #         d = {'data_type': data_type,
    #              'timeseries': timeseries,
    #              'time': timestamps,
    #              'data': data,
    #              'level': level,
    #              'quality': quality,
    #              'version': version}
    #         bottle_df = pd.DataFrame(data=d)
    #         bottle_df.loc[bottle_df['data'] == 999999, 'quality'] = 'm'
    #         bottle_dfs.append(bottle_df)
    #     tiledb_buffer = pd.concat(bottle_dfs, axis=0).reset_index(
    #         drop=True)
    #     tiledb_buffer['data'] = tiledb_buffer['data'].astype(np.float64)
    #     tiledb_buffer['version'] = tiledb_buffer['version'].astype(np.int64)
    #     self.delete_bottles_from_disk()
    #     return tiledb_buffer
    # 
    # 
    # def write_array(self, array):
    #     #writes a 3d tiledb buffer dataframe to a given tiledb array
    #     tiledb.from_pandas(uri=array.uri,
    #                        dataframe=self.tiledb_buffer,
    #                        index_dims=['data_type', 'timeseries', 'time'],
    #                        mode='append',
    #                        ctx=array.ctx
    #                        )
    # 
    # def to_tiledb(self, array):
    #     #method called to trigger building buffer dataframe and writing to tiledb
    #     #if specified array does not exist, it will make one
    #     try:
    #         #num_bottles = len(self.bottles)
    #         num_bottles = len(self.bottle_list)
    #         if num_bottles:
    #             logger.info(f"{self.file_metadata['filename']}: {self.file_metadata['session']} tar contains {num_bottles} bottles.")
    #             #self.tiledb_buffer = self.build_tiledb_buffer(self.bottles)
    #             self.tiledb_buffer = self.build_tiledb_buffer()
    #             logger.info(f"{self.file_metadata['filename']}: Writing to {array.uri}")
    #             self.write_array(array)
    #             logger.info(f"{self.file_metadata['filename']}: Written to {array.uri}")
    #             array.consolidate_meta()
    #             array.vacuum_meta()
    #             self.file_metadata['start_time'] = self.tiledb_buffer['time'].iloc[0]
    #             self.file_metadata['end_time'] = self.tiledb_buffer['time'].sort_values().iloc[-1]
    #         else:
    #             logger.error(f"Error: GtsmLoggerFile object {self.file_metadata['filename']} contains no bottles")
    # 
    #     except tiledb.TileDBError as e:
    #         try:
    #             logger.error(e)
    #             logger.warning(f"Array {array.uri} does not exist, creating.")
    #             array.create()
    #             logger.info(f"Created {array.uri}")
    #             logger.info(f"{self.file_metadata['filename']}: Writing to {array.uri}")
    #             self.write_array(array)
    #             logger.info(f"{self.file_metadata['filename']}: Written to {array.uri}")
    #             array.consolidate_meta()
    #             array.vacuum_meta()
    #             self.file_metadata['start_time'] = self.tiledb_buffer['time'].iloc[0]
    #             self.file_metadata['end_time'] = self.tiledb_buffer['time'].sort_values().iloc[-1]
    # 
    #         except Exception as e:
    #             logger.exception(e)
    # 
    #     except Exception as e:
    #         logger.exception(e)


if __name__ == '__main__':

    #function for running local tests

    t1 = datetime.datetime.now()

    filename = 'B001.2022001Day.tgz'  #24 hr Day session (archive and logger format)
    session = "Day"
    #filename = 'B018.2016366_01.tar' #24 Hour, Hour Session (archive format)
    #session = "Hour_Archive"
    # filename = 'B018.2016366_20.tar' #24 Hour, Min session (archive format)
    # session = "Min_Archive"
    #filename = 'B0012200100.tgz'  #1 Hour, Hour Session (logger format)
    #session = "Hour"
    #filename = 'B0012200100_20.tar' #1 Hour, Min Session (logger format)
    #session = "Min"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.load_bottles()
    for bottle in gbt.bottles:
        print(bottle.file_metadata['filename'])
    #print(bottle.get_unix_ms_timestamps())
    gbt.delete_bottles_from_disk()
    

    
    t2 = datetime.datetime.now()
    elapsed_time = t2 - t1
    logger.info(f'{gbt.file_metadata["filename"]}: Elapsed time {elapsed_time} seconds')
    #print(gbt.file_metadata['filename'], gbt.file_metadata['fcid'], gbt.file_metadata['session'])