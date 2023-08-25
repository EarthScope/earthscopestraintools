import os
import shutil
import tarfile
import datetime
from earthscopestraintools.bottle import Bottle
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


class GtsmBottleTar:
    """
    This class is designed to unpack the following 5 cases of GTSM tar file.
    1. 'Day' session.  24hrs, low rate data.  tgz of bottles.  Logger and DMC Archive format.
    2. 'Hour' session. 1hr, 1s data.  tgz of bottles.  Logger format
    3. 'Min' session.  1hr, 20hz data.  tar of 60 tgz of bottles.  Logger format
    4. 'Hour_Archive' session.  24hr, 1s data.  tar of 24 tgz of bottles.  Archive format
    5. 'Min_Archive' session.  24hr, 20hz data.  tar of 24 tar of 60 tgz of bottles.  Archive format

    Bottles are unpacked and temporarily written to filebase/bottles. 

    """

    def __init__(self, filename, session=None, fileobj=None, verbose=False):
        # requires filename, which is path to local object or original filename of fileobj
        self.file_metadata = {}
        self.file_metadata["filename"] = filename
        self.file_metadata["fcid"] = filename.split("/")[-1][0:4]
        self.file_metadata["filebase"] = filename.split("/")[-1][:-4]
        self.file_metadata["session"] = session
        if fileobj:
            self.fileobj = fileobj
        else:
            self.fileobj = None
        self.untar_files(verbose=verbose)
        self.bottle_list = self.list_bottle_dir()
        self.bottle_list.sort()

    def untar_files(self, verbose=False):
        # unpack tars as complex as  tar(tar(tgz(bottle)))
        logger.info(f"{self.file_metadata['filename']}: unpacking tar file")
        path1 = f"{self.file_metadata['filebase']}/level1"
        path2 = f"{self.file_metadata['filebase']}/level2"
        self.bottle_path = f"{self.file_metadata['filebase']}/bottles"
        bad_files = []
        if self.file_metadata["filename"].endswith(
            "tar"
        ):  # contains more tars or tgzs.  Min, Min_Archive, Hour_Archive session
            with tarfile.open(self.file_metadata["filename"], "r") as tar:
                names = tar.getnames()
                tar.extractall(path=path1)
                for name in names:
                    try:
                        if name.endswith(
                            "tar"
                        ):  # contains more tars or tgzs. Min_Archive session
                            with tarfile.open(f"{path1}/{name}", "r") as tar2:
                                names2 = tar2.getnames()
                                tar2.extractall(path=path2)
                                for name2 in names2:
                                    #print(name2)
                                    if name2.endswith(
                                        "tgz"
                                    ):  # only contains bottles. Min_Archive session
                                        with tarfile.open(
                                            f"{path2}/{name2}", "r:gz"
                                        ) as tgz2:
                                            tgz2.extractall(path=self.bottle_path)
                                    else:
                                        bad_files.append(name2)
                                        logger.error(
                                            f"{name2} expected to be a tgz but is not."
                                        )
                                shutil.rmtree(path2)
                        elif name.endswith(
                            "tgz"
                        ):  # only contains bottles.  Min or Hour_Archive session
                            #logger.info(name)
                            try:
                                with tarfile.open(f"{path1}/{name}", "r:gz") as tgz:
                                    tgz.extractall(path=self.bottle_path)
                            except Exception as e:
                                bad_files.append(name)
                                if verbose:
                                    logger.error(f"{self.file_metadata['filename']}: {type(e)} on {name}, skipping")
                    except Exception as e:
                        bad_files.append(name)
                        if verbose:
                            logger.error(f"{self.file_metadata['filename']}: {type(e)} on {name}, skipping")
                if len(bad_files):
                    logger.warning(f"{self.file_metadata['filename']}: Failed to open {len(bad_files)} tar/tgzs")
                shutil.rmtree(path1)

        elif self.file_metadata["filename"].endswith(
            "tgz"
        ):  # only contains bottles.  Day or Hour session
            try:
                with tarfile.open(self.file_metadata["filename"], "r:gz") as tgz:
                    tgz.extractall(path=self.bottle_path)
            except Exception as e:
                logger.error(f"{self.file_metadata['filename']}: {type(e)} cannot open tgz")
                raise

    def list_bottle_dir(self):
        return os.listdir(self.bottle_path)

    def load_bottles(self, verbose=False):
        # opens and adds bottlefiles to self.bottles
        self.bottles = []
        bottle_list = self.list_bottle_dir()
        bottle_list.sort()
        failed = []
        for bottlename in bottle_list:
            try:
                btl = Bottle(f"{self.bottle_path}/{bottlename}")
                btl.read_header()
                self.bottles.append(btl)
            except Exception as e:
                failed.append(bottlename)
                if verbose:
                    print(verbose)
                    logger.error(f"{self.file_metadata['filename']}: {type(e)} on {bottlename}, skipping")
        if len(failed):
            logger.warning(f"{self.file_metadata['filename']}: Failed to load {len(failed)} of {len(bottle_list)} bottles")
        if len(self.bottles) == 0:  
            raise 

    def load_bottle(self, bottlename):
        # open and returns a bottle
        return Bottle(f"{self.bottle_path}/{bottlename}")

    def delete_bottles_from_disk(self):
        # clean up everything extracted from original file
        shutil.rmtree(self.file_metadata["filebase"])

    def list_bottles(self):
        for bottle in self.bottles:
            print(bottle.file_metadata["filename"])

    def get_bottle_names(self):
        bottle_names = []
        for bottle in self.bottles:
            bottle_names.append(bottle.file_metadata["filename"])
        return bottle_names

    def test(self):
        logger.info(
            f"{self.file_metadata['filename']}: contains {len(self.bottle_list)} bottles"
        )
        self.load_bottles()
        logger.info(f"Successfully loaded {len(self.bottles)} bottles")


#
# if __name__ == "__main__":
#
#     # function for running local tests
#
#     t1 = datetime.datetime.now()
#
#     filename = "B001.2022001Day.tgz"  # 24 hr Day session (archive and logger format)
#     session = "Day"
#     # filename = 'B018.2016366_01.tar' #24 Hour, Hour Session (archive format)
#     # session = "Hour_Archive"
#     # filename = 'B018.2016366_20.tar' #24 Hour, Min session (archive format)
#     # session = "Min_Archive"
#     # filename = 'B0012200100.tgz'  #1 Hour, Hour Session (logger format)
#     # session = "Hour"
#     # filename = 'B0012200100_20.tar' #1 Hour, Min Session (logger format)
#     # session = "Min"
#     gbt = GtsmBottleTar(f"bottles/{filename}", session)
#     gbt.load_bottles()
#     for bottle in gbt.bottles:
#         print(bottle.file_metadata["filename"])
#     # print(bottle.get_unix_ms_timestamps())
#     gbt.delete_bottles_from_disk()
#
#     t2 = datetime.datetime.now()
#     elapsed_time = t2 - t1
#     logger.info(f'{gbt.file_metadata["filename"]}: Elapsed time {elapsed_time} seconds')
#     # print(gbt.file_metadata['filename'], gbt.file_metadata['fcid'], gbt.file_metadata['session'])
