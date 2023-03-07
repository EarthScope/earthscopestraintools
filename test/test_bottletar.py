from earthscopestraintools.bottletar import GtsmBottleTar
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

if __name__ == '__main__':

    t1 = datetime.now()

    filename = 'B001.2022001Day.tgz'  #24 hr Day session (archive and logger format)
    session = "Day"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.test()

    filename = 'B0012200100.tgz'  # 1 Hour, Hour Session (logger format)
    session = "Hour"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.test()

    filename = 'B0012200100_20.tar'  # 1 Hour, Min Session (logger format)
    session = "Min"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.test()

    filename = 'B001.2022001_01.tar' #24 Hour, Hour Session (archive format)
    session = "Hour"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.test()

    filename = 'B001.2022001_20.tar' #24 Hour, Min session (archive format)
    session = "Min"
    gbt = GtsmBottleTar(f"bottles/{filename}", session)
    gbt.test()

    t2 = datetime.now()
    elapsed_time = t2 - t1
    logger.info(f'{filename}: Elapsed time {elapsed_time} seconds')