from earthscopestraintools.bottle2mseed import bottle2mseed

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

    
    print_traces=True
    
    # network = "GF"
    # station = "BUY1"
    # filename = 'BUY12322201.tgz' #Hour Session
    # session = "Hour"
    # bottle2mseed(network, station, filename, session, print_traces)

    # network = "PB"
    # station = "B900"

    # filename = 'B9002323517_20.tar'  
    # session = "Min"
    
    # try:
    #     bottle2mseed(network, station, filename, session, verbose=False, print_traces=print_traces, plot_traces=True)
    # except Exception as e:
    #     logger.error(e)

    network = "PB"
    station = "B072"

    filename = 'B07223196Day.tgz'  #24 hr Day session (archive and logger format)
    session = "Day"
    st = bottle2mseed(network, station, filename, session, print_traces, plot_traces=False)
    #for tr in st:
    #    print(tr.stats.channel, tr.data)
    # filename = 'B0012200100.tgz'  # 1 Hour, Hour Session (logger format)
    # session = "Hour"
    # bottle2mseed(network, station, filename, session, print_traces, plot_traces=True)

    # filename = 'B0012200100_20.tar'  # 1 Hour, Min Session (logger format)
    # session = "Min"
    # bottle2mseed(network, station, filename, session, print_traces)

    # filename = 'B001.2022001_01.tar' #24 Hour, Hour Session (archive format)
    # session = "Hour"
    # bottle2mseed(network, station, filename, session, print_traces)

    # filename = 'B001.2022001_20.tar' #24 Hour, Min session (archive format)
    # session = "Min"
    # bottle2mseed(network, station, filename, session, print_traces)

    # network = "PB"
    # station = "B012"
    # filename = 'B0122136319_20.tar' #Min Session
    # session = "Min"
    # bottle2mseed(network, station, filename, session, print_traces)

    # network = "IV"
    # station = "TSM2"
    # filename = 'TSM22130619_20.tar' #Min Session
    # session = "Min"
    # bottle2mseed(network, station, filename, session, print_traces, plot_traces=True)

    

    # t2 = datetime.now()
    # elapsed_time = t2 - t1
    # logger.info(f'{filename}: Elapsed time {elapsed_time} seconds')
