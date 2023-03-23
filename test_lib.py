from earthscopestraintools.tiledbtools import RawStrainWriter, RawStrainReader
from earthscopestraintools.mseed_tools import load_miniseed_to_df, mseed2pandas
from earthscopestraintools.gtsm_metadata import GtsmMetadata
from earthscopestraintools.timeseries import Timeseries, ts_from_mseed

import logging
logger = logging.getLogger()
logging.basicConfig(
        format="%(message)s", level=logging.INFO
    )

network = 'PB'
station = 'B944'
meta = GtsmMetadata(network, station)
start = "2021-05-01T00:00:00"
end = "2021-05-01T01:00:00"
ts = ts_from_mseed(net=network, sta=station, loc='T0', cha='LS*', start=start, end=end)
ts.stats()
ts.plot()
