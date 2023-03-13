from earthscopestraintools.gtsm_metadata import GtsmMetadata

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    network = 'IV'
    station = 'TSM6'
    meta = GtsmMetadata(network, station)
    meta.show()