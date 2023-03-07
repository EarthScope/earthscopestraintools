from earthscopestraintools.tdb2ascii import write_date_range

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    fcid = "B001"
    net = "PB"
    start_str = "2022-01-01T00:00:00"
    end_str = "2022-02-01T00:00:00"
    write_date_range(net, fcid, start_str, end_str, write_it=False, print_it=True)
    