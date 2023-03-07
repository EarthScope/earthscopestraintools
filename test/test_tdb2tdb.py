from earthscopestraintools.tdb2tdb import export_date_range

if __name__ == '__main__':
    net = "PB"
    fcid = "B001"
    start_str = "2022-01-01T00:00:00"
    end_str = "2022-02-01T00:00:00"
    export_date_range(net, fcid, start_str, end_str, write_it=True, print_it=True)