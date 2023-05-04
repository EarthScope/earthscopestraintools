from earthscopestraintools.tdb2tdb import export_date_range
workdir = "arrays"

if __name__ == '__main__':
    network = "PB"
    station = "B003"
    start_str = "2022-01-01T00:00:00"
    end_str = "2022-02-01T00:00:00"
    uri = f"{workdir}/{network}_{station}_l2_etl.tdb"
    export_date_range(uri, start_str, end_str, write_it=True, print_it=True)