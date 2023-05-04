from earthscopestraintools import ascii2tdb

if __name__ == '__main__':
    workdir = "arrays"
    network = 'PB'
    fcid = "B003"
    year = "2022"
    ascii2tdb.etl_yearly_ascii_file(network, fcid, year, workdir=workdir, print_it=True)

