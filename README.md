This repo contains code to read and write strainmeter data to tiledb.
M. Gottlieb 1-20-23

### straintiledbarray.py
class with methods and metadata to interact with strain tiledb arrays.  
Methods include create, delete, consolidate, vacuum, read, write.  it also 
includes the current schema definition.

### ascii2tdb.py
ETL script to read level 2 ascii files and write them to tiledb local arrays.  then need to
> aws s3 sync arrayname s3://tiledb-strain/arrayname

### tdb2ascii.py
distrubution script to generate level 2 ascii files based on a time query
back from tiledb and package as tarball

### tdb2tdb.py
distribution script to generate a subset tdb array based on a time query and 
package as tarball

### bottle.py
library for reading gtsm bottle files

### bottletar.py
wrapper class for reading 5 cases of tarballs of bottle files and writing to tiledb

### bottle2mseed.py
add on functionality around bottletar to use obspy to write mseed files

### bottle2tiledb.py
add on functionality around bottletar to use straintiledbarray to write to tiledb
