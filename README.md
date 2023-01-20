This repo contains code to read and write strainmeter data to tiledb.
M. Gottlieb 1-20-23

### straintiledbarray.py
class with methods and metadata to interact with strain tiledb arrays.  
Methods include create, delete, consolidate, vacuum, read, write.  it also 
includes the current schema definition.

### ascii2tdb.py
ETL script to read level 2 ascii files and write them to tiledb

### tdb2ascii.py
distrubution script to generate level 2 ascii files based on a time query
back from tiledb and package as tarball

### tdb2tdb.py
distribution script to generate a subset tdb array based on a time query and 
package as tarball

### load_level2 notebook
notebook for demoing various ways to query processed strain data from tiledb and
how to plot the data