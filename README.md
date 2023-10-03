Welcome to earthscopestraintools, an open source python package for working with strainmeter data.  This project is actively under development and should not be considered stable at this time. 
M. Gottlieb 10-3-2023

Documentation can be found at https://earthscopestraintools.readthedocs.io/en/latest/ 

Pypi releases available from
https://pypi.org/project/earthscopestraintools/

earthscopestraintools can be installed with 
    
> pip install earthscopestraintools

Or with optional dependencies:

> pip install 'earthscopestraintools[mseed]'
> 
> pip install 'earthscopestraintools[tiledb]'
> 
> pip install 'earthscopestraintools[mseed,tiledb]'

It is currently recommendeed to install the mseed optional dependencies, which includes obspy and some tools for loading data from the EarthScope miniseed archive.