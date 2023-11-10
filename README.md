Welcome to earthscopestraintools, an open source python package for working with strainmeter data.  This project is actively under development and should not be considered stable at this time. 

M. Gottlieb and C. Hanagan 10-31-2023


Documentation can be found at https://earthscopestraintools.readthedocs.io/en/latest/ 

Pypi releases available from
https://pypi.org/project/earthscopestraintools/

earthscopestraintools can be installed with 
    
> pip install earthscopestraintools

Or with optional dependencies:
```
pip install 'earthscopestraintools[mseed]'
pip install 'earthscopestraintools[tiledb]'
pip install 'earthscopestraintools[mseed,tiledb]'
```

It is currently recommendeed to install the mseed optional dependencies, which includes obspy and some tools for loading data from the EarthScope miniseed archive.  The tiledb functionality, as of version 0.1.21, is not yet well implemented.  

Some functionality (tidal analysis and corrections) depends on BAYTAP-08 and SPOTL, two legacy FORTRAN programs, which have been containerized for use by this library.  In order to run these processing methods, you must have Docker installed and running on your computer.  When required, the library will then fetch the required images and be able to run these processing steps.