Welcome to earthscopestraintools, an open source python package for working with strainmeter data.  This project is available as a beta release, but is still actively under development and should not be considered stable at this time. 

M. Gottlieb and C. Hanagan 6-1-2024


[![Read the Docs](https://readthedocs.org/projects/earthscopestraintools/badge/?version=latest)](https://earthscopestraintools.readthedocs.io/en/latest/)

[![PyPI Version](https://img.shields.io/pypi/v/earthscopestraintools.svg)](https://pypi.org/project/earthscopestraintools) 

[![DOI](https://zenodo.org/badge/692156370.svg)](https://zenodo.org/doi/10.5281/zenodo.11555774)
 


earthscopestraintools can be installed with 
    
> pip install earthscopestraintools

Or with optional dependencies:
```
pip install 'earthscopestraintools[mseed]'
pip install 'earthscopestraintools[tiledb]'
pip install 'earthscopestraintools[plotting]'
pip install 'earthscopestraintools[mseed,tiledb,plotting]'
```

It is currently recommendeed to install the mseed optional dependencies, which includes obspy and some tools for loading data from the EarthScope miniseed archive.  The tiledb functionality, as of version 0.1.39, is not yet well implemented. Also please note that the tiledb dependencies do not work with python 3.12, but should work with python 3.9-3.11.  Plotting dependencies include ipympl for enabling notebook interaction, including plotting strain videos.

Some functionality (tidal analysis and corrections) depends on BAYTAP-08 and SPOTL, two legacy FORTRAN programs, which have been containerized for use by this library.  In order to run these processing methods, you must have Docker installed and running on your computer.  When required, the library will then fetch the required images and be able to run these processing steps.