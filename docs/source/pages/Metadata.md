# Metadata 

This library currently supports one type of strainmeter, the Gladwin Tensor Strainmeter (GTSM).  We have implemented a class and methods to load and store the metadata associated with these instruments.
## earthscopestraintools.gtsm_metadata.GtsmMetadata
Class for loading and storing GTSM processing metadata for a given station. Metadata sources include a summary table containing basic information about all stations

https://www.unavco.org/data/strain-seismic/bsm-data/lib/docs/bsm_metadata.txt

and individual pages with station specific processing parameters (pressure response coefficients, tidal constituents, and calibration matricies)  i.e.

http://bsm.unavco.org/bsm/level2/B001/B001.README.txt

A station metadata object will contain the following attributes:
| Attributes |  |
| ----- | ----- |
| network | (str) FDSN network code|
| station | (str) FDSN station code|
| latitude | (float) station latitude|
| longitude | (float) station longitude|
| elevation | (float) station elevation in meters|
| gap | (float), instrument gap in meters |
| diameter | (float), instrument sensing diameter in meters |
| start_date | (str), formatted as "%Y-%m-%d" |
| orientation | (float), degrees East of North for CH0 |
| reference_strains | (dict), containing 'linear_date':'YYYY:DOY' and each channel ie 'CH0':reference counts  |
| strain_matrices | (dict), contains one or more calibration matrices, keyed to the name of the calibration.   |
| atmp_response | (dict), reponse coefficients for each channel |
| tidal_params | (dict), keys are tuple of (channel, tidal constituent, phz/amp/doodson) |

Example usage:
```
from earthscopestraintools.gtsm_metadata import GtsmMetadata
meta = GtsmMetadata(network='PB',station='B004')
meta.show()

network: PB
station: B004
latitude: 48.20193
longitude: -124.42701
gap: 0.0002
orientation (CH0EofN): 168.2
reference_strains:
 {'linear_date': '2005:180', 'CH0': 48391551, 'CH1': 49872537, 'CH2': 49840454, 'CH3': 49541470}
strain_matrices:
lab:
 [[ 0.2967  0.5185  0.2958  0.2222]
 [-0.2887  0.2983  0.1688 -0.1784]
 [-0.266  -0.2196  0.3531  0.1325]]
ER2010:
 [[ 1.65662916  2.37718929  2.36230912  0.39952254]
 [-0.04469343  0.51274012  0.50047667 -0.09672701]
 [ 0.39438025  1.05488877  1.64758286  0.22625875]]
CH_prelim:
 None
atmp_response:
 {'CH0': -0.004200000000000001, 'CH1': -0.0036000000000000003, 'CH2': -0.0046, 'CH3': -0.0040999999999999995}
tidal_params:
 {('CH0', 'M2', 'phz'): '114.965', ('CH0', 'M2', 'amp'): '9.590', ('CH0', 'M2', 'doodson'): '2 0 0 0 0 0', ...
```