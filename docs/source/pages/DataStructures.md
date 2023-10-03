# Data Structures

## Pandas DataFrames
Within this software package, we store timeseries data in pandas DataFrames, with timestamps as the index and one or more time-aligned datasets as columns.  A typical dataframe for GTSM gauge data would contain four columns, representing the different strain gauges (CH0, CH1, CH2, CH3).  By applying a calibration matrix to these data, we can convert the four gauges into a three column dataframe of areal (Eee+Enn), shear (2Ene), and differential (Eee-Enn) strains.  

Example DataFrame containing 300s gauge microstrain data
```
	                    CH0	        CH1	        CH2	        CH3
time				
2023-01-01 00:00:00	-145.455340	-22.864945	86.758526	-5.665800
2023-01-01 00:05:00	-145.455340	-22.865036	86.758526	-5.665800
2023-01-01 00:10:00	-145.455421	-22.864764	86.758052	-5.666071
2023-01-01 00:15:00	-145.455421	-22.864221	86.757293	-5.666791
2023-01-01 00:20:00	-145.455421	-22.863768	86.756819	-5.667332
...	...	...	...	...
2023-01-31 23:40:00	-145.635846	-22.826812	86.902057	-5.653822
2023-01-31 23:45:00	-145.635119	-22.825997	86.901582	-5.653912
2023-01-31 23:50:00	-145.634393	-22.825001	86.901108	-5.654002
2023-01-31 23:55:00	-145.634150	-22.824457	86.900349	-5.654542
2023-02-01 00:00:00	-145.633666	-22.823823	86.899685	-5.654722
8929 rows × 4 columns
```
New DataFrames are created during each processing step.  Calculated corrections are stored in their own DataFrame(s) as well, which can then be applied to raw data via simple combination.


## On Disk

### CSV
DataFrames can be saved to CSV (with optional compression), and that functionality is supported by the software. 

However, it is often desireable to store more than just the data itself, for example discription of the specific data, or quality flags allow keeping track of any data that is known to be bad or missing or interpolated.  We also may want to be able to version the data. Therefore, we have been developing a three dimensional array structure using TileDB to store processed strain data, as well as a python Class earthscopestraintools.timeseries.Timeseries to handle

### TileDB arrays
   

Processed data will be stored with a Tiledb array per station, and indexed along the following three dimensions.  Implementation of this is still under development, but the schema has been defined as follows.

| Dimensions | |
| --- | --- |
| data_type | variable length string. defines channel (i.e. 'CH0') or strain (i.e. 'Eee-Enn').  may also describe the calibration matrix used (i.e. 'Eee+Enn.ER2010') if choosing a calibration other than the default 'lab' |
| timeseries | variable length string. used to define whether the data is a measurement or a correction.  Options include ['counts', 'microstrain', 'offset_c', 'tide_c', 'trend_c', 'atmp_c'] |
| time | int64 unix milliseconds since 1970. |

Each cell in the multi-dimensional array will also have four attributes.

| Attributes | |
| --- | --- |
| data | (float64) the actual data value |
| quality | (char) single character quality flag (i.e. 'g'=good, 'b'=bad, 'm'=missing, 'i'=interpolated) |
| level | (str) one/two character level flag (i.e. '0','1','2a','2b') |
| version | versioning is intended to be used to identify processing metadata which may change with time.  not yet well implemented. |


## Timeseries Objects

We have created a class earthscopestraintools.timeseries.Timeseries, which is designed to capture all this various extra information and support writing to/reading from TileDB arrays.  Using these Timeseries objects is recommended, as it simplifies the processing workflow and provides built-in stats around missing/bad data.  

Each Timeseries object contains the following attributes:

| Attributes | |
| --- | --- |
| data | (pd.DataFrame) as described above, with datetime index and one or more columns of timeseries data |
| quality_df | (pd.DataFrame) autogenerated with same shape as data, but with a character mapped to each data point. flags include “g”=good, “m”=missing, “i”=interpolated, “b”=bad |
| series | (str) timeseries dimension for TileDB schema, ie ‘raw’, ‘microstrain’, ‘atmp_c’, ‘tide_c’, ‘offset_c’, ‘trend_c’ |
| units | (str) units of data |
| level | (str) level of data. ie. ‘0’,’1’,’2a’,’2b’ |
| period | (float) sample period of data |
| name | (str) optional name of timeseries, used for showing stats and plotting. defaults to network.station |
| network | (str) FDSN two character network code |
| station | (str) FDSN four character station code |


DataFrame data can be initially loaded into a Timeseries object either directly i.e.
```
from earthscopestraintools.timeseries import Timeseries
strain_raw = Timeseries(data=your_data_df, 
                        series="raw",
                        units="counts",
                        level="0",
                        period=1,
                        name="PB.B004.raw",
                        network="PB",
                        station="B004")
```
or (Recommended) by using the function mseed_to_ts(), which will call FDSN-DataSelect web service and load the requested data from the miniseed archive i.e.
```
from earthscopestraintools.mseed_tools import ts_from_mseed
start="2023-01-01T00:00:00"
end = "2023-02-01T00:00:00"
strain_raw = ts_from_mseed(network="PB",
                            station="B004",
                            location='T0',
                            channel='LS*', 
                            start=start, 
                            end=end)

```
Timeseries objects contain a number of processing methods, which build and return new timeseries objects.  For example, decimation of 1s data to the typical 300s data is performed by the following method, which returns a new Timeseries object
```
decimated_counts = strain_raw.decimate_1s_to_300s()
```
They also contain a built-in method stats() which displays a summary of the Timeseries object, including information on missing/interpolated data.  An Epoch is defined as a single row in the data, while a Sample is an individual value.
```
strain_raw.stats()
```
```
PB.B004.T0.LS*
    | Channels: ['CH0', 'CH1', 'CH2', 'CH3']
    | TimeRange: 2023-01-01 00:00:00 - 2023-02-01 00:00:00        | Period:             1s
    | Series:         raw| Units:        counts| Level:          0| Gaps:            0.06%
    | Epochs:     2678401| Good:     2676756.25| Missing:  1644.75| Interpolated:      0.0
    | Samples:   10713604| Good:       10707025| Missing:     6579| Interpolated:        0
```
Another built-in method plot() is useful for visualization of Timeseries data.  
```
strain_raw.plot()
```
See the api docs for more details on available methods and options, and the example notebooks for introductory usage.