# Data Availability

Data at various levels of processing are defined as follows:
- Level 0: Raw data in digital units with associated metadata. 
- Level 1: Raw data in geophysical units.
- Level 2: 5 min processed data and associated metadata.

A complete description of data processing strategies for level 0, 1, and 2 data is in the
[PBO Data Management System Critical Design Review](https://www.unavco.org/projects/past-projects/pbo/lib/docs/dms_cdr.pdf). Level 0 and 2 data are available through [Earthscope Data Services](https://www.unavco.org/data/strain-seismic/bsm-data/bsm-data.html) and the [Earthscope Data Management Center Web Services](http://service.iris.edu/). 

Raw data is collected in bottle format, and made available in ASCII and miniSEED format via the Earthscope DMC (formerly IRIS DMC). We will primarily be accessing raw data in miniSEED format. Here is a brief description of the format with links for more info:
- [SEED](http://ds.iris.edu/ds/nodes/dmc/data/formats/#seed): Standard for the Exchange of Earthquake Data. Archive data format used by IRIS.
- [miniSEED](http://ds.iris.edu/ds/nodes/dmc/data/formats/miniseed/): How the borehole strainmeter data is stored at the Earthscope Data Management Center. This is a stripped-down version of SEED containing only the time and data value, and does not include metadata - the miniSEED metadata contains this information. A station's miniSEED data and metadata compose a larger "SEED volume." 

Level 2 data, where available, are in XML and ASCII formats. This includes gauge measurements in units of linearized strain, areal and shear strains, Earth tide and ocean load corrections, barometric pressure corrections, flagged bad data, and offset estimates. Generally, the raw data is decimated to 5 minute intervals to produce 5 minute Level 2 data products, but special high rate geophysical event data (e.g. data during an earthquake) is available separately at 1 sps.

We will primarily use the raw miniSEED data and metadata associated with the Level 2 data products, which is available on the Earthscope (formerly UNAVCO) website.

Raw miniSEED data is described by FDSN standard naming conventions, including 2-char network code, 4-char station code, 2-char location code, and 3-char channel code.  Below are some  tables of useful codes for data from the Gladwin Tensor Strainmeter instrument.  These are supplied as inputs to FDSN-WS DataSelect in this code via the ts_from_mseed() function.  

| Network  Code | Country | Number of stations | Network Start
| --- | --- | --- | --- | 
| PB | US, Canada | 74 | 2005
| GF | Turkey | 6 | 2014
| IV | Italy | 6 | 2021

<br /> 

| Data Type| Sample Rate | Location Code | Channel Code(s)| Scale Factor
| --- | --- | --- | --- | --- | 
| strain (counts)| 20 hz | T0 | BS1, BS2, BS3, BS4| n/a (1)
| strain  (counts)| 1 hz | T0 | LS1, LS2, LS3, LS4| n/a (1)
| strain  (counts)| 10 m | T0 | RS1, RS2, RS3, RS4| n/a (1)
| barometric pressure (hPa) | 30m | TS | RDO | 0.001 (2)
| rainfall (mm/30m) | 30m | TS | RRO | 0.0001
| downhole temperature (C) | 30m | T0 | RKD | 0.0001
| logger temperature (C) | 30m | T0 | RK1 | 0.0001
| power box temperature (C) | 30m | T0 | RK2 | 0.0001
| charging current (A) | 30m | T0 | REO | 0.0001
| battery voltage (V) | 30m | T0 | RE1 | 0.0001
| system current (A) | 30m | T0 | RE2 | 0.0001  

<br /> 

| Stations may also have | Sample Rate | Location Code | Channel Code(s)| Scale Factor
| --- | --- | --- | --- | --- | 
| pore pressure (hPa)| 1 hz | ++ | LDD | 0.001 (2,3)
| pore temperature (C)| 1 hz | ++ | LKD | 0.0001 (3)
| high rate barometer (hPa)| 1 hz | ++ | LDO | 0.000142937 (4,5)

<br /> 

Note 1: Raw strain data is converted from counts to microstrain during the linearization processing step.

Note 2: Pressure channels have a scale factor of 1/1000 instead of 1/10000.  This is because the processing is typically done in units of hPa rather than kPa.  

Note 3: Only pore pressure logged by Q330s in the PB network are currently available in miniseed via these codes.  IV pore pressure data is logged on field computers as daily ascii files.  Any data we have for these is made available at pore.unavco.org/pore/  

Note 4: High rate LDO pressure, where available, is logged (in counts) on a Setra 270 barometer with a range of 800-1100 hPa.  Conversion requires a scale factor of 0.000142937 plus a static offset of 800 kPa to get absolute and not relative pressure.  

Note 5. A small subset of stations (Yellowstone) use a high altitude version of the Setra 270, with a range of 600-1100 hPa.  For these stations, conversion requires a scale factor of 0.00023842 plus a static offset of 600 kPa to get absolute pressure.  
