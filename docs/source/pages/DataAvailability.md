# Data Availability

Data at various levels of processing are defined as follows:
- Level 0: Raw data in digital units with associated metadata. 
- Level 1: Raw data in geophysical units.
- Level 2: 5 min processed data and associated metadata.

A complete description of data processing strategies for level 0, 1, and 2 data is in the
[PBO Data Management System Critical Design Review](https://www.unavco.org/projects/past-projects/pbo/lib/docs/dms_cdr.pdf). Level 0 and 2 data are available through [Earthscope Data Services](https://www.unavco.org/data/strain-seismic/bsm-data/bsm-data.html) and the [IRIS Data Management Center Web Services](http://service.iris.edu/). 

Raw data are stored in bottle and ASCII format at UNAVCO, and SEED format at IRIS. We will primarily be accessing raw data in miniSEED format. Here is a brief description of the format with links for more info:
- [SEED](http://ds.iris.edu/ds/nodes/dmc/data/formats/#seed): Standard for the Exchange of Earthquake Data. Archive data format used by IRIS.
- [miniSEED](http://ds.iris.edu/ds/nodes/dmc/data/formats/miniseed/): How the borehole strainmeter data is stored at the IRIS Data Management Center. This is a stripped-down version of SEED containing only the time and data value, and does not include metadata - the miniSEED metadata contains this information. A station's miniSEED data and metadata compose a larger "SEED volume." 

Level 2 data are available from UNAVCO in XML and ASCII format. This includes gauge measurements in units of linearized strain, areal and shear strains, Earth tide and ocean load corrections, barometric pressure corrections, flagged bad data, and offset estimates. Generally, the raw data is decimated to 5 minute intervals to produce 5 minute Level 2 data products, but special high rate geophysical event data (e.g. data during an earthquake) is available separately at 1 sps.

We will primarily use the raw miniSEED data from IRIS and metadata associated with the Level 2 data products from UNAVCO. 

Raw miniSEED data is described by FDSN standard naming conventions, including 2-char network code, 4-char station code, 2-char location code, and 3-char channel code.  Below is a table of useful location and channel codes for data from the Gladwin Tensor Strainmeter instrument.  These are supplied as inputs to FDSN-WS DataSelect in this code via the ts_from_mseed() function.  

| Data Type| sample rate | location code | channel code(s)| scale factor
| --- | --- | --- | --- | --- | 
| strain (counts)| 20 hz | T0 | BS1, BS2, BS3, BS4| n/a
| strain  (counts)| 1 hz | T0 | LS1, LS2, LS3, LS4| n/a
| strain  (counts)| 10 m | T0 | RS1, RS2, RS3, RS4| n/a
| barometric pressure (hPa) | 30m | TS | RDO | 0.001
| rainfall (mm/30m) | 30m | TS | RRO | 0.0001

Note: Raw strain data is converted from counts to microstrain during the linearization processing step.
