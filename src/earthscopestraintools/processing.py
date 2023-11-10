import numpy as np
import pandas as pd
from scipy import signal, stats
import subprocess

#from earthscopestraintools.gtsm_metadata import GtsmMetadata
import logging

logger = logging.getLogger(__name__)


def linearize(df: pd.DataFrame, reference_strains: dict, gap: float):
    """Linearize raw gauge strain.

    :param df: Dataframe with four columns corresponding to raw gauge data in units of counts.
    :type df: pd.DataFrame
    :param reference_strains: Dictionary with four entries noting the reference count value on each gauge to zero the timeseries against in the conversion to gauge strain (in microstrain). 
    :type reference_strains: dict
    :param gap: Instrument measurement gap. For most NOTA strainmeters, the reference gap is 0.0001 m.
    :type gap: float
    :return: DataFrame with with four columns of microstrain.
    :rtype: pd.DataFrame
    """

    logger.info(f"Converting raw counts to microstrain")
    # build a series of reference strains.  if using metadata from XML the /1e8 is already included.
    reference_strain_series = pd.Series(dtype="float64")
    for ch in df.columns:
        reference_strain_series[ch] = reference_strains[ch]
    linearized_data = (
        (
            ((df / 100000000) / (1 - (df / 100000000)))
            - (
                (reference_strain_series / 100000000)
                / (1 - (reference_strain_series / 100000000))
            )
        )
        * (gap / 0.087)
        * 1000000
    )
    return linearized_data


def apply_calibration_matrix(
    df: pd.DataFrame,
    calibration_matrix: np.array,
    calibration_matrix_name: str = None,
    use_channels: list = [1, 1, 1, 1],
):
    """Applies a calibration matrix to convert 4 gauges into areal, differential, and shear strains

    :param df: four channels of strain data or correction in microstrain
    :type df: pd.DataFrame
    :param calibration_matrix: calibration matrix 
    :type calibration_matrix: np.array
    :param calibration_matrix_name: name of calibration matrix used, defaults to None
    :type calibration_matrix_name: str, optional
    :param use_channels: not yet implemented, set to 0 to ignore a bad channel, defaults to [1, 1, 1, 1]
    :type use_channels: list, optional
    :return: areal, differential, and shear strains based on the given calibration
    :rtype: pd.DataFrame
    """
    # todo: implement use channels
    logger.info(f"Applying {calibration_matrix_name} matrix: {calibration_matrix}")
    areal = "Eee+Enn"
    differential = "Eee-Enn"
    shear = "2Ene"
    if calibration_matrix_name and calibration_matrix_name != "lab":
        areal += "." + calibration_matrix_name
        differential += "." + calibration_matrix_name
        shear += "." + calibration_matrix_name
    regional_strain_df = np.matmul(
        calibration_matrix, df[df.columns].transpose()
    ).transpose()
    regional_strain_df = regional_strain_df.rename(
        columns={0: areal, 1: differential, 2: shear}
    )
    return regional_strain_df


def butterworth_filter(
    df: pd.DataFrame,
    period: float,
    filter_type: str,
    filter_order: int,
    filter_cutoff_s: float,
):
    """Apply a butterworth filter to a DataFrame using scipy.signal.butter()

    :param df: data to filter
    :type df: pd.DataFrame
    :param period: sample period of data
    :type period: float
    :param filter_type: {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    :type filter_type: str
    :param filter_order: the order of the filter
    :type filter_order: int
    :param filter_cutoff_s: the filter cutoff frequency in seconds
    :type filter_cutoff_s: float
    :return: butterworth filtered data
    :rtype: pandas.DataFrame
    """
    logger.info(f"Applying Butterworth Filter")
    fc = 1 / filter_cutoff_s
    fs = 1 / period
    [bn, an] = signal.butter(filter_order, fc / (1 / 2 * fs), btype=filter_type)
    df2 = pd.DataFrame(index=df.index)
    for ch in df.columns:
        df2[ch] = signal.filtfilt(bn, an, df[ch])
    return df2


def interpolate(
    df: pd.DataFrame,
    replace: int = 999999,
    method: str = "linear",
    limit: int = 3600,
    limit_direction="both",
):
    """Interpolate across gaps in data using pd.DataFrame.interpolate()

    :param df: data to be interpolated
    :type df: pd.DataFrame
    :param replace: gap fill value to interpolate across, defaults to 999999
    :type replace: int, optional
    :param method: interpolation method, defaults to "linear"
    :type method: str, optional
    :param limit: max number of samples to interpolate, defaults to 3600
    :type limit: int, optional
    :param limit_direction: ['forward', 'backward', 'both'], defaults to "both"
    :type limit_direction: str, optional
    :return: interpolated data
    :rtype: pd.DataFrame
    """
    logger.info(f"Interpolating data using method={method} and limit={limit}")
    df2 = df.replace(replace, np.nan).interpolate(
        method=method, limit_direction=limit_direction, limit=limit
    )
    return df2


def decimate_to_hourly(df: pd.DataFrame):
    """decimates a timeseries to hourly by selecting the first and second and minute of each hour

    :param df: time series data to decimate
    :type df: pd.DataFrame
    :return: decimated data
    :rtype: pd.DataFrame
    """
    logger.info(f"Decimating to hourly")
    df1 = df[df.index.minute == 0]
    df2 = df1[df1.index.second == 0]
    return df2


def decimate_1s_to_300s(df: pd.DataFrame, method: str = "linear", limit: int = 3600):
    """Filter and decimate 1hz data to 5 min data using \n
    Agnew, Duncan Carr, and K. Hodgkinson (2007), Designing compact causal digital filters for 
    low-frequency strainmeter data , Bulletin Of The Seismological Society Of America, 97, No. 1B, 91-99

    :param df: 1 hz data
    :type df: pd.DataFrame
    :param method: method to interpolate across gaps, defaults to "linear"
    :type method: str, optional
    :param limit: largest gap to interpolate, defaults to 3600 samples
    :type limit: int, optional
    :return: 300s (5 min) data
    :rtype: pd.DataFrame
    """
    logger.info(f"Decimating to 300s")
    df2 = interpolate(df, method="linear", limit=limit)

    # zero the data to the first value prior to filtering
    initial_values = df2.iloc[0]
    df3 = df2 - initial_values

    # perform a 5 stage filter-and-decimate
    # 1s -> 2s -> 4s -> 12s -> 60s -> 300s
    # FIR filter coefficients from Agnew and Hodgkinson 2007

    wtsdby2d = [
        0.0983262,
        0.2977611,
        0.4086973,
        0.3138961,
        0.0494246,
        -0.1507778,
        -0.1123764,
        0.0376576,
        0.0996838,
        0.0154992,
        -0.0666489,
        -0.0346632,
        0.0322767,
        0.0399294,
        -0.0097461,
        -0.0341585,
        -0.0039241,
        0.0246776,
        0.0099725,
        -0.0157879,
        -0.0099098,
        0.0078510,
        0.0081126,
        -0.0026986,
        -0.0061424,
        0.0007108,
        0.0039659,
        -0.0006209,
        -0.0017117,
        0.0007240,
    ]

    wtsdby3c = [
        0.0373766,
        0.1165151,
        0.2385729,
        0.3083302,
        0.2887327,
        0.1597948,
        0.0058244,
        -0.0973639,
        -0.1051034,
        -0.0358455,
        0.0359044,
        0.0632477,
        0.0302351,
        -0.0168856,
        -0.0356758,
        -0.0190635,
        0.0126188,
        0.0159705,
        0.0082144,
        -0.0087978,
        -0.0037289,
        -0.0017068,
        0.0028335,
    ]

    wtsdby5b = [
        0.0218528,
        0.0458359,
        0.0908603,
        0.1359777,
        0.1830881,
        0.1993418,
        0.1957624,
        0.1561194,
        0.0994146,
        0.0346412,
        -0.0236544,
        -0.0580081,
        -0.0703257,
        -0.0555546,
        -0.0287709,
        0.0032613,
        0.0267938,
        0.0358952,
        0.0311186,
        0.0134283,
        -0.0028524,
        -0.0170042,
        -0.0176765,
        -0.0123123,
        -0.0036798,
        0.0057730,
        0.0059817,
        0.0083501,
        0.0000581,
        0.0005724,
        -0.0033127,
        0.0004411,
        -0.0030766,
        0.0016604,
    ]

    channels = df.columns

    stage1 = pd.DataFrame(index=df3.index)
    for ch in channels:
        data = df3[ch].values
        stage1[ch] = signal.lfilter(wtsdby2d, 1.0, data)
    stage1d = stage1.iloc[::2]

    stage2 = pd.DataFrame(index=stage1d.index)
    for ch in channels:
        data = stage1d[ch].values
        stage2[ch] = signal.lfilter(wtsdby2d, 1.0, data)
    stage2d = stage2.iloc[::2]

    stage3 = pd.DataFrame(index=stage2d.index)
    for ch in channels:
        data = stage2d[ch].values
        stage3[ch] = signal.lfilter(wtsdby3c, 1.0, data)
    stage3d = stage3.iloc[::3]

    stage4 = pd.DataFrame(index=stage3d.index)
    for ch in channels:
        data = stage3d[ch].values
        stage4[ch] = signal.lfilter(wtsdby5b, 1.0, data)
    stage4d = stage4.iloc[::5]

    stage5 = pd.DataFrame(index=stage4d.index)
    for ch in channels:
        data = stage4d[ch].values
        stage5[ch] = signal.lfilter(wtsdby5b, 1.0, data)
    stage5d = stage5.iloc[::5]

    # add back in the initial values
    decimated_data = (stage5d + initial_values).astype(int)
    return decimated_data


def calculate_offsets(df, limit_multiplier: int = 10, cutoff_percentile: float = 0.75):
    """Calculate offsets using first differencing method (add more details).  

    :param df: uncorrected data, as dataframe with datetime (in seconds) index and 1 channel per column
    :type df: pandas.DataFrame
    :param limit_multiplier: _description_, defaults to 10
    :type limit_multiplier: int, optional
    :param cutoff_percentile: _description_, defaults to 0.75
    :type cutoff_percentile: float, optional
    :return: _description_
    :rtype: _type_
    """
    logger.info(
        f"Calculating offsets using cutoff percentile of {cutoff_percentile} and limit multiplier of {limit_multiplier}."
    )
    first_diffs = df.interpolate(method='linear',limit=3600,limit_direction='both').diff()
    #  drop a percentage of 1st differences to estimate an offset cutoff
    drop = round(len(first_diffs) * (1 - cutoff_percentile))
    offset_limit = []
    df_offsets = pd.DataFrame(index=first_diffs.index)
    for ch in df.columns:
        # Offset limit is a multiplier x the average absolute value of first differences
        # within 2 st_dev of the first differences
        offset_limit.append(
            np.mean(first_diffs[ch].abs().sort_values().iloc[0:-drop]) * limit_multiplier
        )

        # CH edit. Calculate offsets from the detrended series
        # Justification: if the offset is calculated from the original series,
        # there may be an overcorrection that is noticeable, especially with large trends
        df_offsets[ch] = first_diffs[first_diffs[ch].abs() > offset_limit[-1]][ch]

    # make a dataframe of running total of offsets
    df_cumsum = df_offsets.fillna(0).cumsum()
    # us_limits =
    logger.info(f"Using offset limits of {[round(x,6) for x in offset_limit]}")
    return df_cumsum


def calculate_pressure_correction(df: pd.DataFrame, response_coefficients: dict):
    """Generate a pressure correction timeseries from pressure data and response coefficients

    :param df: atmospheric pressure data 
    :type df: pd.DataFrame
    :param response_coefficients: response coefficients for each channel loaded from metadata
    :type response_coefficients: dict
    :return: pressure corrections for each channel
    :rtype: pd.DataFrame
    """
    logger.info(f"Calculating pressure correction")
    data_df = pd.DataFrame(index=df.index)
    for key in response_coefficients:
        data_df[key] = df * float(response_coefficients[key])
    return data_df


def calculate_linear_trend_correction(df, method='linear',trend_start=None, trend_end=None):
    """Generate a linear trend correction via either a linear least squares calculation or a median trend calculation. The median trend calculation (based on MIDAS in Blewitt et al., 2016 for GNSS time series analysis) uses the median slope value from all points separated by roughly one lunar day, calculated after outliers beyond 2 median absolute deviations are removed. It will only work with > 3 days of data.

    :param df: uncorrected data, as dataframe with datetime index and 1 channel per column
    :type df: pd.DataFrame
    :param method: linear or median
    :type method: str, default is linear
    :param trend_start: start of window to calculate trend, defaults to first_valid_index()
    :type trend_start: datetime.datetime, optional
    :param trend_end: end of window to calculate trend, defaults to last_valid_index()
    :type trend_end: datetime.datetime, optional
    :return: trend correction timeseries for each column/channel in input data
    :rtype: pd.DataFrame
    """
    logger.info(f"Calculating linear trend correction")
    if not trend_start:
        trend_start = df.first_valid_index()
    if not trend_end:
        trend_end = df.last_valid_index()
    logger.info(f"    Trend Start: {trend_start}")
    logger.info(f"    Trend End: {trend_end}")
    df_trend_c = pd.DataFrame(data=df.index)
    windowed_df = df.copy()[trend_start:trend_end].interpolate().reset_index()
    if method == 'linear':
        for ch in df.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                windowed_df.index, windowed_df[ch]
            )
            df_trend_c[ch] = df_trend_c.index * slope
    elif method == 'median':
        # Goal time difference
        # Mean syndonic month, divided by lunar cycles (so 1 day)
        tdiff = (29*24*60*60+12*60*60+44*60)/30 # seconds
        windowed_df.index = windowed_df.time 
        windowed_df.drop(['time'],axis=1,inplace=True)
        end = pd.to_datetime(windowed_df.index[-1].timestamp() - tdiff,unit='s')
        df1 = windowed_df[:end]
        df2 = windowed_df[len(windowed_df)-len(df1):]
        # actual time difference after shifting dataframes
        new_tdiff = df2.index[0].timestamp()-df1.index[0].timestamp() # seconds
        logger.info(f"  Median trend calculated with points {new_tdiff/60/60} hr apart.")
        for ch in df.columns:
            med = (df2[ch].values - df1[ch].values)/new_tdiff
            medmad_std_dev1 = stats.median_abs_deviation(med)*1.4826 # See Blewitt et al. 2016 and Wilcox 2005
            tmp = med[med<(np.median(med)+medmad_std_dev1*2)]
            med_2sig = tmp[tmp>(np.median(med)-medmad_std_dev1*2)]
            slope = np.median(med_2sig)
            df_trend_c[ch] = (pd.to_numeric(df.index)/1e6 - df.index[0].timestamp())*slope
    # print("df_trend_c", df_trend_c)
    return df_trend_c[df.columns].set_index(df_trend_c["time"])


def calculate_tide_correction(df, period, tidal_parameters, longitude):
    """Generate tidal correction timeseries using SPOTL hartid

    :param df: uncorrected data, as dataframe with datetime index and 1 channel per column
    :type df: pd.DataFrame
    :param period: sample period of data, must be >= 1
    :type period: int
    :param tidal_parameters: tidal parameters loaded from station metadata
    :type tidal_parameters: dict
    :param longitude: station longitude
    :type longitude: float
    :return: tidal correction timeseries for each column/channel in input data
    :rtype: pd.DataFrame
    """
    logger.info(f"Calculating tide correction")
    # check if hartid in path, or else use spotl container
    result = subprocess.run("which hartid", shell=True)
    if result.returncode == 0:
        hartid = "hartid"
    else:
        hartid = "docker run -i --rm ghcr.io/earthscope/spotl hartid"
    start = df.first_valid_index()
    datestring = (
        str(start.year).zfill(4)
        + " "
        + str(start.month).zfill(2)
        + " "
        + str(start.day).zfill(2)
        + " "
        + str(start.hour).zfill(2)
        + " "
        + str(start.minute).zfill(2)
        + " "
        + str(start.second).zfill(2)
    )

    nterms = str(len(df))
    samp = int(period)
    cmd = f"{hartid} {datestring} {nterms} {samp}"
    # cmd = hartid + " " + datestring + " " + nterms + " " + samp

    channels = df.columns
    tides = set()
    for key in tidal_parameters:
        tides.add(key[1])
    cmds = {}
    # for i, ch in enumerate(gauges):
    for ch in channels:
        inputfile = f"printf 'l\n{longitude}\n"
        for tide in tides:
            inputfile += f" {tidal_parameters[(ch, tide, 'doodson')]} {tidal_parameters[(ch, tide, 'amp')].ljust(7, '0')} {tidal_parameters[(ch, tide, 'phz')].ljust(8, '0')}\n"
        inputfile += "-1'"
        cmds[ch] = inputfile + " | " + cmd

    df2 = pd.DataFrame(index=df.index)
    for ch in channels:
        output = subprocess.check_output(cmds[ch], shell=True).decode("utf-8")
        df2[ch] = np.fromstring(output, dtype=float, sep="\n")
    df2 = df2 * 1e-3
    return df2

def baytap_analysis(df,
                    atmp_df,
                    quality_df = None,
                    atmp_quality_df = None,
                    latitude=None,
                    longitude=None,
                    elevation=None,
                    dmin=0.001):
    '''
    This function accesses a docker container to run BAYTAP08 (Tamura 1991; Tamura and Agnew 2008) for tidal analysis. Time series (e.g. strain) and additional auxiliary input (e.g. pressure) are analyzed together to determine the amplitudes and phases of a combination of tidal constituents (M2, O1, P1, K1, N2, S2) in the time series, as well as a coefficient for the auxiliary input response. Please refer to the Baytap08 manual for more details, included suggested length of the time series for full detemination of the tidal constituent suite returned here (365+ days).
    :param df: DataFrame of timeseries with datetime index and one channel per column. Strain should be in microstrain, and pressure in hPa. 
    :type df: pd.DataFrame
    :param atmp_df: DataFrame with atmospheric pressure data and datetime index
    :type atmp_df: pd.DataFrame
    :param quality_df: DataFrame with flags designating the quality of the data. Any points that are not good (g) are ignores in the time series analysis. 
    :type quality_df: pd.DataFrame
    :param units: Units of strain, should match microstrain or nanostrain
    :type units: str
    :param atmp_quality_df: DataFrame with flags designating the quality of the pressure data. Any points that are not good (g) are ignores in the time series analysis. 
    :type atmp_quality_df: pd.DataFrame
    :param atmp_units: Units of atmospheric pressure data, should be hpa
    :type atmp_units: str
    :param latitude: latitude of the station
    :type latitude: float
    :param longitude: longitude of the station
    :type longitude: float
    :param elevation: elevation of the station
    :type elevation: float
    :param dmin: Drift parameter for the program. Large drift expects a linear trend. Small drift allows for rapid changes in the residual time series. 
    :type dmin: float
    :return: Dictionary of amplitudes and phases for each tidal constituent per gauge, and atmospheric pressure coefficient.
    :rtype: dict
    '''
    
    logger.info("Please note, this method expects continuous data in microstrain and pressure in hPa.")
    if quality_df is None and atmp_quality_df is None:
        logger.info("If there are any gaps, please fill them with 999999s or provide a dataframe of quality flags")
    df = df*1e3  #convert microstrain to nanostrain
        
    # Control file text
    span = shift = ndata = int(len(df))
    samp = ((df.index[-1].timestamp() - df.index[0].timestamp())/60/60)/(ndata-1)
    # Check that pressure data is same length 
    atmp_samp = ((atmp_df.index[-1].timestamp() - atmp_df.index[0].timestamp())/60/60)/(ndata-1)
    if atmp_samp != samp:
        print('Pressure data and strain data are not the same length, make sure they have the same sample rate and time frame.')
    yr = str(df.index[0].year); mo = str(df.index[0].month).zfill(2)
    day = str(df.index[0].day).zfill(2); hr = str(df.index[0].hour).zfill(2)
    control_str = f'&param \nkind=7, \n'
    control_str += f'span={span} , shift={shift} , \n'
    control_str += f'dmin={dmin}, \n'
    control_str += f'lpout=0, filout=1, \n'
    control_str += f'iaug=1, lagp=0,\n'
    control_str += f'maxitr=50,\n'
    control_str += f'igrp=5, \n'
    control_str += f'ndata={ndata},\n'
    control_str += f'inform=3, \n'
    control_str += f'year={yr},mon={mo},day={day},hr={hr},delta={samp},\n'
    control_str += f'lat={latitude},long={longitude},ht={elevation},grav=0.0,\n'
    control_str += f'rlim=999990.D0,spectw=3,\n'
    control_str += f'&end\n'
    control_str += f'BSM Strain {df.index[0]} to {df.index[-1]}\n'
    control_str += f'----\n'
    control_str += f'{"Station": <40} STATION NAME\n'
    control_str += f'{"PBO GTSM21": <40} INSTRUMENT NAME\n'
    control_str += f'----\n'
    control_str += f'{"Strain (counts or nstrain)": <40} UNIT OF TIDAL DATA\n'
    control_str += f'{"Barometric Pressure (mbar)": <40} TITLE OF ASSOSIATED DATASET\n'
    
    # Pressure data, auxiliary input    
    if atmp_quality_df is not None:
        atmp_df[atmp_quality_df != 'g'] = 999999
    aux_dstr = 'hpa\n'+np.array2string(atmp_df.values.flatten(),separator='\n',threshold=999999,suppress_small=True).replace('[','').replace(']','')
    
    baytap_results = {}
    baytap_results['atmp_response'] = {}
    baytap_results['tidal_params'] = {}

    shell=True
    #close any existing baytap containers
    tmpout = subprocess.run('docker stop baytap',capture_output=True,shell=shell,text=True)
    tmpout = subprocess.run('docker rm -f baytap',capture_output=True,shell=shell,text=True)
    
    # Start container with temp file system in the docker memory
    cmd1 = f'docker run --rm -i -d --name baytap ghcr.io/earthscope/baytap08 /bin/bash'
    subprocess.run(cmd1, shell=shell,text=True)
    print('Docker container started.')
    
    for ch in df.columns:
        if quality_df is not None:    
            df[quality_df[ch] != 'g'] = 999999
        dstr = ch+'\n'+np.array2string(df[ch].values,separator='\n',threshold=999999,suppress_small=True).replace('[','').replace(']','')
        
        # Write files for baytap
        cmd2 = f'docker exec baytap /bin/bash -c "echo -e \'{dstr}\' > data.txt"'
        cmd3 = f'docker exec baytap /bin/bash -c "echo -e \'{control_str}\' > control.txt"'
        cmd4 = f'docker exec baytap /bin/bash -c "echo -e \'{aux_dstr}\' > aux.txt"'
        # Run baytap
        cmd5 = f'docker exec baytap /bin/bash -c "cat data.txt | baytap08 control.txt results.txt aux.txt >> decomp.txt"'
        # Read results
        cmd6 = f'docker exec baytap /bin/bash -c "cat results.txt"'
        # Actually execute the commands
        subprocess.run(cmd2, shell=shell,text=True)
        subprocess.run(cmd3, shell=shell,text=True)
        subprocess.run(cmd4, shell=shell,text=True)
        subprocess.run(cmd5, shell=shell,text=True)
        output = subprocess.run(cmd6,capture_output=True,shell=shell,text=True)
        res = (output.stdout).split('\n')

        # save the results
        resp = list(filter(lambda x: 'respc' in x, res))
        baytap_results['atmp_response'][ch] = float(resp[0].split(' ')[-1].replace('D','E'))/1000
        dood_list = ['2 0 0 0 0 0','1-1 0 0 0 0','1 1-2 0 0 0',
                    '1 1 0 0 0 0','2-1 0 1 0 0','2 2-2 0 0 0']
        for const,dood in zip(['M2','O1','P1','K1','N2','S2'],dood_list):
            tid = list(filter(lambda x: '>' in x and const in x, res))
            baytap_results['tidal_params'][(ch, const, 'phz')] =  list(filter(lambda x: x != '', tid[0].split(' ')))[-4]
            baytap_results['tidal_params'][(ch, const, 'amp')] =  list(filter(lambda x: x != '', tid[0].split(' ')))[-2]
            baytap_results['tidal_params'][(ch, const, 'doodson')] = dood
        # for k,v in baytap_results['tidal_params'].items():
        #     if k[2] == 'amp':
        #         baytap_results['tidal_params'][k] = str(float(v)*1000)
    print('Atmospheric pressure responses in microstrain/hPa and tidal parameters in degrees/nanostrain ')
    subprocess.run('docker rm -f baytap', shell=True,text=True)
    print('Docker processes finished. Container removed.')
            
    return baytap_results

def spotl_predict_tides(latitude,longitude,elevation,glob_oc,reg_oc,greenf):
    '''
    Returns the complex numbers (from amplitude and phase)
    for the predicted areal and shear strains using spotl

    Expects regional model polygons to have already been constructed in the working directory (in this case, in the Docker container).

    :param latitude: Station latitude
    :type latitude: float
    :param longitude: Station longitude
    :type longitude: float
    :param elevation: Station elevation (m)
    :type elevation: float
    :param glob_oc: Global ocean model from SPOTL. e.g. osu.tpxo72.2010
    :type glob_oc: str
    :param reg_oc: Regional ocean model from SPOTL. e.g. osu.usawest.2010
    :type reg_oc: str
    :param greenf: Green's functions for the elastic earth structure from SPOTL. e.g. green.contap.std
    :type greenf: str
    :return: Areal, differential, and shear strain complex numbers for the M2 and O1 tides. (eEE+eNN)m2, (eEE-eNN)m2, (2EN)m2, (eEE+eNN)o1, (eEE-eNN)o1, (2EN)o1
    :rtype: dict
    '''
    # Start Docker container
    command = f'docker run --rm -w /opt/spotl/working/ --mount type=tmpfs,destination=/opt/spotl/work --name spotl -i -d ghcr.io/earthscope/spotl /bin/bash'
    print('Docker started')
    subprocess.check_output(command,shell=True)
    # # Run polymake for regional models of interest
    command = f'docker exec spotl /bin/bash -c "polymake << EOF > ../work/poly.{reg_oc} \n- {reg_oc} \nEOF"'
    subprocess.check_output(command,shell=True)
    # # M2 ocean load for the ocean model but exclude the area in specified polygon
    command = f'docker exec spotl /bin/bash -c "nloadf BSM {latitude} {longitude} {elevation} m2.{glob_oc} {greenf} l ../work/poly.{reg_oc} - > ../work/ex1m2.f1"'
    subprocess.check_output(command,shell=True)
    # # M2 ocean load for the regional ocean model in the area in specified polygon
    command = f'docker exec spotl /bin/bash -c "nloadf BSM {latitude} {longitude} {elevation} m2.{reg_oc} {greenf} l ../work/poly.{reg_oc} + > ../work/ex1m2.f2"'
    subprocess.check_output(command,shell=True)
    # #  Add the M2 loads computed above together
    command = f'docker exec spotl /bin/bash -c "cat ../work/ex1m2.f1 ../work/ex1m2.f2 | loadcomb c >  ../work/tide.m2"'
    subprocess.check_output(command,shell=True)
    # # O1 ocean load for the ocean model but exclude the area in specified polygon
    command = f'docker exec spotl /bin/bash -c "nloadf BSM {latitude} {longitude} {elevation} o1.{glob_oc} {greenf} l ../work/poly.{reg_oc} - > ../work/ex1o1.f1"'
    subprocess.check_output(command,shell=True)
    # # O1 ocean load for the regional ocean model in the area in specified polygon
    command = f'docker exec spotl /bin/bash -c "nloadf BSM {latitude} {longitude} {elevation} o1.{reg_oc} {greenf} l ../work/poly.{reg_oc} + > ../work/ex1o1.f2"'
    subprocess.check_output(command,shell=True)
    # # Add the O1 loads computed above together
    command = f'docker exec spotl /bin/bash -c "cat ../work/ex1o1.f1 ../work/ex1o1.f2 | loadcomb c >  ../work/tide.o1"'
    subprocess.check_output(command,shell=True)
    # # Compute solid earth wides and combine with above ocean loads
    command = f'docker exec spotl /bin/bash -c "cat ../work/tide.m2 | loadcomb t >  ../work/m2.tide.total"'
    subprocess.check_output(command,shell=True)
    command = f'docker exec spotl /bin/bash -c "cat ../work/tide.o1 | loadcomb t >  ../work/o1.tide.total"'
    subprocess.check_output(command,shell=True)
    # # Find the amps and phases, compute complex numbers:
    command = f'docker exec spotl /bin/bash -c "cat ../work/m2.tide.total"'
    out = subprocess.check_output(command,shell=True,text=True)
    outlist = list(filter(lambda x: x.startswith('s'), out.split('\n')))[0].split(' ')
    Eamp, Ephase, Namp, Nphase, ENamp, ENphase = np.float64(list(filter(lambda x: x != '' and x != 's' , outlist)))
    m2E, m2N, m2EN = complex(Eamp*np.cos(Ephase*np.pi/180),Eamp*np.sin(Ephase*np.pi/180)), complex(Namp*np.cos(Nphase*np.pi/180),Namp*np.sin(Nphase*np.pi/180)), complex(ENamp*np.cos(ENphase*np.pi/180),ENamp*np.sin(ENphase*np.pi/180))
    command = f'docker exec spotl /bin/bash -c "cat ../work/o1.tide.total"'
    out = subprocess.check_output(command,shell=True,text=True)
    outlist = list(filter(lambda x: x.startswith('s'), out.split('\n')))[0].split(' ')
    Eamp, Ephase, Namp, Nphase, ENamp, ENphase = np.float64(list(filter(lambda x: x != '' and x != 's' , outlist)))
    o1E, o1N, o1EN = complex(Eamp*np.cos(Ephase*np.pi/180),Eamp*np.sin(Ephase*np.pi/180)), complex(Namp*np.cos(Nphase*np.pi/180),Namp*np.sin(Nphase*np.pi/180)), complex(ENamp*np.cos(ENphase*np.pi/180),ENamp*np.sin(ENphase*np.pi/180))
    # Combine into areal and shear (differential and engineering) real and imaginary parts
    arealm2, diffm2, engm2 = m2E+m2N, m2E-m2N, 2*m2EN
    arealo1, diffo1, engo1 = o1E+o1N, o1E-o1N, 2*o1EN
    pred_tides = {'M2':{'areal':arealm2,'differential':diffm2,'engineering':engm2},
                    'O1':{'areal':arealo1,'differential':diffo1,'engineering':engo1}}
    subprocess.run('docker rm -f spotl',shell=True)
    print('Docker container stopped and removed.')
    return pred_tides