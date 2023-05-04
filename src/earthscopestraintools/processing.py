import numpy as np
import pandas as pd
from scipy import signal, stats
import subprocess

# from earthscopestraintools.timeseries import Timeseries
from earthscopestraintools.gtsm_metadata import GtsmMetadata
import logging

logger = logging.getLogger(__name__)


def linearize(df: pd.DataFrame, reference_strains: dict, gap: float):
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
    logger.info(f"Interpolating data using method={method} and limit={limit}")
    df2 = df.replace(replace, np.nan).interpolate(
        method=method, limit_direction=limit_direction, limit=limit
    )
    return df2


def decimate_to_hourly(df: pd.DataFrame):
    logger.info(f"Decimating to hourly")
    return df[df.index.minute == 0]


def decimate_1s_to_300s(df: pd.DataFrame, method: str = "linear", limit: int = 3600):
    logger.info(f"Decimating to 300s")
    df2 = interpolate(df, method="linear", limit=3600)

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
    logger.info(
        f"Calculating offsets using cutoff percentile of {cutoff_percentile} and limit multiplier of {limit_multiplier}."
    )
    first_diffs = df.diff()
    #  drop a percentage of 1st differences to estimate an offset cutoff
    drop = round(len(first_diffs) * (1 - cutoff_percentile))
    offset_limit = []
    df_offsets = pd.DataFrame(index=first_diffs.index)
    for ch in df.columns:
        # Offset limit is a multiplier x the average absolute value of first differences
        # within 2 st_dev of the first differences
        offset_limit.append(
            np.mean(abs(first_diffs[ch].sort_values().iloc[0:-drop])) * limit_multiplier
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
    logger.info(f"Calculating pressure correction")
    data_df = pd.DataFrame(index=df.index)
    for key in response_coefficients:
        data_df[key] = df * float(response_coefficients[key])
    return data_df


def calculate_linear_trend_correction(df, trend_start=None, trend_end=None):
    logger.info(f"Calculating linear trend correction")
    if not trend_start:
        trend_start = df.first_valid_index()
    if not trend_end:
        trend_end = df.last_valid_index()
    logger.info(f"    Trend Start: {trend_start}")
    logger.info(f"    Trend Start: {trend_end}")
    df_trend_c = pd.DataFrame(data=df.index)
    windowed_df = df.copy()[trend_start:trend_end].interpolate().reset_index()
    for ch in df.columns:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            windowed_df.index, windowed_df[ch]
        )
        df_trend_c[ch] = df_trend_c.index * slope
    # print("df_trend_c", df_trend_c)
    return df_trend_c[df.columns].set_index(df_trend_c["time"])


def calculate_tide_correction(df, period, tidal_parameters, longitude):
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
