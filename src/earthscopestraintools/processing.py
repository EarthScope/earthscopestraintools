import numpy as np
import pandas as pd
from scipy import signal
from earthscopestraintools.timeseries import Timeseries
from earthscopestraintools.gtsm_metadata import GtsmMetadata
import logging

logger = logging.getLogger(__name__)


def linearize(df: pd.DataFrame, metadata: GtsmMetadata):
    # build a series of reference strains.  if using metadata from XML the /1e8 is already included.
    reference_strains = pd.Series(dtype="float64")
    for ch in df.columns:
        reference_strains[ch] = metadata.linearization[ch]
    linearized_data = (
            (
                    ((df / 100000000) / (1 - (df / 100000000)))
                    - (
                            (reference_strains / 100000000)
                            / (1 - (reference_strains / 100000000))
                    )
            )
            * (metadata.gap / metadata.diameter)
            * 1000000
    )
    return linearized_data


def linearize_ts(ts: Timeseries, metadata: GtsmMetadata):
    """
    Processing step to convert digital counts to microstrain based on geometry of GTSM gauges
    :param ts: Timeseries object, containing data to convert to microstrain
    :param step_params: Dict, optional boolean params Plot, Print, Save
    :return: Timeseries object, in units of microstrain
    """

    # remove any 999999 values in data, ok to leave as Nan rather than interpolate.
    if ts.nines:
        logger.info(f"Found {ts.nines} 999999s, replacing with nans")
        df = ts.remove_999999s(interpolate=False)
    else:
        df = ts.data
    linearized_data = linearize(df, metadata)

    ts2 = Timeseries(
        data=linearized_data,
        quality_df=ts.quality_df,
        series="microstrain",
        units="microstrain",
        level="1",
        period=ts.period,
    )
    return ts2

def apply_calibration_matrix(df: pd.DataFrame, calibration_matrix: np.array, use_channels: list):
    regional_strain_df = np.matmul(
        calibration_matrix, df[df.columns].transpose()
    ).transpose()
    regional_strain_df = regional_strain_df.rename(
        columns={0: "Eee+Enn", 1: "Eee-Enn", 2: "2Ene"}
    )
    return regional_strain_df

def apply_calibration_matrix_ts(ts: Timeseries, calibration_matrix: np.array, use_channels: list = [1,1,1,1]):
    """
    Processing step to convert gauge strains into areal and shear strains
    :param ts: Timeseries object containing gauge data in microstrain
    :param calibration_matrix: np.array containing strain matrix
    :return: Timeseries object, in units of microstrain
    """
    # calculate areal and shear strains from gauge strains
    logger.info(calibration_matrix)
    # todo: implement UseChannels to arbitrary matrices
    data = apply_calibration_matrix(ts.data, calibration_matrix, use_channels)
    quality_df = pd.DataFrame(index=ts.quality_df.index, columns=["Eee+Enn", "Eee-Enn", "2Ene"], data='g')
    quality_df[ts.quality_df[ts.quality_df == 'm'].any(axis=1)] = 'm'
    ts2 = Timeseries(
        data=data,
        quality_df=quality_df,
        series=ts.series,
        units="microstrain",
        level="2a",
        period=ts.period,
    )
    return ts2

def butterworth_filter(
    df: pd.DataFrame,
    period: float,
    filter_type: str,
    filter_order: int,
    filter_cutoff_s: float,
):
    fc = 1 / filter_cutoff_s
    fs = 1 / period
    [bn, an] = signal.butter(filter_order, fc / (1 / 2 * fs), btype=filter_type)
    df2 = pd.DataFrame(index=df.index)
    for ch in df.columns:
        df2[ch] = signal.filtfilt(bn, an, df[ch])
    return df2


def butterworth_filter_ts(
    ts: Timeseries,
    filter_type: str,
    filter_order: int,
    filter_cutoff_s: float,
    series: str = "",
    name: str = None,
):

    df2 = butterworth_filter(
        df=ts.data,
        period=ts.period,
        filter_type=filter_type,
        filter_order=filter_order,
        filter_cutoff_s=filter_cutoff_s,
    )
    if not name:
        name = f"{ts.name}.filtered"
    return Timeseries(
        data=df2,
        quality_df=ts.quality_df,
        series=series,
        units=ts.units,
        period=ts.period,
        level=ts.level,
        name=name,
    )


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


def interpolate_ts(
    ts: Timeseries,
    replace: int = 999999,
    method: str = "linear",
    limit: int = 3600,
    limit_direction="both",
    name: str = None,
):
    # logger.info(f"Interpolating data using method={method} and limit={limit}")
    data = interpolate(
        ts.data,
        replace=replace,
        method=method,
        limit=limit,
        limit_direction=limit_direction,
    )
    mask1 = (data != ts.data).any(axis=1)
    mask2 = ts.data[mask1] == 999999
    quality_df = ts.quality_df.copy()
    quality_df[mask2] = "i"

    if not name:
        name = f"{ts.name}.interpolated"
    return Timeseries(
        data=data,
        quality_df=quality_df,
        series=ts.series,
        units=ts.units,
        level=ts.level,
        period=ts.period,
        name=name,
    )


def decimate_to_hourly(df: pd.DataFrame):
    return df[df.index.minute == 0]


def decimate_to_hourly_ts(ts: Timeseries, name: str = None):
    """

        :param ts: Timeseries
        :return: Timeseries, decimated to hourly
        """
    logger.info(f"Decimating {ts.period}s data to hourly using values where minutes=0.")
    if not name:
        name = f"{ts.name}.hourly"
    return Timeseries(
        data=decimate_to_hourly(ts.data),
        quality_df=decimate_to_hourly(ts.quality_df),
        series=ts.series,
        units=ts.units,
        level=ts.level,
        period=3600,
        name=name,
    )


def decimate_1s_to_300s(df: pd.DataFrame, method: str = "linear", limit: int = 3600):

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


def decimate_1s_to_300s_ts(ts: Timeseries, method: str = "linear", limit: int = 3600):

    data = decimate_1s_to_300s(ts.data, method=method, limit=limit)
    quality_df = ts.quality_df.replace("m", "i")
    name = f"{ts.name}.decimated"
    return Timeseries(
        data=data,
        quality_df=quality_df.reindex(index=data.index),
        series=ts.series,
        units=ts.units,
        level=ts.level,
        name=name,
        period=300,
    )
