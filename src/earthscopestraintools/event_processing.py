import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import datetime

import logging

logger = logging.getLogger(__name__)


def dynamic_strain(df, gauge_weights=[1, 1, 1, 1]):
    logger.info(f"Calculating dynamic strain using gauge weights: {gauge_weights}")
    ser = np.sqrt(
        (
            np.square(df.CH0) * gauge_weights[0]
            + np.square(df.CH1) * gauge_weights[1]
            + np.square(df.CH2) * gauge_weights[2]
            + np.square(df.CH3) * gauge_weights[3]
        )
        / sum(gauge_weights)
    )
    return ser.to_frame(name="dynamic")


def pre_event_trend_correction(df, eq_time):
    df_trend_c = pd.DataFrame(data=df.index)
    df_pre = pd.DataFrame(data=df.index[df.index < eq_time])
    # print(df_pre.index)
    for ch in df.columns:
        # print(df[ch][df.index < eq_time])
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_pre.index, df[ch][df.index < eq_time].interpolate()
        )
        # print(slope)
        df_trend_c[ch] = df_trend_c.index * slope

    return df_trend_c[df.columns].set_index(df_trend_c["time"])


def plot_coseismic_offset(
    df,
    title: str = "",
    remove_9s: bool = True,
    zero: bool = False,
    detrend: bool = None,
    ymin: float = None,
    ymax: float = None,
    plot_type: str = "scatter",
    units: str = None,
    eq_time: datetime.datetime = None,
    coseismic_offset: bool = False,
    color="black",
    save_as: str = None,
):
    fig, axs = plt.subplots(len(df.columns), 1, figsize=(12, 10), squeeze=False)

    if remove_9s:
        df = df.replace(999999, np.nan)
    if zero:
        title += " Zeroed"
        df -= df.loc[df.first_valid_index()]
    if detrend:
        df = df.interpolate()
        if detrend == "linear":
            if eq_time:
                title += " Linearly Detrended on Pre-Event Data"
                trend_c = pre_event_trend_correction(
                    df - df.loc[df.first_valid_index()], eq_time
                )
                df -= trend_c
            else:
                title += " Linearly Detrended"
                for ch in df.columns:
                    df[ch] = signal.detrend(df[ch], type="linear")
        else:
            print("Only linear detrend implemented")
    if coseismic_offset:
        # use first and last quintile
        df_pre = df[
            (df.index > np.percentile(df.index, 0))
            & (df.index <= np.percentile(df.index, 20))
        ]
        df_post = df[
            (df.index > np.percentile(df.index, 20))
            & (df.index <= np.percentile(df.index, 100))
        ]
        coseismic_offsets = df_post.mean() - df_pre.mean()

    for i, ch in enumerate(df.columns):
        if plot_type == "line":
            axs[i][0].plot(df[ch], color=color, label=ch)
        elif plot_type == "scatter":
            axs[i][0].scatter(df.index, df[ch], color=color, s=2, label=ch)
        else:
            print("Plot type must be either 'line' or 'scatter'")
        #         if self.units:
        #             axs[i][0].set_ylabel(self.units)
        if coseismic_offset:
            label = f"Co-seismic offset: {round(coseismic_offsets[i], 3)} ustrain"
            axs[i][0].text(
                0.85,
                0.2,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[i][0].transAxes,
            )
        if ymin or ymax:
            axs[i][0].set_ylim(ymin, ymax)
        if units:
            axs[i][0].set_ylabel(units)
        axs[i][0].ticklabel_format(axis="y", useOffset=False, style="plain")
        axs[i][0].grid()
        axs[i][0].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_as:
        print(f"Saving plot to {save_as}")
        plt.savefig(save_as, facecolor="white", transparent=False)
