import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
from matplotlib import cm
from geopy.distance import distance
import datetime
import math
from earthscopestraintools.gtsm_metadata import get_metadata_df

import logging

logger = logging.getLogger(__name__)

def calc_hypocentral_dist(
        eq_latitude,
        eq_longitude,
        eq_depth,
        station_latitude,
        station_longitude):
    """
    Function calculates hypocentral distance (km) between lat,long and earthquake. Note that the distance calculation does not account for Earth's curvature with depth, and should be used for regional earthquakes only.

    :param eq_latitude: latitude of earthquake
    :type eq_latitude: float
    :param eq_longitude: longitude of earthquake
    :type eq_longitude: float
    :param eq_depth: depth of earthquake
    :type eq_depth: float
    :param station_latitude: latitude of station
    :type station_latitude: float
    :param station_longitude: longitude of station
    :type station_longitude: float
    :return: hypocentral distance in km
    :rtype: int

    """

    ed = distance((eq_latitude, eq_longitude), (station_latitude, station_longitude)).km
    hypocentral_dist = int(math.sqrt(float(ed) ** 2 + float(eq_depth) ** 2))
    return hypocentral_dist


def calculate_p_s_arrival(eq_latitude,
                          eq_longitude,
                          eq_time,
                          station_latitude,
                          station_longitude):
    """
    Function calculates arrival times for P and S waves at a given lat and long

    :param eq_latitude: latitude of earthquake
    :type eq_latitude: float
    :param eq_longitude: longitude of earthquake
    :type eq_longitude: float
    :param eq_time: time of earthquake
    :type eq_time: datetime.datetime
    :param station_latitude: latitude of station
    :type station_latitude: float
    :param station_longitude: longitude of station
    :type station_longitude: float
    :return: p_arrival, s_arrival
    :rtype: datetime.datetime

    """

    event_loc = "[" + str(eq_latitude) + "," + str(eq_longitude) + "]"
    station_loc = "[" + str(station_latitude) + "," + str(station_longitude) + "]"
    url = "https://service.iris.edu/irisws/traveltime/1/query?evloc=" + event_loc + "&staloc=" + station_loc
    df = pd.read_table(url, sep="\s+", header=1, index_col=2, usecols=[2, 3])

    p_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'P').argmax()].Travel))
    s_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'S').argmax()].Travel))
    p_arrival = eq_time + p_delta
    s_arrival = eq_time + s_delta
    return p_arrival, s_arrival

def dynamic_strain(df, gauge_weights=[1, 1, 1, 1]):
    """Calculates dynamic strain as RMS of gauge strains

    :param df: dataframe containing gauge strains as columns and time as an index
    :type df: pandas.DataFrame
    :param gauge_weights: list of which gauges to include, defaults to [1, 1, 1, 1]
    :type gauge_weights: list, optional
    :return: dataframe containing a single column of dynamic strain, time as an index
    :rtype: pandas.DataFrame
    """
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
    """calculate a linear trend correction based on any data provided prior to event start time

    :param df: dataframe containing strain data
    :type df: pandas.DataFrame
    :param eq_time: time of event 
    :type eq_time: datetime.datetime
    :return: dataframe containing a linear trend correction
    :rtype: pandas.DataFrame
    """
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

def calculate_magnitude(dynamic_strain_df, hypocentral_distance, site_term, longitude_term):
    """Calculates a magnitude estimate based on Barbour et al 2021.

    :param dynamic_strain_df: dataframe containing dynamic strain during an event
    :type dynamic_strain_df: pandas.DataFrame
    :param hypocentral_distance: distance from station to event hypocenter, in km
    :type hypocentral_distance: float
    :param site_term: site term from Barbour et al 2021
    :type site_term: float
    :param longitude_term: longitude term from Barbour et al 2021
    :type longitude_term: float
    :return: dataframe containing a dynamic strain based magnitude estimate as a function of time
    :rtype: pandas.DataFrame
    """
    logger.info(f"Calculating magnitude from dynamic strain using site term {site_term} "
                f"and longitude term {longitude_term}")
    df2 = pd.DataFrame(index=dynamic_strain_df.index)
    df2['magnitude'] = ((np.log10(dynamic_strain_df['dynamic'].cummax() / 1000000)) +
                             (0.00072 * hypocentral_distance) +
                             (1.45 * math.log10(hypocentral_distance)) +
                             8.52 - longitude_term - site_term
                             ) / 0.92
    return df2

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
    """plot strain data from an event 

    :param df: strain data including an event signal
    :type df: pandas.DataFrame
    :param title: plot title, defaults to ""
    :type title: str, optional
    :param remove_9s: option to remove gap fill values, defaults to True
    :type remove_9s: bool, optional
    :param zero: option to zero the data against the first valid index, defaults to False
    :type zero: bool, optional
    :param detrend: option to linearly detrend data, will use pre-event data only if eq_time is provided, defaults to None
    :type detrend: bool, optional
    :param ymin: y-axis minimum for plot, defaults to None
    :type ymin: float, optional
    :param ymax: y-axis maximum for plot, defaults to None
    :type ymax: float, optional
    :param plot_type: matplotlib plot type. option of ['scatter','line'], defaults to "scatter"
    :type plot_type: str, optional
    :param units: units to display on y-axis, defaults to None
    :type units: str, optional
    :param eq_time: origin time of event, defaults to None
    :type eq_time: datetime.datetime, optional
    :param coseismic_offset: option to calculate and display the coseismic offset as a difference of the mean of the first quintile and last quintile of data, defaults to False
    :type coseismic_offset: bool, optional
    :param color: matplotlib color option, defaults to "black"
    :type color: str, optional
    :param save_as: filename to save plot, defaults to None
    :type save_as: str, optional
    """
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



def magnitude_plot(dynamic_strain_df: pd.DataFrame,
                   magnitude_df:pd.DataFrame,
                   eq_time: datetime.datetime,
                   eq_mag: datetime.datetime,
                   title: str=None,
                   save_as: str = None,):
    """plot dynamic strain and associated magnitude estimate on the same plot

    :param dynamic_strain_df: dataframe containing dynamic strain during an event
    :type dynamic_strain_df: pd.DataFrame
    :param magnitude_df: dataframe containing strain-based magnitude estimate during an event 
    :type magnitude_df: pd.DataFrame
    :param eq_time: event origin time
    :type eq_time: datetime.datetime
    :param eq_mag: published event magnitude from COMCAT 
    :type eq_mag: datetime.datetime
    :param title: plot title, defaults to None
    :type title: str, optional
    :param save_as: filename to save plot, defaults to None
    :type save_as: str, optional
    """
    num_colors = 4
    fig, ax = plt.subplots(figsize=(12, 3))
    colors = [cm.gnuplot(x) for x in np.linspace(0, 0.8, num_colors)]
    if title:
        fig.suptitle(title)
    ax.plot(dynamic_strain_df['dynamic'], color=colors[0], label='microstrain')
    ax2 = ax.twinx()
    ax2.plot(magnitude_df['magnitude'], color=colors[1],
             label=f'est magnitude: {round(magnitude_df["magnitude"].iloc[-1], 2)}')
    ax.axvline(eq_time, color=colors[2], label='earthquake time')
    ax2.axhline(eq_mag, color=colors[3], label=f'usgs magnitude: {eq_mag}')
    ax.set_ylabel('microstrain')
    ax2.set_ylabel('magnitude')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='right')
    fig.tight_layout()
    if save_as:
        logger.info(f"Saving plot to {save_as}")
        plt.savefig(save_as)

def get_stations_in_radius(latitude, longitude, depth, radius, print_it=False):
    """determine list of stations within a given radius from an event, using lat/long in gtsm metadata summary table

    :param latitude: event latitude
    :type latitude: float
    :param longitude: event longitude
    :type longitude: float
    :param depth: event depth
    :type depth: float
    :param radius: radius from event, in km
    :type radius: float
    :param print_it: option to print list of stations within radius, and calculated distances from event, defaults to False
    :type print_it: bool, optional
    :return: list of four character station codes for stations within the given radius
    :rtype: list
    """
    meta_df = get_metadata_df()
    station_list = []
    for station in meta_df.index:
        dist = calc_hypocentral_dist(latitude,
                                     longitude,
                                     depth,
                                         meta_df.loc[station]["LAT"],
                                         meta_df.loc[station]["LONG"])
        if dist <= radius:
            if print_it:
                print(f"{station} at {dist} km")
            station_list.append(station)
    return station_list