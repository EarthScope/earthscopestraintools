import pandas as pd
from geopy.distance import distance
import math
import datetime
import urllib.request, json
import pytz


def version():
    print("Earthquake metadata module")
    print("Version 1.2.  M. Gottlieb 3-13-2023")


class Earthquake:
    """
    Class object containing parameters for a specific earthquake event
    """
    def __init__(self, event_id):
        self.event_id = event_id

        data = load_url_data(event_id)
        self.name = data["properties"]['title']
        self.mag = data["properties"]['mag']
        self.unix_time = data['properties']['time']
        self.time = datetime.datetime.fromtimestamp(self.unix_time / 1000.0).astimezone(pytz.utc)
        self.lat = data["geometry"]['coordinates'][1]
        self.long = data["geometry"]['coordinates'][0]
        self.depth = data["geometry"]['coordinates'][2]

def load_url_data(event_id):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&eventid=" + event_id
    response = urllib.request.urlopen(url)
    return json.loads(response.read())

def load_event_data(event_id):
    """
    Function loads earthquake parameters for a given event into an earthquake object

    Parameters
    ----------
    :param event_id: str
    :return: eq: earthquake class object
    """
    eq = Earthquake(event_id)
    return eq

def calc_hypocentral_dist(
        lat,
        long,
        eq):
    """
    Function calculates hypocentral distance (km) between lat,long and earthquake

    Parameters
    ----------
    :param lat: float
    :param long: float
    :param eq: earthquake class object
        Must include the following attributes
        :eq.lat: float
        :eq.long: float
        :eq.depth: float
    :return:
        hypocentral_dist: int

    """

    ed = distance((eq.lat, eq.long), (lat, long)).km
    hypocentral_dist = int(math.sqrt(float(ed) ** 2 + float(eq.depth) ** 2))
    return hypocentral_dist


def calculate_p_s_arrival(eq, latitude, longitude):
    """
    Function calculates arrival times for P and S waves at a given lat and long

    Parameters
    ----------
    :param eq: earthquake class object
        Must include the following attributes
        :eq.lat: float
        :eq.long: float
        :eq.time: datetime.datetime
    :param latitude: float, latitude
    :param longitude: float, longitude
    :return:
        :p_arrival: datetime.datetime
        :s_arrival: datetime.datetime
    """

    event_loc = "[" + str(eq.lat) + "," + str(eq.long) + "]"
    station_loc = "[" + str(latitude) + "," + str(longitude) + "]"
    url = "https://service.iris.edu/irisws/traveltime/1/query?evloc=" + event_loc + "&staloc=" + station_loc
    df = pd.read_table(url, sep="\s+", header=1, index_col=2, usecols=[2, 3])

    p_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'P').argmax()].Travel))
    s_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'S').argmax()].Travel))
    p_arrival = eq.time + p_delta
    s_arrival = eq.time + s_delta
    return p_arrival, s_arrival

def get_focal_mechanism(event_id):

    data = load_url_data(event_id)
    focal_mechanism_props = data['properties']['products']['focal-mechanism'][0]['properties']
    np1_dip = focal_mechanism_props['nodal-plane-1-dip']
    np1_rake = focal_mechanism_props['nodal-plane-1-rake']
    np1_strike = focal_mechanism_props['nodal-plane-1-strike']
    mag = data["properties"]['mag']

    return dict(strike=np1_strike, dip=np1_dip, rake=np1_rake, magnitude=mag)
