import datetime
import urllib.request, json
import pytz


def version():
    print("Earthquake metadata module")
    print("Version 1.3.  M. Gottlieb 3-27-2023")


class Earthquake:
    """
    Class object containing parameters for a specific earthquake event
    """
    def __init__(self, event_id, focal_mechanism=False):
        self.event_id = event_id

        self.load_url_data()
        self.name = self.data["properties"]['title']
        self.mag = self.data["properties"]['mag']
        self.unix_time = self.data['properties']['time']
        self.time = datetime.datetime.fromtimestamp(self.unix_time / 1000.0).astimezone(pytz.utc)
        self.lat = self.data["geometry"]['coordinates'][1]
        self.long = self.data["geometry"]['coordinates'][0]
        self.depth = self.data["geometry"]['coordinates'][2]
        if focal_mechanism:
            self.get_focal_mechanism()


    def load_url_data(self):
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&eventid=" + self.event_id
        response = urllib.request.urlopen(url)
        self.data = json.loads(response.read())

    def get_focal_mechanism(self):
        focal_mechanism_props = self.data['properties']['products']['focal-mechanism'][0]['properties']
        np1_dip = focal_mechanism_props['nodal-plane-1-dip']
        np1_rake = focal_mechanism_props['nodal-plane-1-rake']
        np1_strike = focal_mechanism_props['nodal-plane-1-strike']
        #mag = self.data["properties"]['mag']
        self.strike = np1_strike
        self.dip = np1_dip
        self.rake = np1_rake
