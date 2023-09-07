import os
import datetime
import struct
import logging

logger = logging.getLogger(__name__)

BTL_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0, 0, None)
BTL_TYPE = {0: 'INT2', 1: 'INT4', 2: 'FLOAT4'}
BTL_MISSING = {0: 32767, 1: 999999, 2: 999999}
BTL_TYPE_SIZE = {0: 2, 1: 4, 2: 4}
BTL_TYPE_TO_PACK = {0: 'h', 1: 'i', 2: 'f'}
BTL_TYPE_TO_PRINT = {0: 'd', 1: 'd', 2: 'g'}

class Bottle:
    def __init__(self, filepath, fileobj=None):
        """
        """
        self.file_metadata = {}
        self.file_metadata['filepath'] = filepath
        self.file_metadata['filename'] = filepath.split("/")[-1]
        if fileobj and filepath:
            self.file = fileobj
        elif filepath:
            if not os.path.isfile(self.file_metadata['filepath']):
                logger.error("Can not find file '%s'" % self.file_metadata['filepath'])

            self.file = open(self.file_metadata['filepath'], 'rb')

        magic = self.file.read(2)
        if chr(magic[0]) == chr(0x01) and chr(magic[1]) == chr(0x5D):
            self.file_metadata['endian'] = '>'
        elif chr(magic[0]) == chr(0x5D) and chr(magic[1]) == chr(0x01):
            self.file_metadata['endian'] = '<'
        else:
            print(magic)
            print("Not recognized as a bottle file: %s" % self.file_metadata['filename'])
            logger.error("Not a bottle", self.file_metadata['filename'])

    def parse_filename(self):
        """
        Add info encoded within file name of bottle file from GTSM
        Logger to file metadata.  hour and min may be None.  All values are strings, even
        if they contain only digits.
        """

        # FIXME: we are hardcoding here knowledge of how to pick apart
        # a GTSM file name.  no other option really, but just be wary
        # of trusting file names.  and if you feed in something which
        # isn't a GTSM bottle file name, you'll get junk back - or
        # raise an exception.

        f = self.file_metadata['filename']

        stn4char = f[0:4]
        year = "20" + f[4:6]
        dayofyear = f[6:9]

        if f[-3:] == '_20' and len(f) == 19:
            # this is a Min file
            self.file_metadata['session'] = "Min"
            hour = f[9:11]
            min = f[11:13]
            channel = f[13:16]

        elif f[-3:] == '_20' and len(f) == 17:
            # this is the concatenation of 60 Min files
            hour = f[9:11]
            min = None
            channel = f[11:14]

        elif f[-3:-1] == 'CH' and len(f) == 14:
            # this is an Hour file
            self.file_metadata['session'] = "Hour"
            hour = f[9:11]
            min = None
            channel = f[11:14]

        else:
            # well, we have to assume it is a Day file
            self.file_metadata['session'] = "Day"
            hour = None
            min = None
            channel = f[9:]

        self.file_metadata['fcid'] = stn4char
        self.file_metadata['channel'] = channel
        self.file_metadata['year'] = year
        self.file_metadata['dayofyear'] = dayofyear
        self.file_metadata['hour'] = hour
        self.file_metadata['min'] = min

        self.get_seed_codes()

        # return self.file_metadata #(stn4char, channel, year, dayofyear, hour, min)

    def get_seed_codes(self):
        codes = {"BatteryVolts": ["T0", "RE1"], "CH0": ["T0", "RS1"], "CH1": ["T0", "RS2"], "CH2": ["T0", "RS3"],
                 "CH3": ["T0", "RS4"], "CalOffsetCH0G1": ["T1", "RCA"], "CalOffsetCH0G2": ["T2", "RCA"],
                 "CalOffsetCH0G3": ["T3", "RCA"], "CalOffsetCH1G1": ["T1", "RCB"], "CalOffsetCH1G2": ["T2", "RCB"],
                 "CalOffsetCH1G3": ["T3", "RCB"], "CalOffsetCH2G1": ["T1", "RCC"], "CalOffsetCH2G2": ["T2", "RCC"],
                 "CalOffsetCH2G3": ["T3", "RCC"], "CalOffsetCH3G1": ["T1", "RCD"], "CalOffsetCH3G2": ["T2", "RCD"],
                 "CalOffsetCH3G3": ["T3", "RCD"], "CalStepCH0G1": ["T4", "RCA"], "CalStepCH0G2": ["T5", "RCA"],
                 "CalStepCH0G3": ["T6", "RCA"], "CalStepCH1G1": ["T4", "RCB"], "CalStepCH1G2": ["T5", "RCB"],
                 "CalStepCH1G3": ["T6", "RCB"], "CalStepCH2G1": ["T4", "RCC"], "CalStepCH2G2": ["T5", "RCC"],
                 "CalStepCH2G3": ["T6", "RCC"], "CalStepCH3G1": ["T4", "RCD"], "CalStepCH3G2": ["T5", "RCD"],
                 "CalStepCH3G3": ["T6", "RCD"], "DownholeDegC": ["T0", "RKD"], "LoggerDegC": ["T0", "RK1"],
                 "PowerBoxDegC": ["T0", "RK2"], "PressureKPa": ["TS", "RDO"], "RTSettingCH0": ["T0", "RCA"],
                 "RTSettingCH1": ["T0", "RCB"], "RTSettingCH2": ["T0", "RCC"], "RTSettingCH3": ["T0", "RCD"],
                 "Rainfallmm": ["TS", "RRO"], "SolarAmps": ["T0", "REO"], "SystemAmps": ["T0", "RE2"],
                 "CalOffsetCH0G0": ["T7", "RCA"], "CalOffsetCH1G0": ["T7", "RCB"], "CalOffsetCH2G0": ["T7", "RCC"],
                 "CalOffsetCH3G0": ["T7", "RCD"], "CalStepCH0G0": ["T8", "RCA"], "CalStepCH1G0": ["T8", "RCB"],
                 "CalStepCH2G0": ["T8", "RCC"], "CalStepCH3G0": ["T8", "RCD"]}

        self.file_metadata['seed_loc'] = codes[self.file_metadata['channel']][0]
        self.file_metadata['seed_ch'] = codes[self.file_metadata['channel']][1]
        
        if self.file_metadata['channel'].startswith("CH"):
            if self.file_metadata['session'] == "Min":
                self.file_metadata['seed_ch'] = self.file_metadata['seed_ch'].replace("R","B")
            elif self.file_metadata['session'] == "Hour":
                self.file_metadata['seed_ch'] = self.file_metadata['seed_ch'].replace("R","L")


    def read_header(self, print_it=False):
        """
        """
        self.parse_filename()
        self.file.seek(0)
        data = self.file.read(40)
        format = "%shhidfiiiii" % self.file_metadata['endian']
        (self.file_metadata['magic'],
         self.file_metadata['unused'],
         self.file_metadata['header_size'],
         self.file_metadata['start'],
         self.file_metadata['interval'],
         self.file_metadata['num_pts'],
         self.file_metadata['data_type'],
         self.file_metadata['missing'],
         self.file_metadata['usgs_lock'],
         self.file_metadata['id']
         ) = struct.unpack(format, data)
        self.file_metadata['start_timestamp'] = datetime.datetime.isoformat(
            BTL_EPOCH + datetime.timedelta(0, self.file_metadata['start']), ' ')
        if self.file_metadata['data_type'] == 2:
            self.file_metadata['seed_scale_factor'] = 10000
        else:
            self.file_metadata['seed_scale_factor'] = 1
        if print_it:
            print("")
            print("file:        %s" % self.file_metadata['filename'])
            print("magic:       %x" % self.file_metadata['magic'])
            print("unused:      %d" % self.file_metadata['unused'])
            print("header_size: %d" % self.file_metadata['header_size'])
            print("start:       %s" % self.file_metadata['start_timestamp'])
            print("interval:    %g" % self.file_metadata['interval'])
            print("num_pts:     %d" % self.file_metadata['num_pts'])
            if self.file_metadata['data_type'] in list(BTL_TYPE.keys()):
                print("data_type:   %s" % BTL_TYPE[self.file_metadata['data_type']])
            else:
                print("data_type:   %d" % self.file_metadata['data_type'])
            print("missing:     %d" % self.file_metadata['missing'])
            print("usgs_lock:   %d" % self.file_metadata['usgs_lock'])
            print("id:          %d" % self.file_metadata['id'])
            print("")

        return self.file_metadata

    def read_data(self, print_it=False, timestamps=True):
        """
        """
        if print_it:
            if timestamps:
                print_format = "%%s %%%s" % BTL_TYPE_TO_PRINT[self.file_metadata['data_type']]
                # We use the python timedelta class, since it can
                # better represent intervals such as 0.05.  this becomes
                # (0,0,500000) rather than the floating point approximation
                # of 0.05 which prints as 0.050000000000000003.
                seconds = int(self.file_metadata['interval'])
                microseconds = int((self.file_metadata['interval'] - seconds) * 1000000)
                interval = datetime.timedelta(0, seconds, microseconds)
                start_offset = datetime.timedelta(0, self.file_metadata['start'])
            else:
                print_format = "%%%s" % BTL_TYPE_TO_PRINT[self.file_metadata['data_type']]

        pack_format = "%s%s" % (self.file_metadata['endian'], BTL_TYPE_TO_PACK[self.file_metadata['data_type']])
        self.file.seek(self.file_metadata['header_size'])
        self.data = []
        for i in range(self.file_metadata['num_pts']):
            datum = self.file.read(BTL_TYPE_SIZE[self.file_metadata['data_type']])
            if datum == '':
                print("error: read() returned empty string?")
                continue
            if len(datum) != BTL_TYPE_SIZE[self.file_metadata['data_type']]:
                print("error: read() returned unexpected number of bytes")
                continue
            value = struct.unpack(pack_format, datum)
            self.data.append(value[0])
            if print_it:
                if timestamps:
                    print(print_format % (
                        datetime.datetime.isoformat(BTL_EPOCH + start_offset + (i * interval), ' '), value[0]))
                else:
                    print(print_format % (value[0]))

        return self.data

    def get_unix_ms_timestamps(self):
        interval = round(self.file_metadata['interval'], 3)
        timestamps = []
        for i in range(self.file_metadata['num_pts']):
            timestamps.append(int((self.file_metadata['start'] + (i * interval)) * 1000))
        return timestamps

    def get_timestamps(self):
        seconds = int(self.file_metadata['interval'])
        microseconds = int((self.file_metadata['interval'] - seconds) * 1000000)
        interval = datetime.timedelta(0, seconds, microseconds)
        start_offset = datetime.timedelta(0, self.file_metadata['start'])
        timestamps = []
        for i in range(self.file_metadata['num_pts']):
            timestamps.append(datetime.datetime.isoformat(BTL_EPOCH + start_offset + (i * interval)))
        return timestamps

    def get_datetime_timestamps(self):
        seconds = int(self.file_metadata['interval'])
        microseconds = int((self.file_metadata['interval'] - seconds) * 1000000)
        interval = datetime.timedelta(0, seconds, microseconds)
        start_offset = datetime.timedelta(0, self.file_metadata['start'])
        timestamps = []
        for i in range(self.file_metadata['num_pts']):
            timestamps.append(BTL_EPOCH + start_offset + (i * interval))
        return timestamps
