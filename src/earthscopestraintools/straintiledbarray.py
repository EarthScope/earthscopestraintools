import datetime
import tiledb
import numpy as np
import pandas as pd
import json
import logging
from typing import Union

logger = logging.getLogger(__name__)


def str_to_unix_ms(time_string: str):
    # convert string into unix ms
    return np.datetime64(time_string, "ms").astype(int)


class StrainTiledbArray:
    def __init__(self, uri, period=None):
        self.set_default_ctx()
        self.uri = uri
        self.period = period  # period between samples in seconds

    def default_config(self):
        # return a tiledb config
        config = tiledb.Config()
        config["vfs.s3.region"] = "us-east-2"
        config["vfs.s3.scheme"] = "https"
        config["vfs.s3.endpoint_override"] = ""
        config["vfs.s3.use_virtual_addressing"] = "true"
        config["sm.consolidation.mode"] = "fragment_meta"
        config["sm.vacuum.mode"] = "fragment_meta"
        return config

    def set_default_ctx(self):
        config = self.default_config()
        self.ctx = tiledb.Ctx(config=config)

    def get_schema(self):
        filters1 = tiledb.FilterList([tiledb.ZstdFilter(level=7)])
        filters2 = tiledb.FilterList(
            [tiledb.ByteShuffleFilter(), tiledb.ZstdFilter(level=7)]
        )
        filters3 = tiledb.FilterList(
            [tiledb.BitWidthReductionFilter(), tiledb.ZstdFilter(level=7)]
        )
        filters4 = tiledb.FilterList(
            [tiledb.DoubleDeltaFilter(), tiledb.ZstdFilter(level=7)]
        )
        filters5 = tiledb.FilterList(
            [tiledb.FloatScaleFilter(1e-6, 0, bytewidth=8), tiledb.ZstdFilter(level=7)]
        )
        filters6 = tiledb.FilterList(
            [
                tiledb.PositiveDeltaFilter(),
                tiledb.BitWidthReductionFilter(),
                tiledb.ZstdFilter(level=7),
            ]
        )

        ## time dimension with micro-second precision and 24 hour tiles, domain 1970 to 2100
        d0 = tiledb.Dim(name="data_type", dtype="ascii", filters=filters1)
        d1 = tiledb.Dim(name="timeseries", dtype="ascii", filters=filters1)
        d2 = tiledb.Dim(
            name="time",
            domain=(0, 4102444800000),
            tile=86400000,
            dtype=np.int64,
            filters=filters4,
        )
        dom = tiledb.Domain(d0, d1, d2)

        dom = tiledb.Domain(d0, d1, d2)

        a0 = tiledb.Attr(name="data", dtype=np.float64, filters=filters1)
        a1 = tiledb.Attr(name="quality", dtype="ascii", var=True, filters=filters1)
        a2 = tiledb.Attr(name="level", dtype="ascii", var=True, filters=filters1)
        a3 = tiledb.Attr(name="version", dtype=np.int64, filters=filters1)
        attrs = [a0, a1, a2, a3]

        # coords_filters = filters1
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=attrs,
            cell_order="row-major",
            tile_order="row-major",
            capacity=100000,
            offsets_filters=filters6,
        )

        return schema

    def get_schema_from_s3(self):
        s3_schema_uri = "s3://tiledb-strain/STRAIN_SCHEMA.tdb"
        config = tiledb.Config()
        config["vfs.s3.region"] = "us-east-2"
        config["vfs.s3.scheme"] = "https"
        config["vfs.s3.endpoint_override"] = ""
        config["vfs.s3.use_virtual_addressing"] = "true"
        config["sm.consolidation.mode"] = "fragment_meta"
        config["sm.vacuum.mode"] = "fragment_meta"

        with tiledb.open(s3_schema_uri, "r", config=config) as A:
            schema = A.schema
        return schema

    def create(self, schema_source="s3"):
        if schema_source == "s3":
            self.schema = self.get_schema_from_s3()
        else:
            self.schema = self.get_schema()
        try:
            tiledb.Array.create(self.uri, self.schema, ctx=self.ctx)
            with tiledb.Array(self.uri, "w", ctx=self.ctx) as A:
                A.meta["version"] = "3.5"
            logger.info(f"Created array at {self.uri}")
        except tiledb.TileDBError as e:
            logger.warning(e)

    def delete(self):
        try:
            tiledb.remove(self.uri, ctx=self.ctx)
            print("Deleted ", self.uri)
        except tiledb.TileDBError as e:
            print(e)

    def consolidate_fragment_meta(self):
        config = self.ctx.config()
        config["sm.consolidation.mode"] = "fragment_meta"
        tiledb.consolidate(self.uri, config=config)
        logger.info("consolidated fragment_meta")

    def consolidate_array_meta(self):
        config = self.ctx.config()
        config["sm.consolidation.mode"] = "array_meta"
        tiledb.consolidate(self.uri, config=config)
        logger.info("consolidated array_meta")

    def consolidate_fragments(self):
        config = self.ctx.config()
        config["sm.consolidation.mode"] = "fragments"
        tiledb.consolidate(self.uri, config=config)
        logger.info("consolidated fragments")

    def vacuum_fragment_meta(self):
        config = self.ctx.config()
        config["sm.vacuum.mode"] = "fragment_meta"
        tiledb.vacuum(self.uri, config=config)
        logger.info("vacuumed fragment_meta")

    def vacuum_array_meta(self):
        config = self.ctx.config()
        config["sm.vacuum.mode"] = "array_meta"
        tiledb.vacuum(self.uri, config=config)
        logger.info("vacuumed array_meta")

    def vacuum_fragments(self):
        config = self.ctx.config()
        config["sm.vacuum.mode"] = "fragments"
        tiledb.vacuum(self.uri, config=config)
        logger.info("vacuumed fragments")

    def cleanup(self):
        self.consolidate_array_meta()
        self.vacuum_array_meta()
        self.consolidate_fragment_meta()
        self.vacuum_fragment_meta()
        self.consolidate_fragments()
        self.vacuum_fragments()

    def get_nonempty_domain(self):
        with tiledb.open(self.uri, "r", ctx=self.ctx) as A:
            return A.nonempty_domain()[2][0], A.nonempty_domain()[2][1]

    def get_data_types(self):
        with tiledb.open(self.uri, "r", ctx=self.ctx) as A:
            return json.loads(A.meta["dimensions"])["data_types"]

    def get_timeseries(self):
        with tiledb.open(self.uri, "r", ctx=self.ctx) as A:
            return json.loads(A.meta["dimensions"])["timeseries"]

    def print_schema(self):
        with tiledb.open(self.uri, "r", ctx=self.ctx) as A:
            print(A.schema)


class Reader:
    def __init__(self, array: StrainTiledbArray):
        self.array = array

    def to_df(
        self,
        data_types: Union[list, str],
        timeseries: Union[list, str],
        attrs: Union[list, str],
        start_ts: int = None,
        end_ts: int = None,
        start_str: str = None,
        end_str: str = None,
        start_dt: datetime.datetime = None,
        end_dt: datetime.datetime = None,
        reindex=True,
        print_array_range=False,
    ):
        # generic read function. requires all query parameters, accepts strings or lists
        # reindex handles one or more data_types
        # reindex does not handle more than one timeseries or attr, so should be set to False

        try:
            if not start_ts:
                if start_str:
                    start_ts = str_to_unix_ms(start_str)
                elif start_dt:
                    start_ts = int(
                        start_dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                        * 1000
                    )
                else:
                    logger.error("No start time provided for read")
            if not end_ts:
                if end_str:
                    end_ts = str_to_unix_ms(end_str)
                elif start_dt:
                    end_ts = int(
                        end_dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000
                    )
                else:
                    logger.error("No start time provided for read")
            logger.info(f"Query range {start_ts} to {end_ts}")
            with tiledb.open(self.array.uri, "r", ctx=self.array.ctx) as A:
                if print_array_range:
                    print(
                        "Array date range: %s to %s"
                        % (A.nonempty_domain()[2][0], A.nonempty_domain()[2][1])
                    )
                index_col = ["data_type", "timeseries", "time"]
                data_types = [data_types] if isinstance(data_types, str) else data_types
                timeseries = [timeseries] if isinstance(timeseries, str) else timeseries
                attrs = [attrs] if isinstance(attrs, str) else attrs
                df = A.query(index_col=index_col, attrs=attrs).df[
                    data_types, timeseries, start_ts:end_ts
                ]
                df.index = df.index.set_levels(
                    pd.to_datetime(df.index.levels[2], unit="ms"), level=2
                )
            if reindex:
                df = self.reindex_df(df, columns=data_types, attr=attrs[0])
            if not isinstance(df, pd.DataFrame):
                df = df.to_frame()
            self.check_query_result(df, start_ts, end_ts)
            return df

        except (IndexError, KeyError, tiledb.TileDBError) as e:
            logger.error(e)
            logger.error("No data found matching query parameters")
            return pd.DataFrame()

    def reindex_df(self, df: pd.DataFrame, columns: list = [], attr="data"):
        # removes a multi-index and makes each data_type a column
        data_types = columns
        for data_type in data_types:
            df_data_type = df.xs(data_type, level="data_type")[attr].droplevel(level=0)
            df_data_type.name = data_type
            if data_type == data_types[0]:
                df2 = df_data_type
            else:
                df2 = pd.concat([df2, df_data_type], axis=1)
        return df2

    def check_query_result(self, df, start, end):
        if self.array.period is not None:
            expected_samples = int((end - start) / 1000 / self.array.period)
            # expected_samples = int((str_to_unix_ms(end) - str_to_unix_ms(start)) / 1000 / self.array.period)
            # print("expected samples:", expected_samples)
            if len(df) >= expected_samples:
                logger.info(
                    f"Query complete, expected {expected_samples} and returned {len(df)}"
                )
            else:
                logger.info(
                    f"Query incomplete, expected {expected_samples} and returned {len(df)}"
                )
        else:
            logger.info(f"Cannot check query completess without array period")


class Writer:
    def __init__(self, array: StrainTiledbArray):
        self.array = array

    def write_df_to_tiledb(self, df: pd.DataFrame):

        mode = "append"
        tiledb.from_pandas(
            uri=self.array.uri,
            dataframe=df,
            index_dims=["data_type", "timeseries", "time"],
            mode=mode,
            ctx=self.array.ctx,
        )
        # update the string dimension metadata
        data_type = df["data_type"].unique()
        timeseries = df["timeseries"].unique()
        if type(data_type) == str:
            data_type = [data_type]
        if type(timeseries) == str:
            timeseries = [timeseries]
        with tiledb.open(self.array.uri, "r", ctx=self.array.ctx) as A:
            try:
                dimension_json = A.meta["dimensions"]
            except KeyError:
                dimension_json = '{"data_types":[], "timeseries":[]}'

            dimension_dict = json.loads(dimension_json)
            # print(dimension_dict)
            for item in data_type:
                if item not in dimension_dict["data_types"]:
                    dimension_dict["data_types"].append(item)
            for item in timeseries:
                if item not in dimension_dict["timeseries"]:
                    dimension_dict["timeseries"].append(item)

            # print(dimension_dict)
            with tiledb.open(self.array.uri, "w", ctx=self.array.ctx) as A:
                A.meta["dimensions"] = json.dumps(dimension_dict)

    def df_2_tiledb(
        self,
        df: pd.DataFrame,
        data_types: list,
        timeseries: str,
        level: str,
        quality_df: pd.DataFrame = None,
        print_it: bool = False,
    ):
        """
        prepares a dataframe to write to array schema
        df - dataframe with time index, columns are timeseries data (one column per data_type)
        uri - string. which tiledb array to write to
        data_types - list of strings. data_types to map columns into.  ie CH0, 2Ene, pressure, time_index
        timeseries - string. name of timeseries.  counts, microstrain, offset_c, tide_c, atmp_c, atmp, pore, mjd, doy
        level - 2char string.  '0a', '1a', '2a', '2b'
        quality_df- dataframe, optional.  if not included, quality flags will be set to 'g'
        print_it - bool, optional.  show the constructed dataframe as it is being written to tiledb.
        """
        df_buffer = pd.DataFrame(
            columns=[
                "data_type",
                "timeseries",
                "time",
                "data",
                "level",
                "quality",
                "version",
            ]
        )
        if quality_df is None:
            quality_df = pd.DataFrame(index=df.index)
            for ch in data_types:
                quality_df[ch] = "g"
        for ch in data_types:
            data = df[ch].values
            # convert datetimeindex to unix ms
            timestamps = df.index.astype(int) / 10 ** 6
            version = int(datetime.datetime.now().strftime("%Y%j%H%M%S"))
            quality = quality_df[ch].values

            d = {
                "data_type": ch,
                "timeseries": timeseries,
                "time": timestamps,
                "data": data,
                "level": level,
                "quality": quality,
                "version": version,
            }
            ch_df = pd.DataFrame(data=d)
            # ch_df.loc[ch_df['data'] == 999999, 'quality'] = 'm'
            df_buffer = pd.concat([df_buffer, ch_df], axis=0).reset_index(drop=True)
            df_buffer["time"] = df_buffer["time"].astype(np.int64)
            df_buffer["data"] = df_buffer["data"].astype(np.float64)
            df_buffer["version"] = df_buffer["version"].astype(np.int64)

        if print_it:
            print(df_buffer)
        self.write_df_to_tiledb(df_buffer)


# import datetime
# import tiledb
# import numpy as np
# import pandas as pd
# import logging
# from typing import Union
#
# logger = logging.getLogger(__name__)
#
#
# def str_to_unix_ms(time_string: str):
#     #convert string into unix ms
#     return np.datetime64(time_string, 'ms').astype(int)
#
# class StrainTiledbArray:
#     def __init__(self, uri, period=None, location='s3'):
#         if location == 's3':
#             self.set_s3_ctx()
#         elif location == 'local':
#             self.set_local_ctx()
#         self.uri = uri
#         self.period = period  # period between samples in seconds
#
#     def set_local_ctx(self):
#         # default ctx
#         config = tiledb.Config()
#         try:
#             tiledb.default_ctx(config)
#         except tiledb.TileDBError as e:
#             print(e)
#         config["sm.consolidation.mode"] = "fragment_meta"
#         config["sm.vacuum.mode"] = "fragment_meta"
#         self.ctx = tiledb.Ctx(config=config)
#         return self.ctx
#
#     def set_s3_ctx(self):
#         config = tiledb.Config()
#         config["vfs.s3.region"] = "us-east-2"
#         config["vfs.s3.scheme"] = "https"
#         config["vfs.s3.endpoint_override"] = ""
#         config["vfs.s3.use_virtual_addressing"] = "true"
#         config["sm.consolidation.mode"] = "fragment_meta"
#         config["sm.vacuum.mode"] = "fragment_meta"
#         self.ctx = tiledb.Ctx(config=config)
#
#     def get_schema(self):
#         filters1 = tiledb.FilterList([tiledb.ZstdFilter(level=7)])
#         filters2 = tiledb.FilterList([tiledb.ByteShuffleFilter(), tiledb.ZstdFilter(level=7)])
#         filters3 = tiledb.FilterList([tiledb.BitWidthReductionFilter(), tiledb.ZstdFilter(level=7)])
#         filters4 = tiledb.FilterList([tiledb.DoubleDeltaFilter(), tiledb.ZstdFilter(level=7)])
#         filters5 = tiledb.FilterList([tiledb.FloatScaleFilter(1e-6, 0, bytewidth=8), tiledb.ZstdFilter(level=7)])
#         filters6 = tiledb.FilterList([tiledb.PositiveDeltaFilter(), tiledb.BitWidthReductionFilter(),
#                                       tiledb.ZstdFilter(level=7)])
#
#         ## time dimension with micro-second precision and 24 hour tiles, domain 1970 to 2100
#         d0 = tiledb.Dim(name="data_type", dtype="ascii", filters=filters1)
#         d1 = tiledb.Dim(name="timeseries", dtype="ascii", filters=filters1)
#         d2 = tiledb.Dim(name="time", domain=(0, 4102444800000), tile=86400000, dtype=np.int64, filters=filters4)
#         dom = tiledb.Domain(d0, d1, d2)
#
#         dom = tiledb.Domain(d0, d1, d2)
#
#         a0 = tiledb.Attr(name="data", dtype=np.float64, filters=filters1)
#         a1 = tiledb.Attr(name="quality", dtype="ascii", var=True, filters=filters1)
#         a2 = tiledb.Attr(name="level", dtype="ascii", var=True, filters=filters1)
#         a3 = tiledb.Attr(name="version", dtype=np.int64, filters=filters1)
#         attrs = [a0, a1, a2, a3]
#
#         # coords_filters = filters1
#         schema = tiledb.ArraySchema(domain=dom,
#                                     sparse=True,
#                                     attrs=attrs,
#                                     cell_order='row-major',
#                                     tile_order='row-major',
#                                     capacity=100000,
#                                     offsets_filters=filters6)
#
#         return schema
#
#     def get_schema_from_s3(self):
#         s3_schema_uri = "s3://tiledb-strain/STRAIN_SCHEMA.tdb"
#         config = tiledb.Config()
#         config["vfs.s3.region"] = "us-east-2"
#         config["vfs.s3.scheme"] = "https"
#         config["vfs.s3.endpoint_override"] = ""
#         config["vfs.s3.use_virtual_addressing"] = "true"
#         config["sm.consolidation.mode"] = "fragment_meta"
#         config["sm.vacuum.mode"] = "fragment_meta"
#
#         with tiledb.open(s3_schema_uri, 'r', config=config) as A:
#             schema = A.schema
#         return schema
#
#     def create(self, schema_source='s3'):
#         if schema_source == 's3':
#             self.schema = self.get_schema_from_s3()
#         else:
#             self.schema = self.get_schema()
#         try:
#             tiledb.Array.create(self.uri, self.schema, ctx=self.ctx)
#             with tiledb.Array(self.uri, "w", ctx=self.ctx) as A:
#                 A.meta["version"] = '3.5'
#             logger.info(f'Created array at {self.uri}')
#         except tiledb.TileDBError as e:
#             logger.warning(e)
#
#     def delete(self):
#         try:
#             tiledb.remove(self.uri, ctx=self.ctx)
#             print("Deleted ", self.uri)
#         except tiledb.TileDBError as e:
#             print(e)
#
#     def consolidate_fragment_meta(self):
#         config = self.ctx.config()
#         config["sm.consolidation.mode"] = "fragment_meta"
#         tiledb.consolidate(self.uri, config=config)
#         logger.info("consolidated fragment_meta")
#
#     def consolidate_array_meta(self):
#         config = self.ctx.config()
#         config["sm.consolidation.mode"] = "array_meta"
#         tiledb.consolidate(self.uri, config=config)
#         logger.info("consolidated array_meta")
#
#     def consolidate_fragments(self):
#         config = self.ctx.config()
#         config["sm.consolidation.mode"] = "fragments"
#         tiledb.consolidate(self.uri, config=config)
#         logger.info("consolidated fragments")
#
#     def vacuum_fragment_meta(self):
#         config = self.ctx.config()
#         config["sm.vacuum.mode"] = "fragment_meta"
#         tiledb.vacuum(self.uri, config=config)
#         logger.info("vacuumed fragment_meta")
#
#     def vacuum_array_meta(self):
#         config = self.ctx.config()
#         config["sm.vacuum.mode"] = "array_meta"
#         tiledb.vacuum(self.uri, config=config)
#         logger.info("vacuumed array_meta")
#
#     def vacuum_fragments(self):
#         config = self.ctx.config()
#         config["sm.vacuum.mode"] = "fragments"
#         tiledb.vacuum(self.uri, config=config)
#         logger.info("vacuumed fragments")
#
#     def cleanup(self):
#         self.consolidate_array_meta()
#         self.vacuum_array_meta()
#         self.consolidate_fragment_meta()
#         self.vacuum_fragment_meta()
#         self.consolidate_fragments()
#         self.vacuum_fragments()
#
#     def get_nonempty_domain(self):
#         with tiledb.open(self.uri, 'r', ctx=self.ctx) as A:
#             return A.nonempty_domain()[2][0], A.nonempty_domain()[2][1]
#
#     def print_schema(self):
#         with tiledb.open(self.uri, 'r', ctx=self.ctx) as A:
#             print(A.schema)
#
# class Reader:
#     def __init__(self,
#              array: StrainTiledbArray):
#         self.array = array
#
#     def to_df(self,
#                    data_types: Union[list, str],
#                    timeseries: Union[list, str],
#                    attrs: Union[list, str],
#                    start: str,
#                    end: str,
#                    reindex=True,
#                    print_array_range=False):
#         # generic read function. requires all query parameters, accepts strings or lists
#         # reindex handles one or more data_types
#         # reindex does not handle more than one timeseries or attr, so should be set to False
#
#         try:
#             with tiledb.open(self.array.uri, 'r', ctx=self.array.ctx) as A:
#                 if print_array_range:
#                     print("Array date range: %s to %s" % (A.nonempty_domain()[2][0], A.nonempty_domain()[2][1]))
#                 index_col = ['data_type', 'timeseries', 'time']
#                 data_types = [data_types] if isinstance(data_types, str) else data_types
#                 timeseries = [timeseries] if isinstance(timeseries, str) else timeseries
#                 attrs = [attrs] if isinstance(attrs, str) else attrs
#                 df = A.query(index_col=index_col, attrs=attrs).df[data_types, timeseries,
#                      str_to_unix_ms(start):str_to_unix_ms(end)]
#                 df.index = df.index.set_levels(pd.to_datetime(df.index.levels[2], unit='ms'), level=2)
#             if reindex:
#                 df = self.reindex_df(df, columns=data_types, attr=attrs[0])
#             if not isinstance(df, pd.DataFrame):
#                 df = df.to_frame()
#             self.check_query_result(df, start, end)
#             return df
#
#         except (IndexError, KeyError) as e:
#             logger.error('No data found matching query parameters')
#
#     def reindex_df(self, df: pd.DataFrame, columns: list = [], attr='data'):
#         # removes a multi-index and makes each data_type a column
#         data_types = columns
#         for data_type in data_types:
#             df_data_type = df.xs(data_type, level='data_type')[attr].droplevel(level=0)
#             df_data_type.name = data_type
#             if data_type == data_types[0]:
#                 df2 = df_data_type
#             else:
#                 df2 = pd.concat([df2, df_data_type], axis=1)
#         return df2
#
#     def check_query_result(self, df, start, end):
#         if self.array.period is not None:
#             #expected_samples = int((end - start).total_seconds() / self.array.period)
#             expected_samples = int((str_to_unix_ms(end) - str_to_unix_ms(start)) / 1000 / self.array.period)
#             # print("expected samples:", expected_samples)
#             if len(df) >= expected_samples:
#                 logger.info(f'Query complete, expected {expected_samples} and returned {len(df)}')
#             else:
#                 logger.info(f'Query incomplete, expected {expected_samples} and returned {len(df)}')
#         else:
#             logger.info(f'Cannot check query completess without array period')
#
#
# class Writer:
#     def __init__(self,
#                  array: StrainTiledbArray):
#         self.array = array
#
#     def write_df_to_tiledb(self,
#                            df: pd.DataFrame):
#         if tiledb.array_exists(self.array.uri):  # doesnt work in s3
#             mode = "append"
#         else:
#             mode = "ingest"
#         tiledb.from_pandas(uri=self.array.uri,
#                            dataframe=df,
#                            index_dims=['data_type', 'timeseries', 'time'],
#                            mode=mode,
#                            ctx=self.array.ctx
#                            )
#
#     def df_2_tiledb(self,
#                     df: pd.DataFrame,
#                     data_types: list,
#                     timeseries: str,
#                     level: str,
#                     quality_df: pd.DataFrame = None,
#                     print_it: bool = False):
#         """
#         prepares a dataframe to write to array schema
#         df - dataframe with time index, columns are timeseries data (one column per data_type)
#         uri - string. which tiledb array to write to
#         data_types - list of strings. data_types to map columns into.  ie CH0, 2Ene, pressure, time_index
#         timeseries - string. name of timeseries.  counts, microstrain, offset_c, tide_c, atmp_c, atmp, pore, mjd, doy
#         level - 2char string.  '0a', '1a', '2a', '2b'
#         quality_df- dataframe, optional.  if not included, quality flags will be set to 'g'
#         print_it - bool, optional.  show the constructed dataframe as it is being written to tiledb.
#         """
#         df_buffer = pd.DataFrame(columns=['data_type', 'timeseries', 'time',
#                                           'data', 'level', 'quality', 'version'])
#         if quality_df is None:
#             quality_df = pd.DataFrame(index=df.index)
#             for ch in data_types:
#                 quality_df[ch] = 'g'
#         for ch in data_types:
#             data = df[ch].values
#             timestamps = df.index
#             version = int(datetime.datetime.now().strftime("%Y%j%H%M%S"))
#             quality = quality_df[ch].values
#
#             d = {'data_type': ch,
#                  'timeseries': timeseries,
#                  'time': timestamps,
#                  'data': data,
#                  'level': level,
#                  'quality': quality,
#                  'version': version}
#             ch_df = pd.DataFrame(data=d)
#             # ch_df.loc[ch_df['data'] == 999999, 'quality'] = 'm'
#             df_buffer = pd.concat([df_buffer, ch_df], axis=0).reset_index(drop=True)
#             df_buffer['data'] = df_buffer['data'].astype(np.float64)
#             df_buffer['version'] = df_buffer['version'].astype(np.int64)
#         if print_it:
#             print(df_buffer)
#         self.write_df_to_tiledb(df_buffer)
