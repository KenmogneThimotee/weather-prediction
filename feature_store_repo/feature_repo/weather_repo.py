# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String
from feast import FeatureStore
import os 
store = FeatureStore()
# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
city = Entity(name="city", join_keys=["City"])

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
data_path = os.path.join(os.getcwd(), "data/weather_data.pq")
print("Data path : ", data_path)
weather_data_source = FileSource(
    name="weather_data",
    path=data_path,
    timestamp_field="datetime"
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
weather_data_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="weather",
    entities=[city],
    ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="Country", dtype=String),
        Field(name="Latitude", dtype=Float32),
        Field(name="Longitude", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="temperature", dtype=Float32),
        Field(name="pressure", dtype=Float32),
        Field(name="wind_direction", dtype=Float32),
        Field(name="wind_speed", dtype=Float32),
    ],
    online=True,
    source=weather_data_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "weather_data"},
)

city_weather_data = FeatureService(
    name="city_weather_data", features=[weather_data_fv]
)

# Defines a way to push data (to be available offline, online or both) into Feast.
weather_data_push_source = PushSource(
    name="weather_data_push_source",
    batch_source=weather_data_source,
)

# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
weather_fresh_fv = FeatureView(
    name="driver_hourly_stats_fresh",
    entities=[city],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Country", dtype=String),
        Field(name="Latitude", dtype=Float32),
        Field(name="Longitude", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="temperature", dtype=Float32),
        Field(name="pressure", dtype=Float32),
        Field(name="wind_direction", dtype=Float32),
        Field(name="wind_speed", dtype=Float32),
    ],
    online=True,
    source=weather_data_push_source,  # Changed from above
    tags={"team": "driver_performance"},
)