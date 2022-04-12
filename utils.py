import pandas as pd
import numpy as np
from h3 import h3


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def clean(df):
    print(df.info())
    df["timestamp"] = pd.to_datetime(df["pickup_datetime"])
    df = df.sort_values("timestamp")
    print("Min timestamp:")
    print(df["timestamp"].min())
    print("Max timestamp:")
    print(df["timestamp"].max())
    # Basic cleaning
    df["haversine_distance"] = haversine(df["pickup_latitude"], df["pickup_longitude"], df["dropoff_latitude"],
                                         df["dropoff_longitude"])
    # Filter "weird" trips
    df = df[df["haversine_distance"] < 25]
    df = df[df["trip_duration"] < 3600 * 2]

    # NYC borders from here https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial
    df = df[df['pickup_longitude'] <= -73.75]
    df = df[df['pickup_longitude'] >= -74.03]
    df = df[df['pickup_latitude'] <= 40.85]
    df = df[df['pickup_latitude'] >= 40.63]
    df = df[df['dropoff_longitude'] <= -73.75]
    df = df[df['dropoff_longitude'] >= -74.03]
    df = df[df['dropoff_latitude'] <= 40.85]
    df = df[df['dropoff_latitude'] >= 40.63]

    # Time features
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour

    print(df.count())

    return df


def preprocess(df):
    h3_resolutions = [4, 5, 6, 7, 8, 9, 10, 11]

    for h3_res in h3_resolutions:
        df[f"src_h3_{h3_res}"] = [h3.geo_to_h3(x, y, h3_res) for x, y in
                                  zip(df["pickup_latitude"], df["pickup_longitude"])]
        df[f"dst_h3_{h3_res}"] = [h3.geo_to_h3(x, y, h3_res) for x, y in
                                  zip(df["dropoff_latitude"], df["dropoff_longitude"])]
        print(f"{df[f'src_h3_{h3_res}'].nunique()} unique src cells with resolution {h3_res}")
        print(f"{df[f'dst_h3_{h3_res}'].nunique()} unique dst cells with resolution {h3_res}")

    hash_vocab_size = {"src": {}, "dst": {}}

    unique_values = {"src": {}, "dst": {}}

    for point in ["src", "dst"]:
        for h3_res in h3_resolutions:
            print(f"Resolution: {h3_res}")
            df[f"h3_hash_index_{point}_{h3_res}"], uniques = pd.factorize(df[f"{point}_h3_{h3_res}"])

            hash_vocab_size[point][h3_res] = df[f"h3_hash_index_{point}_{h3_res}"].nunique()
            unique_values[point][h3_res] = uniques

            print(f"Vocab size {point}: {hash_vocab_size[point][h3_res]}")

    return df, hash_vocab_size, unique_values
