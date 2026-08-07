import os
import numpy as np
import pandas as pd
from config import *

STATION_INFO = {
    'USW00014707': {'lat': 41.3275, 'lon': -72.0494, 'elev': 3.0 * 0.3048},
    'USW00014740': {'lat': 41.9375, 'lon': -72.6819, 'elev': 53.3 * 0.3048},
    'USW00014758': {'lat': 41.2639, 'lon': -72.8872, 'elev': 0.9 * 0.3048},
    'USW00054734': {'lat': 41.3714, 'lon': -73.4828, 'elev': 139.3 * 0.3048},
    'USW00054767': {'lat': 41.7419, 'lon': -72.1836, 'elev': 75.3 * 0.3048},
    'USW00054788': {'lat': 41.5097, 'lon': -72.8278, 'elev': 31.4 * 0.3048},
    'USW00094702': {'lat': 41.1583, 'lon': -73.1289, 'elev': 1.5 * 0.3048},
}

COLUMN_MAP = {
    'DATE': 'date',
    'HourlyPrecipitation': 'prcp',       # mm
    'HourlyDryBulbTemperature': 'temp',  # degC
    'HourlyRelativeHumidity': 'rhum',    # %
    'HourlySeaLevelPressure': 'pres',    # hPa
    'WindEast': 'uwnd',                  # m/s
    'WindNorth': 'vwnd',                 # m/s
}

def load_station(csv_path, start_year):
    df = pd.read_csv(csv_path, low_memory=False)
    df.rename(columns=COLUMN_MAP, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].dt.year >= start_year]
    station = os.path.splitext(os.path.basename(csv_path))[0]
    df['id'] = station
    df['lat'] = STATION_INFO[station]['lat']
    df['lon'] = STATION_INFO[station]['lon']
    df['elev'] = STATION_INFO[station]['elev']
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_time_features(df):
    df['time_idx'] = np.arange(len(df), dtype=np.int32)
    df['month'] = df['date'].dt.month.astype(str)
    df['hour'] = df['date'].dt.hour.astype(str)
    hour = df['date'].dt.hour
    doy = df['date'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return df

def load_all_stations(data_dir):
    dfs = []
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith('.csv'):
            continue
        start_year = 2005 if file[:-4] in SOURCE_IDS else START_YEAR
        df = load_station(os.path.join(data_dir, file), start_year)
        df = add_time_features(df)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.sort_values(['id','time_idx'], inplace=True)
    all_df.reset_index(drop=True, inplace=True)
    return all_df

def split_source_target(all_df):
    source_df = all_df[all_df['id'].isin(SOURCE_IDS)].copy()
    target_df = all_df[all_df['id'].isin(TARGET_IDS)].copy()
    return source_df, target_df

def split_by_year(df, train_end_year=2020, val_end_year=2022):
    train_df = df[df['date'].dt.year <= train_end_year].copy()
    val_df = df[(df['date'].dt.year > train_end_year) & (df['date'].dt.year <= val_end_year)].copy()
    test_df = df[df['date'].dt.year > val_end_year].copy()
    return train_df, val_df, test_df