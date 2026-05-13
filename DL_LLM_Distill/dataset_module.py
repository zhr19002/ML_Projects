import numpy as np
import pandas as pd

def dataset_module():
    cols = ['u','u_e','u_n']
    
    # Load datasets
    df_buoy = pd.read_csv('WLIS_data.csv')
    df_buoy['TmStamp'] = pd.to_datetime(df_buoy['TmStamp'], format='mixed')
    df_buoy['id'] = 'series_1'

    df_stn = pd.read_csv('Sikorsky_data.csv')
    df_stn = df_stn.rename(columns={'WSPD':'u'})
    df_stn['TmStamp'] = pd.to_datetime(df_stn['TmStamp'], format='mixed')
    df_stn['id'] = 'series_1'

    # Add east and north components
    alpha = -13
    df_stn['rad'] = np.pi/180 * ((alpha + 630 - df_stn['WDIR']) % 360)
    df_stn['u_e'] = df_stn['u'] * np.cos(df_stn['rad'])
    df_stn['u_n'] = df_stn['u'] * np.sin(df_stn['rad'])

    # Train-Test-Validation split
    split1 = pd.to_datetime('2023-12-31 23:59:00')
    split2 = pd.to_datetime('2024-06-30 23:59:00')
    df = pd.merge(df_buoy[['id','TmStamp','H']], df_stn[['TmStamp'] + cols], on='TmStamp', how='left')
    df_train = df[df['TmStamp'] < split1].reset_index(drop=True)
    df_val = df[(df['TmStamp'] > split1) & (df['TmStamp'] < split2)].reset_index(drop=True)
    
    return cols, df, df_train, df_val