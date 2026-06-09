import numpy as np
import pandas as pd

def dataset_module():
    cols = ['u','u_e','u_n']
    
    # Load dataset
    df = pd.read_csv('M01_data.csv')
    df['TmStamp'] = pd.to_datetime(df['TmStamp'], format='mixed')
    df = df.rename(columns={'WSPD':'u'})
    df['id'] = 'series_1'

    # Add east and north components
    df['rad'] = np.pi/180 * ((630 - df['WDIR']) % 360)
    df['u_e'] = df['u'] * np.cos(df['rad'])
    df['u_n'] = df['u'] * np.sin(df['rad'])

    # Train-Test-Validation split
    split1 = pd.to_datetime('2004-07-01 00:00:00')
    split2 = pd.to_datetime('2023-07-01 00:00:00')
    df = df[['id','TmStamp','H'] + cols]
    df_val = df[df['TmStamp'] < split1].reset_index(drop=True)
    df_train = df[(df['TmStamp'] > split1) & (df['TmStamp'] < split2)].reset_index(drop=True)

    return cols, df, df_train, df_val