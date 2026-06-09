import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from chronos import Chronos2Pipeline
from dataset_module import dataset_module
from add_wind_noise import add_wind_noise

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

# Parameters
horizon = 12
contexts = [1024, 2048]
quantiles = [0.5]

# ----- Dataset module -----
cols, df, _, _ = dataset_module()
split = pd.to_datetime('2023-07-01 00:00:00')
df_test = df[df['TmStamp'] > split].reset_index(drop=True)

# Add wind noise
df_fcst = add_wind_noise(df_test)

# Teacher model inference
pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda')

for context in contexts:
    y_true, y_pred_p50 = [], []

    for i in range(context, len(df_test)-horizon+1):
        # Context df
        context_df = df[df['TmStamp'] < df_test['TmStamp'].iloc[i]].tail(context)
        
        # Future df
        future_df = df_fcst.iloc[i:i+horizon][['id','TmStamp'] + cols]
        
        # Generate predictions with covariates
        pred_df = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,    # Number of steps to forecast
            quantile_levels = quantiles,  # Quantile for probabilistic forecast
            id_column='id',               # Column identifying different time series
            timestamp_column='TmStamp',   # Column with datetime information
            target='H',                   # Column with time series values to predict
        )

        # Collect sequences
        y_true.append(df_test.iloc[i:i+horizon]['H'].values)
        y_pred_p50.append(pred_df['0.5'].values)

    y_true, y_pred_p50 = np.array(y_true), np.array(y_pred_p50)
    
    # Test error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_p50))
    rmse_arr = np.sqrt(mean_squared_error(y_true, y_pred_p50, multioutput='raw_values'))
    print(f'Test RMSE: {rmse:.3f}')
    print('Test RMSE per output:', ', '.join([f'{v:.3f}' for v in rmse_arr]))
    
    mae = mean_absolute_error(y_true, y_pred_p50)
    mae_arr = mean_absolute_error(y_true, y_pred_p50, multioutput='raw_values')
    print(f'Test MAE: {mae:.3f}')
    print('Test MAE per output:', ', '.join([f'{v:.3f}' for v in mae_arr]))

    r2 = r2_score(y_true, y_pred_p50)
    r2_arr = r2_score(y_true, y_pred_p50, multioutput='raw_values')
    print(f'Test R2: {r2:.3f}')
    print('Test R2 per output:', ', '.join([f'{v:.3f}' for v in r2_arr]))