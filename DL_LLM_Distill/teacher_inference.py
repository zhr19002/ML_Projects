import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from chronos import Chronos2Pipeline
from dataset_module import dataset_module
from datetime import datetime

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

# Load dataset
df_fcst = pd.read_csv('Sikorsky_fcst.csv')
df_fcst = df_fcst.rename(columns={'WSPD':'u'})
df_fcst['TmStamp'] = pd.to_datetime(df_fcst['TmStamp'], format='mixed')
df_fcst['id'] = 'series_1'

# Parameters
horizon = 12
contexts = [24, 48, 72, 96, 120, 256, 512, 1024, 2048]
quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]

# ----- Dataset module -----
cols, df, _, _ = dataset_module()
split = pd.to_datetime('2024-06-30 23:59:00')
df_test = df[df['TmStamp'] > split].reset_index(drop=True)

# Teacher model inference
pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda')
log_file = 'teacher_inference_time.txt'
open(log_file, 'w').close()

for context in contexts:
    start_time = datetime.now()
    y_true, y_pred_p50 = [], []
    y_pred_p80L, y_pred_p80U = [], []
    y_pred_p95L, y_pred_p95U = [], []

    for i in range(len(df_fcst) - horizon + 1):
        # Context df
        context_df = df[df['TmStamp'] < df_fcst['TmStamp'].iloc[i]].tail(context)
        
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
        y_pred_p95L.append(pred_df['0.025'].values)
        y_pred_p80L.append(pred_df['0.1'].values)
        y_pred_p80U.append(pred_df['0.9'].values)
        y_pred_p95U.append(pred_df['0.975'].values)

    y_true, y_pred_p50 = np.array(y_true), np.array(y_pred_p50)
    y_pred_p80L, y_pred_p80U = np.array(y_pred_p80L), np.array(y_pred_p80U)
    y_pred_p95L, y_pred_p95U = np.array(y_pred_p95L), np.array(y_pred_p95U)
    
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

    # Prediction interval coverage probability (PICP)
    picp_p80 = 100 * np.mean((y_true >= y_pred_p80L) & (y_true <= y_pred_p80U))
    picp_p80_per_output = 100 * np.mean((y_true >= y_pred_p80L) & (y_true <= y_pred_p80U), axis=0)
    print(f'PICP (80% PI): {picp_p80:.1f}')
    print('PICP (80% PI) per output:', ', '.join([f'{p:.1f}' for p in picp_p80_per_output]))

    picp_p95 = 100 * np.mean((y_true >= y_pred_p95L) & (y_true <= y_pred_p95U))
    picp_p95_per_output = 100 * np.mean((y_true >= y_pred_p95L) & (y_true <= y_pred_p95U), axis=0)
    print(f'PICP (95% PI): {picp_p95:.1f}')
    print('PICP (95% PI) per output:', ', '.join([f'{p:.1f}' for p in picp_p95_per_output]))

    # Mean prediction interval width (MPIW)
    mpiw_p80 = np.mean(y_pred_p80U - y_pred_p80L)
    mpiw_p80_per_output = np.mean(y_pred_p80U - y_pred_p80L, axis=0)
    print(f'MPIW (80% PI): {mpiw_p80:.3f}')
    print('MPIW (80% PI) per output:', ', '.join([f'{w:.3f}' for w in mpiw_p80_per_output]))

    mpiw_p95 = np.mean(y_pred_p95U - y_pred_p95L)
    mpiw_p95_per_output = np.mean(y_pred_p95U - y_pred_p95L, axis=0)
    print(f'MPIW (95% PI): {mpiw_p95:.3f}')
    print('MPIW (95% PI) per output:', ', '.join([f'{w:.3f}' for w in mpiw_p95_per_output]))
    
    # Inference time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    with open(log_file, 'a') as f:
        f.write(f'context = {context}\n')
        f.write(f'Start time: {start_time}\n')
        f.write(f'End time: {end_time}\n')
        f.write(f'Duration (seconds): {duration:.2f}\n\n')