import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DataConfig, ModelConfig, TrainConfig
from dataset_module import dataset_module
from student_module import student_module
from datetime import datetime

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
df_fcst = pd.read_csv('Sikorsky_fcst.csv')
df_fcst = df_fcst.rename(columns={'WSPD':'u'})
df_fcst['TmStamp'] = pd.to_datetime(df_fcst['TmStamp'], format='mixed')

# Parameters
context, horizon = 1024, 12

# ----- Dataset module -----
cols, df, df_train, df_val = dataset_module()
split = pd.to_datetime('2024-06-30 23:59:00')
df_test = df[df['TmStamp'] > split].reset_index(drop=True)
data_cfg = DataConfig(df_train=df_train, df_val=df_val, cols=cols)
model_cfg = ModelConfig(context=context, horizon=horizon, input_dim=len(cols)+1)

# Student model inference
log_file = 'student_inference_time.txt'
open(log_file, 'w').close()
start_time = datetime.now()

y_true, y_pred_p50 = [], []
y_pred_p80L, y_pred_p80U = [], []
y_pred_p95L, y_pred_p95U = [], []

for q in ['p50','p80L','p80U','p95L','p95U']:
    # ----- Teacher (Chronos) module -----
    train_preds = np.load('train_preds_' + q + '.npy')
    val_preds = np.load('val_preds_' + q + '.npy')
    train_cfg = TrainConfig(train_preds, val_preds)

    # ----- Student (TCN encoder + decoder) module -----
    student, _ = student_module(data_cfg, model_cfg, train_cfg)
    student_ckpt = torch.load('best_student_' + q + '.pt')
    student.load_state_dict(student_ckpt['model_state_dict'])

    student.eval()
    for i in range(len(df_fcst) - horizon + 1):
        context_df = df[df['TmStamp'] < df_fcst['TmStamp'].iloc[i]].tail(context)
        future_df = df_fcst.iloc[i:i+horizon]
        x_context = torch.tensor(context_df[['H'] + cols].values, dtype=torch.float32, device='cuda').unsqueeze(0)
        x_future = torch.tensor(future_df[cols].values, dtype=torch.float32, device='cuda').unsqueeze(0)
        with torch.no_grad():
            pred = student(x_context, x_future)
        
        # Collect sequences
        if q == 'p80L':
            y_pred_p80L.append(pred.squeeze(0).cpu().numpy())
        elif q == 'p80U':
            y_pred_p80U.append(pred.squeeze(0).cpu().numpy())
        elif q == 'p95L':
            y_pred_p95L.append(pred.squeeze(0).cpu().numpy())
        elif q == 'p95U':
            y_pred_p95U.append(pred.squeeze(0).cpu().numpy())
        else:
            y_true.append(df_test.iloc[i:i+horizon]['H'].values)
            y_pred_p50.append(pred.squeeze(0).cpu().numpy())

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