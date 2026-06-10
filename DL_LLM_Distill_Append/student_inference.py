import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DataConfig, ModelConfig, TrainConfig
from dataset_module import dataset_module
from student_module import student_module

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
df_fcst = pd.read_csv('M01_fcst.csv')
df_fcst['TmStamp'] = pd.to_datetime(df_fcst['TmStamp'], format='mixed')

# Parameters
context, horizon = 1024, 12

# ----- Dataset module -----
cols, df, df_train, df_val = dataset_module()
split = pd.to_datetime('2023-07-01 00:00:00')
df_test = df[df['TmStamp'] > split].reset_index(drop=True)
data_cfg = DataConfig(df_train=df_train, df_val=df_val, cols=cols)
model_cfg = ModelConfig(context=context, horizon=horizon, input_dim=len(cols)+1)

y_true, y_pred_p50 = [], []

# ----- Teacher (Chronos) module -----
train_preds = np.load('train_preds_p50.npy')
val_preds = np.load('val_preds_p50.npy')
train_cfg = TrainConfig(train_preds, val_preds)

# ----- Student (TCN encoder + decoder) module -----
student, _ = student_module(data_cfg, model_cfg, train_cfg)
student_ckpt = torch.load('best_student_p50.pt')
student.load_state_dict(student_ckpt['model_state_dict'])

student.eval()
for i in range(context, len(df_fcst)-horizon+1):
    context_df = df[df['TmStamp'] < df_fcst['TmStamp'].iloc[i]].tail(context)
    future_df = df_fcst.iloc[i:i+horizon]
    x_context = torch.tensor(context_df[['H'] + cols].values, dtype=torch.float32, device='cuda').unsqueeze(0)
    x_future = torch.tensor(future_df[cols].values, dtype=torch.float32, device='cuda').unsqueeze(0)
    with torch.no_grad():
        pred = student(x_context, x_future)
    y_true.append(df_test.iloc[i:i+horizon]['H'].values)
    y_pred_p50.append(pred.squeeze(0).cpu().numpy())

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