import os
import random
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from config import DataConfig, ModelConfig
from dataset_module import dataset_module

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def teacher_module(data_cfg: DataConfig, model_cfg: ModelConfig):
    df_train = data_cfg.df_train
    df_val = data_cfg.df_val
    cols = data_cfg.cols

    context = model_cfg.context
    horizon = model_cfg.horizon

    # ----- Distillation dataset construction -----
    class DistillDataset(torch.utils.data.Dataset):
        def __init__(self, df, context, horizon):
            self.Timestamps = []
            self.X_context = []
            self.X_future = []
            periods = context + horizon
            for i in range(len(df) - periods + 1):
                window = df.iloc[i:i+periods]
                expect = pd.date_range(start=window['TmStamp'].iloc[0], periods=periods, freq='h')
                if not window['TmStamp'].reset_index(drop=True).equals(pd.Series(expect)):
                    continue
                values = window[['H'] + cols].values.astype(np.float32)
                self.Timestamps.append(window['TmStamp'].tolist())
                self.X_context.append(values[:context])
                self.X_future.append(values[context:context+horizon, 1:])

        def __len__(self):
            return len(self.X_context)

        def __getitem__(self, idx):
            return (self.Timestamps[idx],
                    torch.tensor(self.X_context[idx], dtype=torch.float32),
                    torch.tensor(self.X_future[idx], dtype=torch.float32))

    # ----- Teacher (Chronos) predictions -----
    pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda')
    train_ds = DistillDataset(df_train, context, horizon)
    val_ds = DistillDataset(df_val, context, horizon)

    def teacher_prediction(dataset, name):
        preds_p50 = []
        preds_p80L, preds_p80U = [], []
        preds_p95L, preds_p95U = [], []

        for i in range(len(dataset)):
            timestamps, x_context, x_future = dataset[i]

            context_df = pd.DataFrame(x_context.numpy(), columns=['H'] + cols)
            context_df['TmStamp'] = timestamps[:context]
            context_df['id'] = 'series_1'

            future_df = pd.DataFrame(x_future.numpy(), columns=cols)
            future_df['TmStamp'] = timestamps[context:context+horizon]
            future_df['id'] = 'series_1'

            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=horizon,
                quantile_levels=[0.025, 0.1, 0.5, 0.9, 0.975],
                id_column='id',
                timestamp_column='TmStamp',
                target='H')
            
            preds_p95L.append(pred_df['0.025'].values.astype(np.float32))
            preds_p80L.append(pred_df['0.1'].values.astype(np.float32))
            preds_p50.append(pred_df['0.5'].values.astype(np.float32))
            preds_p80U.append(pred_df['0.9'].values.astype(np.float32))
            preds_p95U.append(pred_df['0.975'].values.astype(np.float32))
        
        np.save(name + '_preds_p95L.npy', np.array(preds_p95L))
        np.save(name + '_preds_p80L.npy', np.array(preds_p80L))
        np.save(name + '_preds_p50.npy', np.array(preds_p50))
        np.save(name + '_preds_p80U.npy', np.array(preds_p80U))
        np.save(name + '_preds_p95U.npy', np.array(preds_p95U))
    
    teacher_prediction(train_ds, 'train')
    teacher_prediction(val_ds, 'val')

def main():
    cols, _, df_train, df_val = dataset_module()
    data_cfg = DataConfig(df_train=df_train, df_val=df_val, cols=cols)
    model_cfg = ModelConfig(context=1024, horizon=12, input_dim=len(cols)+1)
    teacher_module(data_cfg, model_cfg)

if __name__ == "__main__":
    main()