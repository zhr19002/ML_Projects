import pandas as pd
from config import *
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

station_encoder = NaNLabelEncoder(add_nan=True)
station_encoder.fit(pd.Series(SOURCE_IDS + TARGET_IDS))

def build_source_set(source_train):
    source_set = TimeSeriesDataSet(
        source_train,
        max_encoder_length = ENCODER_LENGTH,
        max_prediction_length = PREDICTION_LENGTH,
        time_idx = 'time_idx',
        target = 'prcp',
        group_ids = ['id'],
        static_categoricals = ['id'],
        static_reals = ['lat','lon','elev'],
        time_varying_known_categoricals = ['month','hour'],
        time_varying_known_reals = ['hour_sin','hour_cos','doy_sin','doy_cos'],
        time_varying_unknown_reals = ['prcp','temp','rhum','pres','uwnd','vwnd'],
        categorical_encoders = {'id': station_encoder},
        target_normalizer = GroupNormalizer(groups=['id']),
        add_relative_time_idx = True,
        add_target_scales = True,
        add_encoder_length = True,
    )
    return source_set

def build_dataloaders(train_set, val_df, test_df, batch_size):
    val_set = TimeSeriesDataSet.from_dataset(train_set, val_df, predict=False, stop_randomization=True)
    test_set = TimeSeriesDataSet.from_dataset(train_set, test_df, predict=False, stop_randomization=True)
    train_loader = train_set.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = val_set.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    test_loader = test_set.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    return (train_loader, val_loader, test_loader)

def build_target_set(source_set, target_train):
    target_set = TimeSeriesDataSet.from_dataset(source_set, target_train, predict=False, stop_randomization=True)
    return target_set