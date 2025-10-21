import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, MultiHeadAttention, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, Hyperband

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

wind_cols = ['uSq','e_uSq','n_uSq','Z2','Z3','Z4']
scaled_wind_cols = ['scaled_' + col for col in wind_cols]

# Load datasets
df_buoy_04_13 = pd.read_csv('Data_Buoy_2004_2013.csv')
df_stn_04_13 = pd.read_csv('Data_Sikorsky_2004_2013.csv')
df_stn_73_04 = pd.read_csv('Data_Sikorsky_1973_2004.csv')

# Rename columns
df_buoy_04_13 = df_buoy_04_13.rename(columns={'SWHft':'H'})
df_stn_04_13 = df_stn_04_13.rename(columns={'Fetch':'Z3'})
df_stn_73_04 = df_stn_73_04.rename(columns={'Fetch':'Z3'})

# Add transformed wind speed (u)
df_stn_04_13['u'] = df_stn_04_13['rWSPD_T'].shift(-1).ffill()
df_stn_73_04['u'] = df_stn_73_04['rWSPD_T'].shift(-1).ffill()

# Add squared terms (u^2)
df_stn_04_13['uSq'] = df_stn_04_13['u']**2
df_stn_73_04['uSq'] = df_stn_73_04['u']**2

# Add east and north components (e_uSq, n_uSq)
alpha = -13
df_stn_04_13['rad'] = np.pi/180 * ((alpha + 630 - df_stn_04_13['WDIR']) % 360)
df_stn_73_04['rad'] = np.pi/180 * ((alpha + 630 - df_stn_73_04['WDIR']) % 360)
df_stn_04_13['e_uSq'] = df_stn_04_13['uSq'] * np.cos(df_stn_04_13['rad'])
df_stn_73_04['e_uSq'] = df_stn_73_04['uSq'] * np.cos(df_stn_73_04['rad'])
df_stn_04_13['n_uSq'] = df_stn_04_13['uSq'] * np.sin(df_stn_04_13['rad'])
df_stn_73_04['n_uSq'] = df_stn_73_04['uSq'] * np.sin(df_stn_73_04['rad'])

# Add interaction terms (Z2, Z4)
df_stn_04_13['Z2'] = np.sqrt(df_stn_04_13['Z3']) * df_stn_04_13['u']
df_stn_73_04['Z2'] = np.sqrt(df_stn_73_04['Z3']) * df_stn_73_04['u']
df_stn_04_13['Z4'] = (df_stn_04_13['Z3']**1.5) / df_stn_04_13['u']
df_stn_73_04['Z4'] = (df_stn_73_04['Z3']**1.5) / df_stn_73_04['u']

# Data aggregation
df1 = pd.concat([df_buoy_04_13[['TimeStamp_1','H']], df_stn_04_13[wind_cols]], axis=1)
df2 = df_stn_73_04[['TimeStamp_1'] + wind_cols].copy()

# Set the timestamp column as the index
df1['TimeStamp_1'] = pd.to_datetime(df1['TimeStamp_1'], format='mixed')
df2['TimeStamp_1'] = pd.to_datetime(df2['TimeStamp_1'], format='mixed')
df1.set_index('TimeStamp_1', inplace=True)
df2.set_index('TimeStamp_1', inplace=True)

# Train-Test split
split1 = pd.to_datetime('2007-10-31 23:59:00')
split2 = pd.to_datetime('2008-10-31 23:59:00')
df_train = df1[(df1.index < split1) | (df1.index > split2)]
df_test = df1[(df1.index > split1) & (df1.index < split2)]

# Data normalization
scaled_wave, scaled_wind = MinMaxScaler(), MinMaxScaler()
df_train, df_test = df_train.copy(), df_test.copy()

# Normalize wave in the first dataset
df_train.loc[:, 'scaled_H'] = scaled_wave.fit_transform(df_train[['H']])
df_test.loc[:, 'scaled_H'] = scaled_wave.transform(df_test[['H']])

# Normalize wind in both datasets
df_train.loc[:, scaled_wind_cols] = scaled_wind.fit_transform(df_train[wind_cols])
df_test.loc[:, scaled_wind_cols] = scaled_wind.transform(df_test[wind_cols])
df2.loc[:, scaled_wind_cols] = scaled_wind.transform(df2[wind_cols])

time_step = 6
features = ['scaled_H'] + scaled_wind_cols

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:(i+time_step+1), 1:])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

split3 = pd.to_datetime('2009-10-31 23:59:00')
df_train1 = df_train[df_train.index < split1]
df_train2 = df_train[df_train.index > split3]
df_val = df_train[(df_train.index > split2) & (df_train.index < split3)]

X_train1, y_train1 = create_sequences(df_train1[features].values, time_step)
X_train2, y_train2 = create_sequences(df_train2[features].values, time_step)
X_train, y_train = np.concatenate((X_train1, X_train2), axis=0), np.concatenate((y_train1, y_train2), axis=0)
X_val, y_val = create_sequences(df_val[features].values, time_step)
X_test, y_test = create_sequences(df_test[features].values, time_step)

epochs = 100
batch_size = 64

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        # Input layer with shape: (batch_size, time_steps, features)
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = input_layer
        
        # Tunable LSTM layers and units
        num_layers = hp.Int('layers', min_value=1, max_value=3, step=1)
        for i in range(num_layers):
            x = LSTM(
                units = hp.Choice(f'units_{i}', values=[32, 64, 128, 256]),
                activation = 'tanh',
                return_sequences = True)(x)

        # Attention layer
        num_heads = hp.Choice('heads', values=[2, 4, 8])
        attn_layer = MultiHeadAttention(
            num_heads = num_heads,
            key_dim = x.shape[-1] // num_heads,
            name = "multihead_attn")
        attn_out, attn_scores = attn_layer(x, x, return_attention_scores=True)

        # Pooling layer
        pooled_out = GlobalAveragePooling1D()(attn_out)

        # Output layer
        output_layer = Dense(1, activation='relu')(pooled_out)
        
        # Compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
            loss = 'mse',
            metrics = ['mse'],
        )
        
        return model

# Search space setup
tuner = Hyperband(
    LSTMHyperModel(),
    objective = 'val_loss',
    max_epochs = 100,
    factor = 3,
    directory = 'hp_tuning',
    project_name = 'model_reg0',
)

# Hyperparameter tuning
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es])