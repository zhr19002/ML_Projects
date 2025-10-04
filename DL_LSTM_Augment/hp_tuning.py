import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, MultiHeadAttention, GlobalAveragePooling1D, Masking
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, Hyperband

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

wind_MLP = load_model('wind_MLP.keras')

cols = ['u','u_e','u_n']
scaled_cols = ['scaled_' + col for col in cols]

# Load datasets
df_buoy = pd.read_csv('WLIS_data.csv')
df_buoy = df_buoy.rename(columns={'WSPD':'u'})
df_buoy['TmStamp'] = pd.to_datetime(df_buoy['TmStamp'], format='mixed')
df_buoy.set_index('TmStamp', inplace=True)

df_stn = pd.read_csv('Sikorsky_data.csv')
df_stn = df_stn.rename(columns={'WSPD':'u'})
df_stn['TmStamp'] = pd.to_datetime(df_stn['TmStamp'], format='mixed')
df_stn.set_index('TmStamp', inplace=True)

# Add east and north components
alpha = -13
df_buoy['rad'] = np.pi/180 * ((alpha + 630 - df_buoy['WDIR']) % 360)
df_buoy['u_e'] = df_buoy['u'] * np.cos(df_buoy['rad'])
df_buoy['u_n'] = df_buoy['u'] * np.sin(df_buoy['rad'])

df_stn['rad'] = np.pi/180 * ((alpha + 630 - df_stn['WDIR']) % 360)
df_stn['u_e'] = df_stn['u'] * np.cos(df_stn['rad'])
df_stn['u_n'] = df_stn['u'] * np.sin(df_stn['rad'])

# Transform station wind into buoy wind
def wind_transform(df_stn):
    wind_T_uc = wind_MLP.predict(df_stn[cols[1:]], verbose=0)
    u_e, u_n = wind_T_uc[:, 0], wind_T_uc[:, 1]
    wind_T_u = np.sqrt(u_e**2 + u_n**2)
    df_T = pd.DataFrame({'u': wind_T_u, 'u_e': u_e, 'u_n': u_n})
    df_T.index = df_stn.index    
    return df_T

df_T = wind_transform(df_stn)

# df_test: [2024-07-01 00:00:00 ~ 2025-06-30 23:00:00]
split = pd.to_datetime('2024-06-30 23:59:00')
df_buoy_train = df_buoy[df_buoy.index < split]
df_buoy_test = df_buoy[df_buoy.index > split]
df_T_train = df_T[df_T.index < split]
df_T_test = df_T[df_T.index > split]

# Data normalization
scaled_wave, scaled_wind, scaled_T = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

df_buoy_train, df_buoy_test = df_buoy_train.copy(), df_buoy_test.copy()
df_buoy_train.loc[:, 'scaled_H'] = scaled_wave.fit_transform(df_buoy_train[['H']])
df_buoy_test.loc[:, 'scaled_H'] = scaled_wave.transform(df_buoy_test[['H']])
df_buoy_train.loc[:, scaled_cols] = scaled_wind.fit_transform(df_buoy_train[cols])
df_buoy_test.loc[:, scaled_cols] = scaled_wind.transform(df_buoy_test[cols])

df_T_train, df_T_test = df_T_train.copy(), df_T_test.copy()
df_T_train.loc[:, scaled_cols] = scaled_T.fit_transform(df_T_train[cols])
df_T_test.loc[:, scaled_cols] = scaled_T.transform(df_T_test[cols])

# Data preparation
step, output = 24, 12
features = ['scaled_H'] + scaled_cols

def create_sequences(df_buoy, df_T, step, output):
    X, y = [], []
    periods = step + output
    for i in range(len(df_buoy) - periods + 1):
        window = df_buoy.iloc[i:(i+periods)]
        expect = pd.date_range(start=window.index[0], periods=periods, freq='h')
        if not window.index.equals(expect):
            continue
        X_seq = window.values.copy()
        X_seq[step:, 0] = -1
        X_seq[step:, 1:] = df_T.iloc[(i+step):(i+periods)].values
        X.append(X_seq)
        y.append(window.iloc[step:, 0].values)
    return np.array(X), np.array(y)

# Validation set: [2007-11-01 00:00:00 ~ 2008-10-31 23:00:00]
split1 = pd.to_datetime('2007-10-31 23:59:00')
split2 = pd.to_datetime('2008-10-31 23:59:00')
df_buoy_val = df_buoy_train[(df_buoy_train.index > split1) & (df_buoy_train.index < split2)]
df_buoy_train = df_buoy_train[(df_buoy_train.index < split1) | (df_buoy_train.index > split2)]
df_T_val = df_T_train[(df_T_train.index > split1) & (df_T_train.index < split2)]
df_T_train = df_T_train[(df_T_train.index < split1) | (df_T_train.index > split2)]

X_train, y_train = create_sequences(df_buoy_train[features], df_T_train[scaled_cols], step, output)
X_val, y_val = create_sequences(df_buoy_val[features], df_T_val[scaled_cols], step, output)
X_test, y_test = create_sequences(df_buoy_test[features], df_T_test[scaled_cols], step, output)

epochs = 100
batch_size = 64

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        # Input layer with shape: (batch_size, time_steps, features)
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

        # Masking layer
        mask_layer = Masking(-1)(input_layer)
        x = mask_layer
        
        # Tunable LSTM layers and units
        num_layers = hp.Int('layers', min_value=1, max_value=4, step=1)
        for i in range(num_layers):
            x = LSTM(
                units = hp.Choice(f'units_{i}', values=[16, 32, 64, 128, 256]),
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
        output_layer = Dense(output, activation='relu')(pooled_out)
        
        # Compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
            loss = 'mse')
        
        return model

# Search space setup
tuner = Hyperband(
    LSTMHyperModel(),
    objective = 'val_loss',
    max_epochs = 100,
    factor = 3,
    directory = 'hp_tuning',
    project_name = 'model_LSTM')

# Hyperparameter tuning
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es])