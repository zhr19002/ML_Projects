import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, MultiHeadAttention, GlobalAveragePooling1D, Masking
from keras.optimizers import Adam

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
    return pd.DataFrame({'u': wind_T_u, 'u_e': u_e, 'u_n': u_n}, index=df_stn.index)

df_T = wind_transform(df_stn)

# df_test: [2024-07-01 00:00:00 ~ 2025-06-30 23:00:00]
split = pd.to_datetime('2024-06-30 23:59:00')
df_buoy_train = df_buoy[df_buoy.index < split]
df_buoy_test = df_buoy[df_buoy.index > split]
df_stn_train = df_stn[df_stn.index < split]
df_stn_test = df_stn[df_stn.index > split]
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
df_stn_val = df_stn_train[(df_stn_train.index > split1) & (df_stn_train.index < split2)]
df_stn_train = df_stn_train[(df_stn_train.index < split1) | (df_stn_train.index > split2)]
df_T_val = df_T_train[(df_T_train.index > split1) & (df_T_train.index < split2)]
df_T_train = df_T_train[(df_T_train.index < split1) | (df_T_train.index > split2)]

X_val, y_val = create_sequences(df_buoy_val[features], df_T_val[scaled_cols], step, output)
X_test, y_test = create_sequences(df_buoy_test[features], df_T_test[scaled_cols], step, output)

rho = 0.054

# Add future wind noise
def add_noise(df, abs_std, rel_std, random_seed=42):
    rng = np.random.default_rng(random_seed)
    wind = df[cols].to_numpy(dtype=float)
    stds = abs_std + rel_std * wind[:, 0]
    cov = np.array([[1, rho], [rho, 1]])
    L_cov = np.linalg.cholesky(cov)
    z = rng.normal(0, 1, size=(wind.shape[0], 2))
    noise = z @ L_cov.T * stds[:, None]
    nu_e = wind[:, 1] + noise[:, 0]
    nu_n = wind[:, 2] + noise[:, 1]
    nu = np.linalg.norm(np.column_stack([nu_e, nu_n]), axis=1)
    nu = np.maximum(nu, 0.25)
    return pd.DataFrame({'u':nu, 'u_e':nu_e, 'u_n':nu_n}, index=df.index)

# Model architecture
def create_model(X):
    # Input layer with shape: (batch_size, time_steps, features)
    input_layer = Input(shape=(X.shape[1], X.shape[2]))

    # Masking layer
    mask_layer = Masking(-1)(input_layer)

    # LSTM layer
    units = 16
    lstm_1 = LSTM(units=256, activation='tanh', return_sequences=True)(mask_layer)
    lstm_2 = LSTM(units=16, activation='tanh', return_sequences=True)(lstm_1)
    lstm_3 = LSTM(units=64, activation='tanh', return_sequences=True)(lstm_2)
    lstm_out = LSTM(units=units, activation='tanh', return_sequences=True)(lstm_3)
    
    # Attention layer
    num_heads = 2
    attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=units//num_heads)
    attn_out, attn_scores = attn_layer(lstm_out, lstm_out, return_attention_scores=True)
    
    # Pooling layer
    pooled_out = GlobalAveragePooling1D()(attn_out)

    # Output layer
    output_layer = Dense(12)(pooled_out)
    
    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005271053626795346), loss='mse')
    attn_model = Model(inputs=input_layer, outputs=attn_scores)
    
    return model, attn_model

epochs = 100
samples = 20
batch_size = 64
model_file = 'model_aug.keras'

# Train and save the model
model, attn_model = create_model(X_test)
loss, val_loss = [], []
best_val_loss, count = np.inf, 0

for i in range(epochs):
    epoch_loss, epoch_val_loss = [], []

    for n in range(samples):
        rng_epoch = np.random.default_rng(100*(i+1)+n)
        abs_std = rng_epoch.uniform(0, 1.5)
        rel_std = rng_epoch.uniform(0, 0.5)

        # Add wind noise
        df_stn_train_nu = add_noise(df_stn_train[cols], abs_std, rel_std, random_seed=100*(i+1)+n)
        df_stn_train_nu_T = wind_transform(df_stn_train_nu)
        df_stn_train_nu_T.loc[:, scaled_cols] = scaled_T.transform(df_stn_train_nu_T[cols])
        X_train, y_train = create_sequences(df_buoy_train[features], df_stn_train_nu_T[scaled_cols], step, output)

        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
        epoch_loss.append(history.history['loss'][0])
        epoch_val_loss.append(history.history['val_loss'][0])
    
    loss.append(np.mean(epoch_loss))
    val_loss.append(np.mean(epoch_val_loss))
    print(f'Epoch {i+1}: loss={np.mean(epoch_loss):.4f}, val_loss={np.mean(epoch_val_loss):.4f}')

    # Early stopping
    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        model.save(model_file)
        count = 0
    else:
        count += 1
        if count >= 10:
            break

with open('loss_1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'loss', 'val_loss'])
    for i, (l, vl) in enumerate(zip(loss, val_loss), 1):
        writer.writerow([i, l, vl])