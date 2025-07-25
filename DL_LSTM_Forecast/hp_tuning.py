import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, MultiHeadAttention, GlobalAveragePooling1D, Masking, Bidirectional
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
df = pd.read_csv('WLIS_data.csv')
df = df.rename(columns={'WSPD':'u', 'Fetch':'Z3'})
df['TmStamp'] = pd.to_datetime(df['TmStamp'], format='mixed')
df.set_index('TmStamp', inplace=True)

# Add squared terms (uSq)
df['uSq'] = df['u']**2

# Add east and north components (e_uSq, n_uSq)
alpha = -13
df['rad'] = np.pi/180 * ((alpha + 630 - df['WDIR']) % 360)
df['e_uSq'] = df['uSq'] * np.cos(df['rad'])
df['n_uSq'] = df['uSq'] * np.sin(df['rad'])

# Add interaction terms (Z2, Z4)
df['Z2'] = np.sqrt(df['Z3']) * df['u']
df['Z4'] = (df['Z3']**1.5) / df['u']

# Train-Test split
split1 = pd.to_datetime('2007-10-31 23:59:00')
split2 = pd.to_datetime('2008-10-31 23:59:00')
df_train = df[(df.index < split1) | (df.index > split2)]
df_test = df[(df.index > split1) & (df.index < split2)]

# Data normalization
scaled_wave, scaled_wind = MinMaxScaler(), MinMaxScaler()
df_train, df_test = df_train.copy(), df_test.copy()
df_train.loc[:, 'scaled_H'] = scaled_wave.fit_transform(df_train[['H']])
df_test.loc[:, 'scaled_H'] = scaled_wave.transform(df_test[['H']])
df_train.loc[:, scaled_wind_cols] = scaled_wind.fit_transform(df_train[wind_cols])
df_test.loc[:, scaled_wind_cols] = scaled_wind.transform(df_test[wind_cols])

# Data preparation
step, output = 24, 12
features = ['scaled_H'] + scaled_wind_cols

def create_sequences(df, step, output):
    X, y = [], []
    periods = step + output
    for i in range(len(df) - periods + 1):
        window = df.iloc[i:(i+periods)]
        expect = pd.date_range(start=window.index[0], periods=periods, freq='h')
        if not window.index.equals(expect):
            continue
        X_seq = window.values.copy()
        X_seq[step:, 0] = -1
        y_seq = window.iloc[step:, 0].values
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

split3 = pd.to_datetime('2009-10-31 23:59:00')
df_val = df_train[(df_train.index > split2) & (df_train.index < split3)]
df_train = df_train[(df_train.index < split1) | (df_train.index > split3)]

X_train, y_train = create_sequences(df_train[features], step, output)
X_val, y_val = create_sequences(df_val[features], step, output)
X_test, y_test = create_sequences(df_test[features], step, output)

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
            x = Bidirectional(LSTM(
                units = hp.Choice(f'units_{i}', values=[16, 32, 64, 128, 256]),
                activation = 'tanh',
                return_sequences = True))(x)

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
    project_name = 'model_BiLSTM',
)

# Hyperparameter tuning
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es])