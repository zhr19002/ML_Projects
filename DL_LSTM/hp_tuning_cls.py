import os
import random
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
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

spike = 4
wind_cols = ['uSq','e_uSq','n_uSq','Z2','Z3','Z4']
scaled_wind_cols = ['scaled_' + col for col in wind_cols]

# Load datasets
df1 = pd.read_csv('Data_2004_2013.csv')
df1['Spike'] = (df1['H'] > spike).astype(int)

# Set the timestamp column as the index
df1['TimeStamp_1'] = pd.to_datetime(df1['TimeStamp_1'], format='mixed')
df1.set_index('TimeStamp_1', inplace=True)

# Train-Test split
split1 = pd.to_datetime('2007-10-31 23:59:00')
split2 = pd.to_datetime('2008-10-31 23:59:00')
df_train = df1[(df1.index < split1) | (df1.index > split2)]
df_test = df1[(df1.index > split1) & (df1.index < split2)]

time_step = 6
features = ['Spike'] + scaled_wind_cols

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
        output_layer = Dense(1, activation='sigmoid')(pooled_out)
        
        # Compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
            loss = 'binary_crossentropy',
            metrics = ['AUC'],
        )
        
        return model

# Search space setup
tuner = Hyperband(
    LSTMHyperModel(),
    objective = 'val_loss',
    max_epochs = 100,
    factor = 3,
    directory = 'hp_tuning',
    project_name = 'model_cls',
)

# Class weights computation
w_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
w_dict = {i:w_array[i] for i in range(len(w_array))}

# Hyperparameter tuning
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=w_dict, validation_data=(X_val, y_val), callbacks=[es])