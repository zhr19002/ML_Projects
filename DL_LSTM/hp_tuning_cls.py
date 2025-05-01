import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import tensorflow as tf
import keras
from keras import Sequential, regularizers
from keras.layers import Layer, Input, LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, Hyperband

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

@keras.saving.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W_q = self.add_weight(shape=(dim, dim), initializer='glorot_uniform', trainable=True)
        self.W_k = self.add_weight(shape=(dim, dim), initializer='glorot_uniform', trainable=True)
        self.W_v = self.add_weight(shape=(dim, dim), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)

        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attn_weights, v)
        
        return tf.reduce_sum(output, axis=1)

epochs = 50
batch_size = 64

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        
        # Input layer with shape: (batch_size, time_steps, features)
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        # Tunable LSTM layers and units
        num_layers = hp.Int('layers', min_value=1, max_value=3, step=1)
        for i in range(num_layers):
            model.add(LSTM(
                hp.Choice(f'units_{i}', values=[16, 32, 64, 128]),
                activation = 'tanh',
                kernel_regularizer = regularizers.l2(0.01),
                dropout = 0.2,
                return_sequences = True))
        
        # Attention layer
        model.add(Attention())
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
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
    max_epochs = 30,
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