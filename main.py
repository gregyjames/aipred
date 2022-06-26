import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import keras_tuner as kt
from os.path import exists

plt.style.use('fivethirtyeight')

import tensorflow as tf

df = web.DataReader('SPY',data_source='yahoo',start='2012-01-01',end='2022-06-20')
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

def gen_df(df):
    df["ema-5"] = pd.Series(df.Close.ewm(span=5).mean())
    df["ema-10"] = pd.Series(df.Close.ewm(span=10).mean())
    df["ema-20"] = pd.Series(df.Close.ewm(span=20).mean())
    df["ema-50"] = pd.Series(df.Close.ewm(span=50).mean())
    df["ema-100"] = pd.Series(df.Close.ewm(span=100).mean())
    df["macd"] = pd.Series(df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean())
    df["macd_signal"] = pd.Series(df.macd.ewm(span=9).mean())
    df["macd_over"] = df["macd"] > df["macd_signal"]
    df["macd_under"] = df["macd"] < df["macd_signal"]
    df["stochastic"] = pd.Series(df.Close.ewm(span=14).mean() / df.Close.rolling(window=14).std())
    df["stochastic_signal"] = pd.Series(df.stochastic.ewm(span=9).mean())
    df["stochastic_over"] = df["stochastic"] > df["stochastic_signal"]
    df["stochastic_under"] = df["stochastic"] < df["stochastic_signal"]
    df["aroon"] = pd.Series(df.Close.rolling(window=20).apply(lambda x: 100 * (x[-1] - x.min()) / (x.max() - x.min())))
    df["aroon_signal"] = pd.Series(df.aroon.ewm(span=9).mean())
    df["aroon_over"] = df["aroon"] > df["aroon_signal"]
    df["aroon_under"] = df["aroon"] < df["aroon_signal"]
    df["aroon_over"] = df["aroon"] > df["aroon_signal"]
    df["zigzag"] = pd.Series(df.Close.rolling(window=20).apply(lambda x: 100 * (x[-1] - x.min()) / (x.max() - x.min())))
    df["atr"] = pd.Series(df.Close.rolling(window=20).apply(lambda x: x.std() * math.sqrt(252)))
    df["bollinger_upper"] = pd.Series(df.Close.rolling(window=20).apply(lambda x: x.mean() + 2 * x.std()))
    df["bollinger_lower"] = pd.Series(df.Close.rolling(window=20).apply(lambda x: x.mean() - 2 * x.std()))
    df["obv"] = pd.Series(df.Close.diff() * df.Volume)
    df["volume_ratio"] = pd.Series(df.Volume / df.Volume.rolling(window=20).mean())
    return df


df = gen_df(df).dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

df = df.drop(['Open','High','Adj Close','Close'],1)
x = scaler.fit_transform(df)

print(x.shape)
print(y.shape)

prediction_days = 90

x_train = []
y_train = []

for i in range(prediction_days, len(x)):
    x_train.append(x[i-prediction_days:i, 0])
    y_train.append(y[i])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

def model_builder(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=hp.Int('units_3', min_value=32, max_value=512, step=32)))
    model.add(Dropout(0.2))

    for i in range(hp.Int('layers', 4, 12)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 512, step=32),activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
        model.add(Dropout(0.2))

    model.add(Dense(units=1))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-6])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_squared_error', metrics=['cosine_similarity'])
    return model

def train_model(x_train, y_train):
    if not exists('my_model.h5'):
        tuner = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=25,
                        directory='tuning')

        tuner.search(x_train, y_train, validation_split=0.2,use_multiprocessing=True)
        
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x_train, 
        y_train, 
        epochs=100, 
        validation_split=0.2,
        verbose=1,
        workers=16,
        use_multiprocessing=True)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        history.save('my_model.h5')
        return history
    else:
        history = keras.models.load_model('my_model.h5')
        history.fit(x_train, 
                    y_train, 
                    epochs=100,
                    validation_split=0.2,
                    workers=16,
                    batch_size=64,
                    verbose=1,
                    use_multiprocessing=True)
        history.save('my_model.h5')
        return history

train_model(x_train, y_train)