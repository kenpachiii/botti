import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from tqdm import tqdm
from datetime import datetime, timezone
from time import time
import tensorflow as tf
import tensorflow_addons as tfa
import multiprocessing as mp

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Conv1D, MaxPooling1D, Convolution1D, Activation, LeakyReLU, Flatten, BatchNormalization
from keras import optimizers, callbacks
from keras.optimizers.schedules.learning_rate_schedule import InverseTimeDecay
  
def create_series(df, xcol):
    features_considered = xcol
    features = df[features_considered]
    return features

# https://keras.io/examples/timeseries/timeseries_traffic_forecasting/#creating-tensorflow-datasets
def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    batch_size: int = 128,
    shuffle: bool = False
):

    input_data = data_array[:-input_sequence_length]
    targets = data_array[input_sequence_length:]

    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    input_data, targets, sequence_length=input_sequence_length,batch_size=batch_size,shuffle=shuffle,seed=10)

    return dataset.prefetch(16).cache()

def lookback_window(row, values, method = 'sum', *args, **kwargs):
    loc = values.index.get_loc(row.name)
    return getattr(values.iloc[0: loc + 1], method)(*args, **kwargs)

def stationarity_test(X, log_x = "Y", sample_size = 500):

    X = X[:sample_size]

    if log_x == "Y":
        X = np.log(X[X>0])
    
    from statsmodels.tsa.stattools import adfuller
    dickey_fuller = adfuller(X)

    print('ADF Stat is: {}.'.format(dickey_fuller[0]))
    print('P Val is: {}.'.format(str(dickey_fuller[1])))
    print('--------------------------------------')

def auto_cov(X, h):
    n = len(X)
    X_bar = np.mean(X)
    X_centered = X - X_bar
    return np.dot( X_centered[:(n-h)], X_centered[h:] ) / (n-h)

def correlation(x):
    print(x.corr())
    print('--------------------------------------')

def portmanteau(x):

    from scipy.stats import chi2

    alpha = 0.05
    quantile = chi2.ppf(q=1-alpha, df=5)
    sum_of_autocov = 0
    n = x.shape[0]

    for h in np.arange(1,6):
        sum_of_autocov += auto_cov(x, h) ** 2

    test_statistic = n * sum_of_autocov / (auto_cov(x, 0) ** 2)

    if test_statistic > quantile:
        print('The null hypothesis can be rejected.')
        print('TS = '  + str(np.round(test_statistic, 2)) + ' > ' + str(np.round(quantile,2)))
        print('--------------------------------------')
    else:
        print('The null hypothesis cannot be rejected.')
        print('--------------------------------------')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def transform_ma(x):
    return np.array([moving_average(x[:i + 1], i + 1)[0] for i in range(len(x))])

def transform_ema(x):
    x = pd.DataFrame(x)
    return x.ewm(span = 6).mean().fillna(0).to_numpy()

def transform(df: pd.DataFrame):

    from statsmodels.tsa.api import SimpleExpSmoothing
    np.warnings.filterwarnings('ignore')

    df.loc[:,('log_returns')] = np.nan_to_num(np.log(df['price'] / df['price'].shift(-1)), nan = 2.220446049250313e-16)

    log_es = SimpleExpSmoothing(df.loc[:,('log_returns')], initialization_method = 'heuristic')
    log_es.fit(smoothing_level = 0.070, optimized = True, remove_bias = True, use_brute = True)

    df.loc[:,('log_returns_mu')] = log_es.predict(log_es.params, start = 0, end = None)
    df.loc[:,('price_mu')] = df['price'].rolling(48).mean()
    df.loc[:,('amount_mu')] = df['amount'].rolling(48).mean()

    return df.dropna()

def relative_returns(df):
    df = pd.DataFrame(df)
    return np.log(df / df.shift(-1))

def difference(x):
    return np.nan_to_num(np.diff(x))

def invert_difference(x, diff):
    return np.concatenate(([diff], x)).cumsum()

def normalize(scalar, x):

    # scalar_0 = MinMaxScaler((0, 1))
    return scalar.transform(x.reshape(-1, 3))

    return np.stack((scalar.transform(x[:, 0].reshape(-1, 1)), scalar_0.fit_transform(x[:, 1].reshape(-1, 1))), axis = -1).reshape(-1, 2)

def denormalize(scalar, x):

    x = x.reshape(-1, 1)

    m, n = x.shape
    out = np.zeros((m, 3 * n), dtype = x.dtype)
    out[:,::3] = x

    return scalar.inverse_transform(out)

def preprocess_data(dataset, numeric_colname):
            
    rnn_df = create_series(dataset, numeric_colname)
    
    # print('Correlation Coefficients\n')
    # correlation(dataset)
    
    # print('ADF Test For Stationarity (price)\n')
    # stationarity_test(X = rnn_df[numeric_colname])

    # print('Portmanteau Test For Independence (price)\n')
    # portmanteau(rnn_df[numeric_colname])

    # print('ADF Test For Stationarity (returns)\n')
    # stationarity_test(X = np.log(rnn_df[numeric_colname] / rnn_df[numeric_colname].shift(1))[1:])

    # print('Portmanteau Test For Independence (returns)\n')
    # portmanteau(np.log(rnn_df[numeric_colname] / rnn_df[numeric_colname].shift(1))[1:])
    
    return rnn_df

def rsquared(y_true, y_pred):
    """ Return R^2 where x and y are array-like."""

    from sklearn.metrics import r2_score

    return r2_score(y_true, y_pred)

def accuracy(y_true, y_pred):

    # check if all positive and get diff to measure direction
    # if np.sum(np.sign(y_true)) == len(y_true):
    #     y_true, y_pred = np.diff(y_true.reshape(-1,)), np.diff(y_pred.reshape(-1,))

    t_p, t_n, f_p, f_n = 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16
    for x, y in zip(y_true, y_pred):
        
        if np.sign(x) == -1 and np.sign(y) == -1:
            t_n += 1

        if np.sign(x) == 1 and np.sign(y) == 1:
            t_p += 1

        if np.sign(y) == -1 and (np.sign(x) == 1 or np.sign(x) == 0):
            f_n += 1

        if np.sign(y) == 1 and (np.sign(x) == -1 or np.sign(x) == 0):
            f_p += 1

    return t_p, t_n, f_p, f_n


def custom_loss_function(y_true, y_pred):

    alpha = 100
    loss = K.switch(
        K.less(y_true * y_pred, 0), 
        alpha * y_pred**2 - K.sign(y_true) * y_pred + K.abs(y_true), 
        K.abs(y_true - y_pred)
    )
    
    return K.mean(loss, axis = -1)

def build_model_0():
    tf.random.set_seed(20)
    np.random.seed(10)

    model_input = Input(shape = (input_sequence_length, 3), name = 'input_for_model')

    inputs = Conv1D(filters = 8, kernel_size = 2, dilation_rate = 2**0, activation = 'relu')(model_input)
    inputs = MaxPooling1D(pool_size = 2, padding = 'same')(inputs)
    inputs = LeakyReLU()(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = Conv1D(filters = 8, kernel_size = 2, dilation_rate = 2**1, activation = 'relu')(inputs)
    inputs = MaxPooling1D(pool_size = 2, padding = 'same')(inputs)
    inputs = LeakyReLU()(inputs)
    inputs = Conv1D(filters = 8, kernel_size = 2, dilation_rate = 2**2, activation = 'relu')(inputs)
    inputs = MaxPooling1D(pool_size = 2, padding = 'same')(inputs)
    inputs = Flatten()(inputs)
    output = Dense(1, activation = 'linear')(inputs)
    
    model = Model(inputs = model_input, outputs = output)
    adam = optimizers.Adam(learning_rate = 0.001)

    model.compile(optimizer = adam, loss = 'mae')
    model.summary()

    return model

def bytes_size(path):

    if isinstance(path, str):
        path = [path]

    return np.sum([os.path.getsize(file) for file in path])

def read_file(pbar, file):
    ddf = dd.read_csv(file, header = 0, names = ['id', 'side', 'amount', 'price', 'timestamp'], blocksize = None)
    pbar.update(bytes_size(file))
    return ddf

def fetch_samples(n_symbols = None, n_files = None, use_preloaded_scalers = False):

    path = os.path.join('data', 'okx')

    ddf: dd.DataFrame = None
    symbols = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'ICP-USDT-SWAP', 'DOGE-USDT-SWAP'] 

    symbols = symbols[:n_symbols]
    
    for symbol in symbols:

        files = glob.glob(os.path.join(path, 'trades', symbol, '*.zip'))
        files.sort()

        files = files[:n_files]
        sz = bytes_size(files)

        with tqdm(total=sz) as pbar:

            batch_ddf = None
            if use_preloaded_scalers:
                batch_ddf = joblib.load(os.path.join(path, 'trades', symbol, 'batch.save'))
            else:

                batch_ddf: dd.DataFrame = dd.concat([read_file(pbar, file) for file in files])
                batch_ddf = batch_ddf.groupby(by=['timestamp', 'side']).agg({ 'price': np.mean, 'amount': np.sum }, split_out = len(files)).reset_index()

                batch_ddf = batch_ddf.persist()

                batch_ddf = batch_ddf.set_index('timestamp').repartition(npartitions = len(files))
                batch_ddf.index = dd.to_datetime(batch_ddf.index, utc = True, unit = 'ms')

                batch_ddf = batch_ddf.persist()

                batch_ddf_buy = batch_ddf[batch_ddf.side == 'BUY']
                batch_ddf_sell = batch_ddf[batch_ddf.side == 'SELL']

                batch_ddf_buy = batch_ddf_buy.resample(f'1800S').agg({ 'price': np.mean, 'amount': np.sum }).dropna()
                batch_ddf_sell = batch_ddf_sell.resample(f'1800S').agg({ 'price': np.mean, 'amount': np.sum }).dropna()

                batch_ddf = batch_ddf.persist()

                batch_ddf = dd.concat([batch_ddf_buy, batch_ddf_sell])
                batch_ddf = dd.from_pandas(batch_ddf.compute().sort_index(), npartitions = batch_ddf.npartitions)

                spath = os.path.join(path, 'trades', symbol, 'batch.save')
                joblib.dump(batch_ddf, spath, compress = ('xz', 3))

            if getattr(ddf, '_name', None) == None:
                ddf: dd.DataFrame = batch_ddf
            else:
                ddf: dd.DataFrame = dd.concat([ddf, batch_ddf])

    return ddf

def example_trade(unprocessed, y_pred):

    y_pred = y_pred.reshape(-1,)

    longs, shorts = np.empty((0, 2)), np.empty((0, 2))

    success_long, success_short, total, n_steps, i = 0, 0, 2.220446049250313e-16, np.array([]), 0
    for y_pred, unprocessed in zip(y_pred, unprocessed):

        if np.sign(y_pred) == 1 and y_pred > ((np.square(1 + 0.0005) - 1)):
            longs = np.append(longs, [[unprocessed * np.square(1 + 0.0005), i]], axis = 0)
            total += 1

        if np.sign(y_pred) == -1 and y_pred < ((np.square(1 - 0.0005) - 1)):
            shorts = np.append(shorts, [[unprocessed * np.square(1 - 0.0005), i]], axis = 0)
            total += 1

        idx = unprocessed > longs[:, 0]
        if np.sum(idx) > 0:
            success_long += np.sum(idx)
            n_steps = np.append(n_steps, i - longs[~~idx][:, 1])
            longs = longs[~idx]

        idx = unprocessed < shorts[:, 0]
        if np.sum(idx) > 0:
            success_short += np.sum(idx)
            n_steps = np.append(n_steps, i - shorts[~~idx][:, 1])
            shorts = shorts[~idx]

        i += 1

    print(f'example trade {(success_long + success_short) / total} - {total} - [{success_long}, {success_short}] - {np.mean(n_steps)}, {np.std(n_steps)}')

def print_stats(model, history, test_dataset, scalar, unprocessed):

    x_test, y = next(test_dataset.as_numpy_iterator())

    y_pred = model.predict(x_test, workers = mp.cpu_count(), use_multiprocessing = True)

    x_test, y, y_pred = x_test[:, -1], y[:, 0], y_pred[:, 0]

    from sklearn.metrics import mean_squared_error

    naive_rmse, model_rmse, r2, cc = (
        mean_squared_error(y, x_test[:, 0], squared = False),
        mean_squared_error(y, y_pred, squared = False),
        rsquared(y, y_pred),
        np.corrcoef(y, y_pred)
    )

    print(f'naive RMSE: {naive_rmse}, model RMSE: {model_rmse}, r2: {r2}, cc: {np.min(cc[0])}')

    scalar_0 = MinMaxScaler((0, 1))

    t_p, t_n, f_p, f_n = accuracy(denormalize(scalar, y)[:, 0], denormalize(scalar, y_pred)[:, 0])
    print(f'accuracy: [{(t_p + t_n) / (t_p + t_n + f_p + f_n)}] - {t_p / (t_p + f_n)} {t_n / (t_n + f_p)} - {(t_p + t_n + f_p + f_n)} samples') 

    example_trade(unprocessed, denormalize(scalar, y_pred)[:, 0].reshape(-1,))

    import plotly.express as px
    index = [i for i in range(0, len(getattr(history, 'history')['loss']))]
    fig = px.line(x = index, y = [getattr(history, 'history')['loss'], getattr(history, 'history')['val_loss']]) 
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash')
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=False,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash')

    fig.update_layout(hoverdistance = 0)
    fig.update_traces(xaxis = 'x', hoverinfo = 'none')

    fig.show()

    index = [i for i in range(0, len(y_pred))]
    fig = px.line(x = index, y = [scalar_0.fit_transform(unprocessed.reshape(-1, 1)).reshape(-1,), y_pred]) 

    newnames = { 'wide_variable_0': 'unprocessed', 'wide_variable_1': 'y_pred' }
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))

    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash')
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=False,
                 showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash')

    fig.update_layout(hoverdistance = 0)
    fig.update_traces(xaxis = 'x', hoverinfo = 'none')

    fig.show()

if __name__ == "__main__":
    
    validation_split = (0.90, 0.05)
    batch_size = 32
    input_sequence_length = 64

    samples = fetch_samples(n_symbols = 2, use_preloaded_scalers = True).compute().reset_index()
    rnn_df: pd.DataFrame = preprocess_data(samples, numeric_colname = ['price', 'amount'])
 
    num_time_steps = rnn_df.shape[0]
    num_train, num_val = (
        int(num_time_steps * validation_split[0]),
        int(num_time_steps * validation_split[1]),
    )

    # log_returns is in lined up with past price and not the current price
    train_df = transform(rnn_df[:num_train])
    val_df = transform(rnn_df[num_train : (num_train + num_val)])
    test_df = transform(rnn_df[num_train + num_val:])

    train_array = np.stack((train_df.loc[:,('log_returns_mu')].to_numpy(), train_df.loc[:,('price_mu')].to_numpy(), train_df.loc[:,('amount_mu')].to_numpy()), axis = -1)
    val_array = np.stack((val_df.loc[:,('log_returns_mu')].to_numpy(), val_df.loc[:,('price_mu')].to_numpy(), val_df.loc[:,('amount_mu')].to_numpy()), axis = -1)
    test_array = np.stack((test_df.loc[:,('log_returns_mu')].to_numpy(), test_df.loc[:,('price_mu')].to_numpy(), test_df.loc[:,('amount_mu')].to_numpy()), axis = -1)

    print(f'train: {train_array.shape} {np.min(train_array)} {np.max(train_array)} {np.mean(train_array)} {np.sum(np.sign(train_array) == 1), np.sum(np.sign(train_array) == -1), np.sum(np.sign(train_array) == 0)}')
    print(f'validation: {val_array.shape} {np.min(val_array)} {np.max(val_array)} {np.mean(val_array)} {np.sum(np.sign(val_array) == 1), np.sum(np.sign(val_array) == -1), np.sum(np.sign(val_array) == 0)}')
    print(f'test: {test_array.shape} {np.min(test_array)} {np.max(test_array)} {np.mean(test_array)} {np.sum(np.sign(test_array) == 1), np.sum(np.sign(test_array) == -1), np.sum(np.sign(test_array) == 0)}')

    scalar = MinMaxScaler((0, 1))
    scalar = scalar.fit(train_array.reshape(-1, 3))

    train_dataset, val_dataset = (create_tf_dataset(data_array, input_sequence_length, batch_size, shuffle = True) for data_array in [normalize(scalar, train_array), normalize(scalar, val_array)])
    test_dataset = create_tf_dataset(
        normalize(scalar, test_array),
        input_sequence_length,
        batch_size=test_array.shape[0],
        shuffle=False
    )
    
    model = build_model_0()
    history = model.fit(train_dataset, validation_data = val_dataset, batch_size = batch_size, epochs = 100, validation_split = validation_split, callbacks = [
        callbacks.EarlyStopping(monitor = 'loss', patience = 3, restore_best_weights = True),
        callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True),
        callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
    ], workers = mp.cpu_count(), use_multiprocessing = True)
    # model.save('model')
    # joblib.dump(scalar, 'model/assets/model.scalar')


    
    print_stats(model, history, test_dataset, scalar, test_df.loc[:,('price')][:-input_sequence_length][:-63].to_numpy())

