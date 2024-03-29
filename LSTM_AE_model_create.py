# Task 1: Import Libraries
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import os
import pickle
import plotly.graph_objects as go
import tensorflow as tf
from pathlib import Path
from platform import node
import time
import timeit
import json
import scipy.signal as sps
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras as kr
# from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from LSTM_AE_custom_models import *
from Notification_module import *
os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings('ignore')
# program control flags
USE_CUDA = True  # use CPU or Nvidia GPU
USE_DEBUGPRINT = True  # Add additional debug flags
USE_FFT = True  # get spectrogram plot
USE_ANOTHERTESTFILE = True  # use prediction model on another NOK time series file
parent_dir = Path(__file__).resolve().parent
out_dir = str(Path.joinpath(parent_dir, "output")) + os.sep
tmp_dir = str(Path.joinpath(parent_dir, "tmp_dir")) + os.sep
cache_dir = str(Path.joinpath(parent_dir, "cache_dir")) + os.sep

runtime_start = timeit.default_timer()
timestr = time.strftime("%Y%m%d-%H%M%S")
model_parameters = dict(
    # Model variables are set here:
    Timestamp=timestr,  # timestring for identification
    # data preparation
    my_learningsequence="dataset2_ultrasonic_nok", #visc6_ultrasonic_ok visc6_nosonic_ok
    my_samplingfrequency=0,  # automatic detection ok
    sequence_start=6361,  #9190 start second in audio file for  subsequence analysis
    sequence_stop=6481,  #9210 stop second in audio file for subsequence analysis
    train_test_split=0.8,  # 80/20 split for training/testing set
    time_steps=100,  # 30 size of sub-sequences for LSTM feeding
    # model learining
    my_epochs=200,  # 10  times when the entire dataset passed through the entire network
    my_batch_size=64,  # 32  dimensions of time steps for 2d input pattern
    my_validation_split=0.2,  # 0.1
    # my_dropout=0.2, #  model-depending, not global. likely not useful for sequences
    # model quality criteria
    my_loss='mae',
    my_optimizer='adam',
    # anomaly detection
    my_threshold=2.5,
    # early stop paramerers
    my_min_delta=0.0001,
    my_monitor='val_loss',
    my_patience=5,  # number epochs to train without improvement. after 3 -> stop
    my_mode='min',
    my_verbose=1,
    my_predictsequence="dataset2_ultrasonic_nok",  # use this file to predict on a second timeseries
    my_nok_startsec=12308,  # startpoint for second timeseries
    my_nok_stopsec=12555,  # endpoint for second timeseries
    my_traintime='',
    my_ostype='',
    my_cudaversion='',
    my_fftusage=False,
    my_pythonversion=sys.version,
    my_tensorflowversion=tf.__version__,
    my_kerasversion=kr.__version__
)

def save_training_parameters(modelparameters):
    with open(out_dir + timestr + '_my_globals.json', 'w') as fp:
        json.dump(modelparameters, fp)

OS_TYPE = os.name
MACHINE_ID = node()
print(f"OS_TYPE: {OS_TYPE}")
print(f"MACHINE_ID: {MACHINE_ID}")
print(f"Working directory: {parent_dir}{os.sep}")
print(f"Working directory: {out_dir}")
np.random.seed(1)
tf.random.set_seed(1)
if USE_CUDA:
    print("Enable CUDA Support")
    import cupy as cp  # install pip cupy-cuda12x
    import cupyx.scipy as cps
    import cusignal
    feedback_cuda = cp.cuda.runtime.driverGetVersion()
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
        if(not(gpu_devices)):
            feedback_cuda = "Using CPU only"
        else:
            model_parameters['my_cudaversion'] = feedback_cuda

    print(f"CUDA driver version is {feedback_cuda}")
    print("Cuda devices available:", tf.config.list_physical_devices('GPU'))
else:
    print(f"Using CPU only")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    feedback_cuda="Using CPU only"
print('Tensorflow version:', tf.__version__)

model_parameters.update({
    'my_ostype': OS_TYPE,
    'my_cudaversion': feedback_cuda,
    'my_fftusage': USE_FFT,
})

def read_flac_to_pandas(filename, start_sec=None, stop_sec=None):
    match OS_TYPE:
        case "nt":
            work_dir = "D:\\wk_messungen\\"
        case "posix":
            if MACHINE_ID == 'wowa-desktopL':
                # myuser = os.environ.get('USER')
                work_dir = "//mnt/Windows_data//wk_messungen//"
            elif MACHINE_ID == "wowa-backend":
                work_dir = "//mnt//datadrive//wk_messungen//"
            elif MACHINE_ID == "wowa-laptop-linux":
                work_dir = "//home//wowa//Schreibtisch//MA//dataset2_ai//"

        case _:
            work_dir = "//media//wowa//Windows Data//wk_messungen//"
    file_type = ".flac"
    audiofile_path = (work_dir + filename + file_type)
    my_sf = sf.SoundFile(file=audiofile_path)
    samplerate = my_sf.samplerate
    model_parameters.update({"my_samplingfrequency": samplerate})
    my_sf.close()
    full_signal, sampling_frequency = sf.read(audiofile_path, start=start_sec * samplerate, stop=stop_sec * samplerate,
                                              dtype="float32")
    # full_signal.astype(dtype=np.float16)
    signal_length = full_signal.shape[0] / sampling_frequency  # signal length in seconds
    full_time = np.linspace(start=start_sec, stop=stop_sec, num=full_signal.shape[0], dtype=np.float32)
    if USE_DEBUGPRINT:
        print(f"Opening file:{audiofile_path}")
        print(f"sampling_frequency = {sampling_frequency}")
        print(f"full_signal.shape = {full_signal.shape}")
        print(f"full_time.shape = {full_time.shape}")
        print(f"full signal length  = {signal_length} [s]")
    dframe = pd.DataFrame({'date': pd.Series(full_time, dtype=np.float32),
                           'close': pd.Series(full_signal, dtype=np.float16)})
    # return should be dictionary [date][close]
    return dframe

# Task 2: Load and Inspect the S&P 500 Index Data
# df = pd.read_csv('S&P_500_Index_Data.csv',parse_dates=['date'])
df = read_flac_to_pandas(filename=model_parameters["my_learningsequence"], start_sec=model_parameters['sequence_start'],
                         stop_sec=model_parameters['sequence_stop'])
df.head()
df.info()

# plot original time series snippet
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='close'))  # lines mode for lineplot
fig.update_layout(title='source time series - ' + timestr, xaxis_title="Time", yaxis_title='audio samples',
                  showlegend=True)
fig.write_html(out_dir + timestr + "_plot_timeseries.html")
# fig.show()

# Task 3: Data Preprocessing
# split data into train/test set
train_size = int(len(df) * model_parameters['train_test_split'])
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:]  # create index for train and test dataset
print("train.shape,test.shape: ", train.shape, test.shape)
scaler = StandardScaler()
scaler = scaler.fit(train[['close']])
train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])


# Task 4: Create Training and Test Splits
# sliding window implementation
def create_sequences(x_sequence, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x_sequence) - time_steps):
        v = x_sequence.iloc[i:(i + time_steps)].values
        xs.append(v)
        ys.append(y.iloc[i + time_steps])
    # TODO: apply hamming window here,
    # TODO: apply fft for sequence here
    return np.array(xs), np.array(ys)

try:
    print("creating sub-sequences")
    X_train, y_train = create_sequences(train[['close']], train['close'], model_parameters['time_steps'])
    X_test, y_test = create_sequences(test[['close']], test['close'], model_parameters['time_steps'])
    print("X_train.shape,y_train.shape: ", X_train.shape, y_train.shape)
    print("X_test.shape,y_test.shape: ", X_test.shape, y_test.shape)

    # Task 5: Build an LSTM Autoencoder
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]

    print("assembly model")
    # X = Sequential()
    # TODO: select model here
    my_model = LSTM_AE_model_delta2(X_train)
    my_model.compile(loss=model_parameters['my_loss'], optimizer=model_parameters['my_optimizer'])
    my_model.summary()

    tf.keras.utils.plot_model(my_model, show_shapes=True, to_file=out_dir + timestr + '_my_modelplot.png',
                              show_layer_names=True)

    # Task 6: Train the model

    early_stop = EarlyStopping(monitor=model_parameters['my_monitor'], patience=model_parameters['my_patience'],
                               mode=model_parameters['my_mode'])  # if the monitored metric does not change -> exit

    keras_callbacks = [
        EarlyStopping(monitor=model_parameters['my_monitor'], patience=model_parameters['my_patience'],
                      mode=model_parameters['my_mode'], min_delta=model_parameters["my_min_delta"]),
        ModelCheckpoint(tmp_dir, monitor=model_parameters['my_monitor'], save_best_only=True,
                        mode=model_parameters['my_mode'], verbose=model_parameters["my_verbose"])
    ]
    my_history = my_model.fit(X_train, y_train, epochs=model_parameters['my_epochs'],
                              batch_size=model_parameters['my_batch_size'],
                              validation_split=model_parameters['my_validation_split'], callbacks=keras_callbacks,
                              shuffle=False)

    tf.keras.models.save_model(my_model, out_dir + timestr + "_my_model.keras", overwrite=True)

    with open(out_dir + timestr + "_my_trainhistory.pckl", 'wb') as file_pi:
        pickle.dump(my_history, file_pi)

    # Task 7: Plot Metrics and Evaluate the Model

    # Load our saved model
    model_loaded = tf.keras.models.load_model(out_dir + timestr + "_my_model.keras", compile=True)
    with open(out_dir + timestr + "_my_trainhistory.pckl", "rb") as file_pi:
        my_history = pickle.load(file_pi)

    print("predicting anomaly")
    X_train_pred = model_loaded.predict(X_train)
    train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])
    X_test_pred = model_loaded.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    # Task 8: Detect Anomalies in Data
    test_score_df = pd.DataFrame(test[model_parameters['time_steps']:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = model_parameters['my_threshold']
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df[
        'threshold']  # this yields T/F but could be adapted to anomaly score
    test_score_df['close'] = test[model_parameters['time_steps']:]['close']
    test_score_df.head()
    test_score_df.tail()

    # this prints test loss over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[model_parameters['time_steps']:]['date'], y=test_score_df['loss'], mode='lines',
                             name='Test Loss'))
    fig.add_trace(go.Scatter(x=test[model_parameters['time_steps']:]['date'], y=test_score_df['threshold'], mode='lines',
                             name='Threshold'))
    fig.update_layout(title='training loss to time - ' + timestr, xaxis_title='Time', yaxis_title='Loss', showlegend=True)
    fig.write_html(out_dir + timestr + "_my_loss_time.html")
    fig.show()

    # marking anomaly true/false depending on test score
    anomalies = test_score_df[test_score_df['anomaly'] == True]
    anomalies.head()

    myfresh_x = test[model_parameters['time_steps']:]['date']
    # myfresh_y = test[time_steps:]['close']
    myfresh_y = scaler.inverse_transform(test[model_parameters['time_steps']:]['close'].values.reshape(-1, 1))

    print("shapes testdata x,y,:", myfresh_x.shape, myfresh_y.shape)
    myotherfresh_x = anomalies['date']
    # myotherfresh_y = anomalies['close']
    myotherfresh_y = scaler.inverse_transform(anomalies['close'].values.reshape(-1, 1))
    print("shapes anomalies x,y,:", myotherfresh_x.shape, myotherfresh_y.shape)

    # this should plot anomaly datapoints over original data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=myfresh_x, y=myfresh_y[:, 0], mode='lines', name='audio data points'))
    fig.add_trace(go.Scatter(x=myotherfresh_x, y=myotherfresh_y[:, 0], mode='markers', name='Anomaly'))
    fig.update_layout(title='Audio spectrum with anomalies - ' + timestr, xaxis_title='Time', yaxis_title='Audio spectrum',
                      showlegend=True)
    fig.write_html(out_dir + timestr + "_my_anomalies.html")
    fig.show()

    # this prints test loss over epochs
    fig = go.Figure()
    loss_ticks = np.arange(1, len(my_history.history['loss']))
    val_loss_ticks = np.arange(1, len(my_history.history['val_loss']))
    # fig.add_trace(go.Scatter(x=model_parameters["my_epochs"],y=train_mae_loss['Error'],mode='lines',name='train loss',))
    # fig.add_trace(go.Scatter(x=model_parameters["my_epochs"],y=test_mae_loss['Error'],mode='lines',name='test loss'))
    fig.add_trace(go.Scatter(y=my_history.history['loss'], x=loss_ticks, mode='lines', name='train loss'))
    fig.add_trace(go.Scatter(y=my_history.history['val_loss'], x=val_loss_ticks, mode='lines', name='test loss'))
    fig.update_layout(title='training loss to epochs - ' + timestr, xaxis_title='epochs', yaxis_title='loss',
                      showlegend=True)
    fig.write_html(out_dir + timestr + "_my_loss_epochs.html")
    fig.show()

    # saving hyperparameter
    runtime_stop = timeit.default_timer()
    runtime_time = runtime_stop - runtime_start
    model_parameters['my_trainingtime'] = runtime_time
    model_parameters['my_trainingsuccess'] = True
    save_training_parameters(model_parameters)
    print(f"Runtime: {runtime_time}")
    timestr_alternative = time.strftime("%Y%m%d-%H%M%S")
    if USE_ANOTHERTESTFILE:

        nok_sequence = read_flac_to_pandas(filename=model_parameters["my_predictsequence"], start_sec=model_parameters['my_nok_startsec'],
                                           stop_sec=model_parameters['my_nok_stopsec'])
        print("predicting anomaly in NOK sequence")

        nok_sequence_size = int(len(nok_sequence) * model_parameters['train_test_split'])
        # no test size. test_size = len(df) - train_size
        nok_sequence = nok_sequence.iloc[0:nok_sequence_size]

        nok_scaler = StandardScaler()
        nok_scaler = nok_scaler.fit(nok_sequence[['close']])
        nok_sequence['close'] = nok_scaler.transform(nok_sequence[['close']])

        nok_X_train, nok_y_train = create_sequences(nok_sequence[['close']], nok_sequence['close'],
                                                    model_parameters['time_steps'])
        nok_X_train_pred = model_loaded.predict(nok_X_train)
        nok_mae_loss = pd.DataFrame(np.mean(np.abs(nok_X_train_pred - nok_X_train), axis=1), columns=['Error'])

        # Task 8: Detect Anomalies in Data
        test_score_df = pd.DataFrame(nok_sequence[model_parameters['time_steps']:])
        test_score_df['loss'] = nok_mae_loss
        test_score_df['threshold'] = model_parameters['my_threshold']
        test_score_df['anomaly'] = test_score_df['loss'] > test_score_df[
            'threshold']  # this yields T/F but could be adapted to anomaly score
        test_score_df['close'] = nok_sequence[model_parameters['time_steps']:]['close']
        test_score_df.head()
        test_score_df.tail()
        nok_anomalies = test_score_df[test_score_df['anomaly'] == True]
        nok_anomalies.head()

        nok_myfresh_x = nok_sequence[model_parameters['time_steps']:]['date']
        nok_myfresh_y = nok_scaler.inverse_transform(
            nok_sequence[model_parameters['time_steps']:]['close'].values.reshape(-1, 1))

        print("shapes testdata x,y,:", nok_myfresh_x.shape, nok_myfresh_y.shape)
        nok_myotherfresh_x = nok_anomalies['date']
        nok_myotherfresh_y = nok_scaler.inverse_transform(nok_anomalies['close'].values.reshape(-1, 1))
        print("shapes anomalies x,y,:", nok_myotherfresh_x.shape, nok_myotherfresh_y.shape)

        # plot original nok time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nok_myfresh_x, y=nok_myfresh_y[:, 0], mode='lines', name='audio data points'))
        fig.update_layout(title='Audio spectrum with NOK anomalies - ' + timestr, xaxis_title='Time',
                          yaxis_title='Audio spectrum', showlegend=True)
        fig.write_html(out_dir + timestr + "_my_nok_timeseries." + timestr_alternative + ".html")
        fig.show()
        # this should plot anomaly datapoints over original data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nok_myfresh_x, y=nok_myfresh_y[:, 0], mode='lines', name='audio data points'))
        fig.add_trace(go.Scatter(x=nok_myotherfresh_x, y=nok_myotherfresh_y[:, 0], mode='markers', name='Anomaly'))
        fig.update_layout(title='Audio spectrum with NOK anomalies - ' + timestr, xaxis_title='Time',
                          yaxis_title='Audio spectrum', showlegend=True)
        fig.write_html(out_dir + timestr + "_predictedNOK_" + timestr_alternative + ".html")
        fig.show()
    if USE_FFT:
        def comp_stft(in_arr, fs, return_onesided, nperseg=None):

            in_arr = cp.array(in_arr.astype(np.float16))
            win = cp.hanning(in_arr.shape[0]).astype(cp.float16)
            in_arr = in_arr * win
            # f, t, spectrogram = cp.fft.rfft(
            #     in_arr,
            #     fs=fs,
            #     return_onesided=return_onesided,
            #     #boundary="zeros",
            #     #padded=True,
            # )
            f, t, spectrogram = cusignal.spectrogram(in_arr,
                                                     fs=fs,
                                                     return_onesided=return_onesided,
                                                     nperseg=nperseg,
                                                     mode="magnitude"
                                                     )
            # spect = cp.fft.rfft(in_arr)
            # spectrogram = cp.abs(spect)
            # f = cp.float16(f.get())  # cupy to numpy has to be explicit
            # t = cp.float16(t.get())
            # spectrogram = cp.float32(spectrogram.get())
            return f.get(), t.get(), spectrogram.get()


        def create_spectrogram(x_in, x_ticks, samplerate_in):
            # spectogram parameters. adjust to improve plot quality
            real_only = True
            nperseg = int(
                256 * 16)  # samplerate_in//x_in.shape[0]  #  frequency resolution TODO: make adaptive on len(x_in)
            print("frequency/time resolution:", nperseg)
            ticks_offset = x_ticks[0]
            if USE_CUDA:
                # CuSignal version, requires building cusignal.
                # y_arr, x_arr, spectrogram_arr = sps.spectrogram(x_in, samplerate_in, return_onesided=real_only)

                y_arr, x_arr, spectrogram_arr = comp_stft(x_in, fs=samplerate_in, return_onesided=real_only,
                                                          nperseg=nperseg)
            else:
                # y_arr, x_arr, spectrogram_arr = sps.spectrogram(x_in, fs=samplerate_in, return_onesided=real_only, nperseg=nperseg)
                y_arr, x_arr, spectrogram_arr = sps.spectrogram(x_in, fs=samplerate_in, return_onesided=real_only,
                                                                nperseg=nperseg)

            # returns arrays of stample frequency, array of time steps, spectrogram of x.
            # Last axis of spectrogram corresponds to array of times
            return y_arr, x_arr + ticks_offset, spectrogram_arr

        start_time = timeit.default_timer()
        spectrogram_frequency, spectrogram_time, spectrogram_map = create_spectrogram(df['close'], df['date'],
                                                                                      model_parameters[
                                                                                          'my_samplingfrequency'])
        print(timeit.default_timer() - start_time)

        print("plotting spectrogram")
        print("map shape:", spectrogram_map.shape)
        print("date shape:", df.shape)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(spectrogram_map)
        spectrogram_map = scaler.transform(spectrogram_map)
        # Plot with plotly
        trace = [go.Heatmap(
            x=spectrogram_time,
            y=spectrogram_frequency,
            z=spectrogram_map,
            colorscale='Jet'
        )]
        layout = go.Layout(
            title='Spectrogram',
            yaxis=dict(title='Frequency'),  # x-axis label
            xaxis=dict(title='Time'),  # y-axis label
            # yaxis_type="log"
        )
        fig = go.Figure(data=trace, layout=layout)
        # fig.update_yaxes(title_text="Frequency in logarithmic scale", type="log")

        # fig.write_html(out_dir + "00_spectrogram_"+timestr+".html")
        fig.update_layout(title='Spectrogram - ' + timestr, showlegend=True)
        fig.write_image(out_dir + timestr + "_spectrogram_" + timestr_alternative + ".png")

        fig.show()

except Exception as error:
    print("sending failed mail with following error:")
    print(error)
    send_error_notification(error)
else:
    print("sending ok mail")
    send_finish_notification()
