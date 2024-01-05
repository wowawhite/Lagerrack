# Task 1: Import Libraries

import numpy as np
import soundfile as sf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sys
import tensorflow as tf
from pathlib import Path
import platform
import time
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
import keras as kr
from LSTM_AE_custom_models import *
from Notification_module import *

print("Select .keras model file")
root = tk.Tk()
root.withdraw()
model_file_path = filedialog.askopenfilename()
print("Using .keras model file:", model_file_path)

# program control flags
USE_CUDA = True
USE_DEBUGPRINT = True
USE_STARTSCRIPT = False
USE_FFT = False
parent_dir = Path(__file__).resolve().parent
out_dir = str(Path.joinpath((parent_dir),"output"))+os.sep

timestr = time.strftime("%Y%m%d-%H%M%S")
model_parameters = dict(
    # Model variables are set here:
    Timestamp=timestr,  # timestring for identification
    # data preparation
    my_learningsequence="visc6_ultrasonic_nok", #visc6_ultrasonic_ok visc6_nosonic_ok
    my_samplingfrequency=0,  # automatic detection ok
    sequence_start=8159,  # start second in audio file for  subsequence analysis
    sequence_stop=8162,  # stop second in audio file for subsequence analysis
    train_test_split=0.8,  # 80/20 split for training/testing set
    time_steps=30,  # 30 size of sub-sequences for LSTM feeding
    # model learining
    my_epochs=20,  # 10
    my_batch_size=32,  # 32  dimensions of time steps for 2d input pattern
    my_validation_split=0.2,  # 0.1
    # my_dropout=0.2, #  model-depending, not global. likely not useful for sequences
    # model quality criteria
    my_loss='mae',
    my_optimizer='adam',
    # anomaly detection
    my_threshold=2.0,
    # early stop paramerers
    my_monitor='val_loss',
    my_patience=3,
    my_mode='min',
    my_predictsequence="visc6_ultrasonic_nok",  # visc6_ultrasonic_nok visc6_nosonic_nok
    my_nok_startsec=6416,  # same for visc6_ultrasonic_nok and visc6_nosonic_nok
    my_nok_stopsec=6428,  # same for visc6_ultrasonic_nok and visc6_nosonic_nok
     my_traintime='',
    my_ostype='',
    my_cudaversion='',
    my_fftusage=False,
    my_pythonversion=sys.version,
    my_tensorflowversion=tf.__version__,
    my_kerasversion=kr.__version__
)

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    # TODO: apply hamming window here,
    # TODO: apply fft for sequence here
    return np.array(Xs), np.array(ys)

OS_TYPE = os.name
MACHINE_ID = platform.node()
print(f"OS_TYPE: {OS_TYPE}")
print(f"MACHINE_ID: {MACHINE_ID}")
print(f"Working directory: {parent_dir}{os.sep}")
print(f"Working directory: {out_dir}")
np.random.seed(1)
tf.random.set_seed(1)
if USE_CUDA:
    print("Enable CUDA Support")
    import cupy as cp  # install pip cupy-cuda12x
    feedback_cuda = cp.cuda.runtime.driverGetVersion()
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"CUDA driver version is {feedback_cuda}")
    print("Cuda devices available:",tf.config.list_physical_devices('GPU'))
else:
    print(f"Using CPU only")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Tensorflow version:', tf.__version__)

# get parent path


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

# Task 2: Load model

# Load our saved model
parent_dir = Path(__file__).resolve().parent
out_dir = str(Path.joinpath((parent_dir),"output"))+os.sep

timetag = model_file_path[-30:-15]  # parsing timestamp from keras model filename

history_file_path = out_dir+timetag+"_my_trainhistory.pckl"
model_loaded = tf.keras.models.load_model(model_file_path, compile=True)
model_loaded.summary()
with open(history_file_path, "rb") as file_pi:
    my_history = pickle.load(file_pi)

# Task 3: Load target file
nok_sequence = read_flac_to_pandas(filename=model_parameters["my_predictsequence"],start_sec=model_parameters["my_nok_startsec"], stop_sec=model_parameters["my_nok_stopsec"])
try:
    print("predicting anomaly in NOK sequence")
    nok_sequence_size = int(len(nok_sequence))
    nok_sequence = nok_sequence.iloc[0:nok_sequence_size]
    nok_scaler = StandardScaler()
    nok_scaler = nok_scaler.fit(nok_sequence[['close']])
    nok_sequence['close'] = nok_scaler.transform(nok_sequence[['close']])
    nok_X_train, nok_y_train = create_sequences(nok_sequence[['close']], nok_sequence['close'], model_parameters['time_steps'])
    nok_X_train_pred = model_loaded.predict(nok_X_train)
    nok_mae_loss = pd.DataFrame(np.mean(np.abs(nok_X_train_pred - nok_X_train),axis=1),columns=['Error'])

    #Task 8: Detect Anomalies in Data
    test_score_df = pd.DataFrame(nok_sequence[model_parameters['time_steps']:])
    test_score_df['loss'] = nok_mae_loss
    test_score_df['threshold'] = model_parameters['my_threshold']
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']  # this yields T/F but could be adapted to anomaly score
    test_score_df['close'] = nok_sequence[model_parameters['time_steps']:]['close']
    test_score_df.head()
    test_score_df.tail()
    nok_anomalies = test_score_df[test_score_df['anomaly'] == True]
    nok_anomalies.head()

    nok_myfresh_x = nok_sequence[model_parameters['time_steps']:]['date']
    nok_myfresh_y = nok_scaler.inverse_transform(nok_sequence[model_parameters['time_steps']:]['close'].values.reshape(-1, 1))

    print("shapes testdata x,y,:", nok_myfresh_x.shape, nok_myfresh_y.shape)
    nok_myotherfresh_x = nok_anomalies['date']
    nok_myotherfresh_y = nok_scaler.inverse_transform(nok_anomalies['close'].values.reshape(-1, 1))
    print("shapes anomalies x,y,:", nok_myotherfresh_x.shape, nok_myotherfresh_y.shape)
    # plot original nok time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nok_myfresh_x, y=nok_myfresh_y[:, 0], mode='lines', name='audio data points'))
    fig.update_layout(title='Audio spectrum with NOK anomalies - ' + timetag, xaxis_title='Time',
                      yaxis_title='Audio spectrum', showlegend=True)
    fig.write_html(out_dir + timetag + "_my_nok_timeseries.html")
    fig.show()

    # plot anomaly datapoints over original data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nok_myfresh_x, y=nok_myfresh_y[:, 0], mode='lines', name='audio data points'))
    fig.add_trace(go.Scatter(x=nok_myotherfresh_x, y=nok_myotherfresh_y[:, 0], mode='markers', name='Anomaly'))
    fig.update_layout(title='Audio spectrum with NOK anomalies - ' + timetag, xaxis_title='Time',
                      yaxis_title='Audio spectrum', showlegend=True)
    fig.write_html(out_dir + timetag + "_my_nok_anomalies.html")
    fig.show()
except Exception as error:
    print("sending failed mail with following error:")
    print(error)
    send_error_notification(error)
else:
    print("sending ok mail")
    send_finish_notification()
print("done.EOF")
