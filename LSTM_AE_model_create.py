# https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb

# TODO: hamming window


# Task 1: Import Libraries

import numpy as np
import soundfile as sf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import os
import pickle
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from pathlib import Path
import platform
import time

USE_CUDA = True
USE_DEBUGPRINT = True

OS_TYPE = os.name
MACHINE_ID = platform.node()
print(f"OS_TYPE: {OS_TYPE}")
print(f"MACHINE_ID: {MACHINE_ID}")

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


def read_flac_to_pandas(filename):
    parent = Path(__file__).resolve().parent

    match OS_TYPE:
        case "nt":
            work_dir = ("https://youtu.be/KSG_-PqzHnM?si=L7j4wEsRZJYoN3aF&t=12")
        case "posix":
            if MACHINE_ID=='wowa-desktopL':
                #myuser = os.environ.get('USER')
                work_dir = ("//media//wowa//Windows Data//wk_messungen//")
            elif(MACHINE_ID=="wowa-backend"):
                work_dir = ("//mnt//datadrive//wk_messungen//")

        case _:
            work_dir = ("//media//wowa//Windows Data//wk_messungen//")
    file_type = ".flac"
    audiofile_path = ( work_dir + filename + file_type)

    full_signal, sampling_frequency = sf.read(audiofile_path)
    full_signal.astype(dtype=np.float16)
    signal_length = full_signal.shape[0] / sampling_frequency  # signal length in seconds
    delta_timestep = signal_length / full_signal.shape[0]
    full_time = np.linspace(0,signal_length,full_signal.shape[0] ,dtype=np.float32)



    my_array = np.vstack((full_time, full_signal)).T
    if (USE_DEBUGPRINT):
        print(f"Opening file:{audiofile_path}")
        print(f"sampling_frequency = {sampling_frequency}")
        print(f"full_signal.shape = {full_signal.shape}")
        print(f"full_time.shape = {full_time.shape}")
        print(f"full signal length  = {signal_length} [s]")

    # dframe = pd.DataFrame(my_array, columns=['date', 'close'])
    dframe = pd.DataFrame({'date': pd.Series(full_time, dtype=np.float32),
                       'close': pd.Series(full_signal, dtype=np.float16)})
    #return should be dictionary [date][close]
    return dframe

# Task 2: Load and Inspect the S&P 500 Index Data

#df = pd.read_csv('S&P_500_Index_Data.csv',parse_dates=['date'])
df = read_flac_to_pandas("visc6_ultrasonic_ok") [(384000*10):(384000*110)]

df.head()
df.info()

# using Plotly for interactive graphs
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'],y=df['close'],mode='lines',name='close')) # lines mode for lineplot
fig.update_layout(title='S&P 500',xaxis_title="Time",yaxis_title='INDEXSP',showlegend=True)
fig.show()

#Task 3: Data Preprocessing

# Model variables are set here:
#data preparation
sequence_start = 30
sequence_stop = 330
train_test_split = 0.8  # 80/20 split for training/testing set
time_steps = 30  # size of sub-sequences for LSTM feeding
#model learining
my_epochs=10  # 10
my_batch_size=32  #32
my_validation_split=0.1  # 0.1
# model quality criteria
my_loss='mae'
my_optimizer='adam'

# split data into train/test set
train_size = int(len(df) * train_test_split)
test_size = len(df) - train_size

train, test = df.iloc[0:train_size], df.iloc[train_size:]  # create index for train and test dataset

print("train.shape,test.shape: ",train.shape,test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['close']])

train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])
#
#Task 4: Create Training and Test Splits

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    # TODO: apply hamming window here
    return np.array(Xs), np.array(ys)

print("creating sub-sequences")
X_train, y_train = create_sequences(train[['close']],train['close'],time_steps)
X_test, y_test = create_sequences(test[['close']],test['close'],time_steps)
print("X_train.shape,y_train.shape: ", X_train.shape,y_train.shape)

# Task 5: Build an LSTM Autoencoder

timesteps = X_train.shape[1]
num_features = X_train.shape[2]
timestr = time.strftime("%Y%m%d-%H%M%S")

print("assembly model")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model = Sequential()
model.add(LSTM(128,input_shape=(timesteps,num_features))) # TODO: what are 128 units for?
model.add(Dropout(0.1))
model.add(RepeatVector(timesteps)) # Repeats the input n times.
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(num_features)))  # apply a layer to every temporal slice of an input.

model.compile(loss=my_loss,optimizer=my_optimizer)
model.summary()
tf.keras.utils.plot_model(model, to_file='my_modelplot_'+timestr+'.png', show_shapes=True, show_layer_names=True)

# Task 6: Train the Autoencoder

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=3,mode='min') # if the monitored metric does not change wrt to the mode applied
my_history = model.fit(X_train,y_train,epochs=my_epochs,batch_size=my_batch_size,validation_split=my_validation_split,callbacks=[early_stop],shuffle=False)
#my_history = model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.1,shuffle=False)

# save model while training with checkpoints
# https://keras.io/api/callbacks/model_checkpoint/
# https://www.tensorflow.org/tutorials/keras/save_and_load

# list all data in history# saving model for later use
# list all data in history
# alternative save https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# https://stackoverflow.com/questions/66827371/difference-between-tf-saved-model-savemodel-path-to-dir-and-tf-keras-model-sa

tf.keras.models.save_model( model, "/home/wowa/PycharmProjects/Lagerrack/my_model_"+timestr+".keras",overwrite=True)

with open('my_trainhistory'+timestr, 'wb') as file_pi:
    pickle.dump(my_history, file_pi)

# Task 7: Plot Metrics and Evaluate the Model

# Load our saved model
model_loaded = tf.keras.models.load_model("/home/wowa/PycharmProjects/Lagerrack/my_model"+timestr+".keras", compile=True)
with open('my_trainhistory'+timestr, "rb") as file_pi:
    my_history = pickle.load(file_pi)


print("my_history.history.keys()\n",my_history.history.keys())  # prints dict_keys(['loss', 'val_loss'])
print(f"model hist is : \n {my_history.history}")

err = pd.DataFrame(my_history.history)
err.plot()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')

# Threshold for anomaly marker TODO: play around to improve
threshold = 0.90

# Calculating the mae for training data
sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 16, 6 # set figsize for all images

# TODO: bug when plotting with seaborn. Window is not shown
X_train_pred = model_loaded.predict(X_train)
train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train),axis=1),columns=['Error'])
sns.distplot(train_mae_loss,bins=50,kde=True)  # Plot histogram of training losses

X_test_pred = model_loaded.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test),axis=1)
sns.distplot(test_mae_loss, bins=50, kde=True)  # Plot histogram of test losses / KDE=kernel density estimation

#Task 8: Detect Anomalies in the S&P 500 Index Data

test_score_df = pd.DataFrame(test[time_steps:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']  # this yields T/F but could be adapted to anomaly score
test_score_df['close'] = test[time_steps:]['close']
test_score_df.head()
test_score_df.tail()


fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:]['date'],y=test_score_df['loss'],mode='lines',name='Test Loss'))
fig.add_trace(go.Scatter(x=test[time_steps:]['date'],y=test_score_df['threshold'],mode='lines',name='Threshold'))
fig.update_layout(xaxis_title='Time',yaxis_title='Loss',showlegend=True)
fig.show()

anomalies = test_score_df[test_score_df['anomaly'] == True]
anomalies.head()

fig = go.Figure()

myfresh_x = test[time_steps:]['date']
#myfresh_y = test[time_steps:]['close']
myfresh_y = scaler.inverse_transform(test[time_steps:]['close'].values.reshape(-1,1))

print("shapes testdata x,y,:", myfresh_x.shape, myfresh_y.shape)
# this should plot the original test data
fig.add_trace(go.Scatter(x=myfresh_x, y=myfresh_y[:,0], mode='lines',name='Close Price'))

myotherfresh_x = anomalies['date']
#myotherfresh_y = anomalies['close']
myotherfresh_y = scaler.inverse_transform(anomalies['close'].values.reshape(-1,1))
print("shapes anomalies x,y,:", myotherfresh_x.shape, myotherfresh_y.shape)
# this should plot anomaly datapoints
fig.add_trace(go.Scatter(x=myotherfresh_x,y=myotherfresh_y[:,0],mode='markers',name='Anomaly'))

fig.update_layout(title='S&P 500 with Anomalies',xaxis_title='Time',yaxis_title='INDEXSP',showlegend=True)
fig.show()

print("ok")