# https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb

# Task 1: Import Libraries

import numpy as np

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

USE_CUDA = True


OS_TYPE = os.name
print(f"OS_TYPE: {OS_TYPE}")
sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 16, 6 # set figsize for all images

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
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
print('Tensorflow version:', tf.__version__)

# get parent path
path_parent = Path(__file__).resolve().parent

print("type path_parent:", type(path_parent))
print("path_parent:", path_parent)


# Task 2: Load and Inspect the S&P 500 Index Data

df = pd.read_csv('S&P_500_Index_Data.csv',parse_dates=['date'])
df.head()
df.info()

# using Plotly for interactive graphs
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'],y=df['close'],mode='lines',name='close')) # lines mode for lineplot
fig.update_layout(title='S&P 500',xaxis_title="Time",yaxis_title='INDEXSP',showlegend=True)
fig.show()

#Task 3: Data Preprocessing

# split data into train/test set
train_size = int(len(df) * 0.8) # 80% size for training set
test_size = len(df) - train_size

train, test = df.iloc[0:train_size], df.iloc[train_size:]

print(train.shape,test.shape)

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
    return np.array(Xs), np.array(ys)

time_steps = 30
X_train, y_train = create_sequences(train[['close']],train['close'],time_steps)
X_test, y_test = create_sequences(test[['close']],test['close'],time_steps)
print(X_train.shape,y_train.shape)

# Task 5: Build an LSTM Autoencoder

timesteps = X_train.shape[1]
num_features = X_train.shape[2]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model = Sequential()
model.add(LSTM(128,input_shape=(timesteps,num_features)))
model.add(Dropout(0.2))
model.add(RepeatVector(timesteps)) # Repeats the input n times.
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(num_features)))  # apply a layer to every temporal slice of an input.

model.compile(loss='mae',optimizer='adam')
model.summary()

# Task 6: Train the Autoencoder

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=3,mode='min') # if the monitored metric does not change wrt to the mode applied
my_history = model.fit(X_train,y_train,epochs=1,batch_size=32,validation_split=0.1,callbacks=[early_stop],shuffle=False)
#my_history = model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.1,shuffle=False)

# save model while training with checkpoints
# https://keras.io/api/callbacks/model_checkpoint/
# https://www.tensorflow.org/tutorials/keras/save_and_load


# list all data in history# saving model for later use
# list all data in history
# alternative save https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# https://stackoverflow.com/questions/66827371/difference-between-tf-saved-model-savemodel-path-to-dir-and-tf-keras-model-sa

# should print sth like ['accuracy', 'loss', 'val_accuracy', 'val_loss']
# TODO: BUG: saving model deletes history
tf.keras.models.save_model( model, "/home/wowa/PycharmProjects/Lagerrack/my_model.keras",overwrite=True)

with open('my_trainhistory', 'wb') as file_pi:
    pickle.dump(my_history, file_pi)

# Task 7: Plot Metrics and Evaluate the Model

# Load our saved model
model_loaded = tf.keras.models.load_model("/home/wowa/PycharmProjects/Lagerrack/my_model.keras", compile=True)
with open('my_trainhistory', "rb") as file_pi:
    my_history = pickle.load(file_pi)


print("my_history.history.keys()\n",my_history.history.keys())  # prints dict_keys(['loss', 'val_loss'])
print(f"model hist is : \n {my_history.history}")

err = pd.DataFrame(my_history.history)
err.plot()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')

# Calculating the mae for training data
X_train_pred = model.predict(X_train)
train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train),axis=1),columns=['Error'])
sns.distplot(train_mae_loss,bins=50,kde=True)  # Plot histogram of training losses
threshold = 0.65


# Calculate mae for test data
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test),axis=1)
sns.distplot(test_mae_loss, bins=50, kde=True)  # Plot histogram of test losses

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