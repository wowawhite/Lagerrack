# https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb

# Task 1: Import Libraries

import numpy as np
import tensorflow as tf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 16, 6 # set figsize for all images

np.random.seed(1)
tf.random.set_seed(1)

print('Tensorflow version:', tf.__version__)

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

train_size = int(len(df) * 0.8) # 80% size for training set
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:]

print(train.shape,test.shape)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar = scalar.fit(train[['close']])

train['close'] = scalar.transform(train[['close']])
test['close'] = scalar.transform(test[['close']])

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
model.add(TimeDistributed(Dense(num_features))) # apply a layer to every temporal slice of an input.

model.compile(loss='mae',optimizer='adam')
model.summary()

# Task 6: Train the Autoencoder

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=3,mode='min') # if the monitored metric does not change wrt to the mode applied
history = model.fit(X_train,y_train,epochs=100,batch_size=32,validation_split=0.1,callbacks=[early_stop],shuffle=False)

# saving model for later use
model.save('anomaly_model.h5')

# Task 7: Plot Metrics and Evaluate the Model

# Load our saved model
history = tf.keras.models.load_model('anomaly_model.h5')

err = pd.DataFrame(history.history)
err.plot()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')

# Calculating the mae for training data
X_train_pred = model.predict(X_train)
train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train),axis=1),columns=['Error'])
sns.distplot(train_mae_loss,bins=50,kde=True) # Plot histogram of traning losses
threshold = 0.65


# Calculate mae for test data
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test),axis=1)
sns.distplot(test_mae_loss, bins=50, kde=True); # Plot histogram of test losses

#Task 8: Detect Anomalies in the S&P 500 Index Data

test_score_df = pd.DataFrame(test[time_steps:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['close'] = test[time_steps:]['close']
test_score_df.head()
test_score_df.tail()
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:]['date'],y=test_score_df['loss'],mode='lines',name='Test Loss'))
fig.add_trace(go.Scatter(x=test[time_steps:]['date'],y=test_score_df['threshold'],mode='lines',name='Threshold'))
fig.update_layout(xaxis_title='Time',yaxis_title='Loss',showlegend=True)
fig.show()
anomalies = test_score_df[test_score_df['anomaly']==True]
anomalies.head()
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:]['date'],y=scalar.inverse_transform(test[time_steps:]['close']),mode='lines',name='Close Price'))
fig.add_trace(go.Scatter(x=anomalies['date'],y=scalar.inverse_transform(anomalies['close']),mode='markers',name='Anomaly'))
fig.update_layout(title='S&P 500 with Anomalies',xaxis_title='Time',yaxis_title='INDEXSP',showlegend=True)
fig.show()

