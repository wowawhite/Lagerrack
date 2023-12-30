from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
import sys
# https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb
def LSTM_AE_model_alpha1(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]

    model.add(LSTM(128,input_shape=(timesteps,num_features), name='LSTM_AE_model_alpha1'))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    #model = Model(inputs=inputs, outputs=output)
    return model

def LSTM_AE_model_alpha2(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='LSTM_AE_model_alpha2'))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    #model = Model(inputs=inputs, outputs=output)
    return model
def LSTM_AE_model_alpha3(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='LSTM_AE_model_alpha3'))
    model.add(LSTM(128, activation='relu',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    #model = Model(inputs=inputs, outputs=output)
    return model
def LSTM_AE_model_beta1(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='LSTM_AE_model_beta1'))
    model.add(LSTM(16, activation='relu', return_sequences=True,kernel_regularizer=regularizers.l2(0.00)))
    model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

# https://www.kaggle.com/code/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders
def LSTM_AE_model_gamma1(model, inputs, hyperparameters=None):
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='LSTM_AE_model_gamma1'))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(6, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(1))
    return

# https://github.com/thomashuang02/LSTM-Autoencoder-for-Time-Series-Anomaly-Detection/blob/main/lstm_autoencoder.ipynb
def LSTM_AE_model_delta1(model, inputs, hyperparameters=None):
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='LSTM_AE_model_delta1'))
    model.add(LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='encoder_L1'))
    model.add(LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_L2'))
    model.add(LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_L3'))
    model.add(RepeatVector(timesteps, name='encoder_decoder_bridge'))
    #decoder_input = Input(shape=(timesteps, 16), name='decoder_input')
    model.add(LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L1'))
    model.add(LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L2'))
    model.add(LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L3'))
    model.add(TimeDistributed(Dense(num_features)))
    return model

# https://machinelearningmastery.com/lstm-autoencoders/
# TODO: this one is incomplete
def LSTM_AE_model_epsilon1(model, inputs, hyperparameters=None):
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, 1),name='LSTM_AE_model_epsilon1'))
    model.add(LSTM(100, activation='relu'))
    # define reconstruct decoder
    model.add(RepeatVector(timesteps))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    # define predict decoder
    model.add(RepeatVector(num_features))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    # tie it together
    return model
