from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from tensorflow.keras.models import Sequential
import sys
# https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb
# Requirements for CUDNN-optimizer:
# activation == tanh
# recurrent_activation == sigmoid
# recurrent_dropout == 0
# unroll is False
# use_bias is True
# Inputs, if use masking, are strictly right-padded.
# Eager execution is enabled in the outermost context.

def LSTM_AE_model_alpha1(inputs, hyperparameters=None):
    my_modelname = "alpha1"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(LSTM(128,input_shape=(timesteps,num_features),name ="LSTM_AE_model_alpha1"))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_alpha2(inputs, hyperparameters=None):
    my_modelname = "alpha2"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_alpha2"))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    return model
def LSTM_AE_model_alpha3(inputs, hyperparameters=None):
    my_modelname = "alpha3"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_alpha3"))
    model.add(LSTM(128, activation='relu',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_alpha4(inputs, hyperparameters=None):
    my_modelname = "alpha4"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_alpha4"))
    model.add(LSTM(128, activation='softmax',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128, activation='softmax', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_alpha5(inputs, hyperparameters=None):
    my_modelname = "alpha5"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_alpha5"))
    model.add(LSTM(128,return_sequences=False))
    model.add(RepeatVector(30))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

# https://github.com/Jithsaavvy/Explaining-deep-learning-models-for-detecting-anomalies-in-time-series-data-RnD-project
def LSTM_AE_model_beta1( inputs, hyperparameters=None):
    my_modelname = "beta1"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_beta1"))
    model.add(LSTM(16, activation='relu', return_sequences=True,kernel_regularizer=regularizers.l2(0.00)))
    model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_beta2(inputs, hyperparameters=None):
    my_modelname = "beta2"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_beta2"))
    model.add(LSTM(16, activation='softmax', return_sequences=True,kernel_regularizer=regularizers.l2(0.00)))
    model.add(LSTM(4, activation='softmax', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(4, activation='softmax', return_sequences=True))
    model.add(LSTM(16, activation='softmax', return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_beta3(inputs, hyperparameters=None):
    my_modelname = "beta3"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_beta3"))
    model.add(LSTM(16, return_sequences=True,kernel_regularizer=regularizers.l2(0.00)))
    model.add(LSTM(4,  return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

# https://www.kaggle.com/code/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders
# gamma models incomplete and need debugging
def LSTM_AE_model_gamma1(inputs, hyperparameters=None):
    my_modelname = "gamma1"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    # model.name ="LSTM_AE_model_gamma1"
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_gamma1"))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(6, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(1))
    return model

def LSTM_AE_model_gamma2( inputs, hyperparameters=None):
    my_modelname = "gamma2"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    # model.name ="LSTM_AE_model_gamma1"
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_gamma2"))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(6,  return_sequences=True))
    model.add(LSTM(1))
    model.add(Dense(10, kernel_initializer='glorot_normal'))
    model.add(Dense(10, kernel_initializer='glorot_normal'))
    model.add((Dense(1)))
    return model

def LSTM_AE_model_gamma3(inputs, hyperparameters=None):
    my_modelname = "gamma3"
    model = Sequential(name=my_modelname)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_gamma3"))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(6, activation='softmax', return_sequences=True))
    model.add(LSTM(1, activation='softmax'))
    model.add(Dense(10, kernel_initializer='glorot_normal', return_sequences=True))
    model.add(Dense(10, kernel_initializer='glorot_normal', return_sequences=False))
    model.add((Dense(1)))
    return model
# https://github.com/thomashuang02/LSTM-Autoencoder-for-Time-Series-Anomaly-Detection/blob/main/lstm_autoencoder.ipynb
# Delta model promising.
def LSTM_AE_model_delta1( inputs, hyperparameters=None):
    my_modelname = "delta1"
    model = Sequential(name=my_modelname)
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]

    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_delta1"))
    # https://keras.io/api/layers/initializers/ glorot_uniform is default
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

def LSTM_AE_model_delta2(inputs, hyperparameters=None):
    my_modelname = "delta2"
    model= Sequential(name=my_modelname)
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name =my_modelname))
    # https://keras.io/api/layers/initializers/
    model.add(LSTM(64, kernel_initializer='glorot_normal', return_sequences=True, name='encoder_L1'))
    model.add(LSTM(32, kernel_initializer='glorot_normal', return_sequences=True, name='encoder_L2'))
    model.add(LSTM(16, kernel_initializer='glorot_normal', return_sequences=False, name='encoder_L3'))
    model.add(RepeatVector(timesteps, name='encoder_decoder_bridge'))
    #decoder_input = Input(shape=(timesteps, 16), name='decoder_input')
    model.add(LSTM(16, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L1'))
    model.add(LSTM(32, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L2'))
    model.add(LSTM(64, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L3'))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_delta3(inputs, hyperparameters=None):
    # TODO: https://stackoverflow.com/questions/62728083/change-the-model-name-given-automatically-by-keras-in-model-summary-output/62728323#62728323
    my_modelname = "delta3"
    model= Sequential(name=my_modelname)
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name =my_modelname))
    # https://keras.io/api/layers/initializers/
    model.add(LSTM(128, kernel_initializer='glorot_normal', return_sequences=True, name='encoder_L1'))
    model.add(LSTM(64, kernel_initializer='glorot_normal', return_sequences=True, name='encoder_L1'))
    model.add(LSTM(32, kernel_initializer='glorot_normal', return_sequences=True, name='encoder_L2'))
    model.add(LSTM(16, kernel_initializer='glorot_normal', return_sequences=False, name='encoder_L3'))
    model.add(RepeatVector(timesteps, name='encoder_decoder_bridge'))
    #decoder_input = Input(shape=(timesteps, 16), name='decoder_input')
    model.add(LSTM(16, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L1'))
    model.add(LSTM(32, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L2'))
    model.add(LSTM(64, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L3'))
    model.add(LSTM(128, kernel_initializer='glorot_normal', return_sequences=True, name='decoder_L3'))
    model.add(TimeDistributed(Dense(num_features)))
    return model

# https://machinelearningmastery.com/lstm-autoencoders/
# TODO: LSTM_AE_model_epsilon is incomplete and thows error, see:
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 1228768) + inhomogeneous part.
def LSTM_AE_model_epsilon1(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    seq_out = inputs[:, 1:, :]
    n_out = timesteps-1
    # model.add(Input(shape=(timesteps, 1),name='LSTM_AE_model_epsilon1'))
    # model.add(LSTM(100, activation='sigmoid',return_sequences=True))
    # # define reconstruct decoder
    # model.add(RepeatVector(1))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # # define predict decoder
    # model.add(RepeatVector(num_features))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))

    # define encoder
    visible = Input(shape=(timesteps,1), name ="LSTM_AE_model_epsilon1")
    encoder = LSTM(100, activation='relu')(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(timesteps)(encoder)
    decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    return model

def LSTM_AE_model_epsilon2(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    seq_out = inputs[:, 1:, :]
    n_out = timesteps
    # define encoder
    visible = Input(shape=(timesteps,num_features), name ="LSTM_AE_model_epsilon2")
    encoder = LSTM(100)(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(timesteps)(encoder)
    decoder1 = LSTM(100, return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(100, return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    return model


def LSTM_AE_model_epsilon3(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    seq_out = inputs[:, 1:, :]
    n_out = timesteps
    # define encoder
    visible = Input(shape=(timesteps,num_features), name ="LSTM_AE_model_epsilon3")
    encoder = LSTM(100)(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(timesteps)(encoder)
    decoder1 = LSTM(100, return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(num_features))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(100, return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(num_features))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    return model

def LSTM_AE_model_epsilon4(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    seq_out = inputs[:, 1:, :]
    n_out = timesteps
    # define encoder
    visible = Input(shape=(timesteps,1), name ="LSTM_AE_model_epsilon4")
    encoder = LSTM(100)(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(timesteps)(encoder)
    decoder1 = LSTM(100, return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(100, return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    return model

def LSTM_AE_model_zeta1(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_zeta1"))
    # https://keras.io/api/layers/initializers/
    model.add(LSTM(128, return_sequences=True, name='L1'))
    model.add(LSTM(128, return_sequences=True, name='L2'))
    model.add(LSTM(128, return_sequences=True, name='L3'))
    model.add(LSTM(128, return_sequences=True, name='L4'))
    model.add(LSTM(128, return_sequences=True, name='L5'))
    model.add(LSTM(128, return_sequences=True, name='L6'))
    model.add(LSTM(128, return_sequences=True, name='L7'))
    model.add(LSTM(128, return_sequences=True, name='L8'))
    model.add(LSTM(128, return_sequences=True, name='L9'))
    model.add(LSTM(128, return_sequences=True, name='L10'))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def LSTM_AE_model_zeta2(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name ="LSTM_AE_model_zeta2"))
    # https://keras.io/api/layers/initializers/
    model.add(LSTM(128, return_sequences=True, name='L1'))
    model.add(LSTM(64, return_sequences=True, name='L2'))
    model.add(LSTM(32, return_sequences=True, name='L3'))
    model.add(LSTM(64, return_sequences=True, name='L4'))
    model.add(LSTM(128, return_sequences=True, name='L5'))

    model.add(TimeDistributed(Dense(num_features)))
    return model