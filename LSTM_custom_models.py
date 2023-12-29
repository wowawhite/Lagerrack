from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

def autoencoder_model_alpha(inputs, hyperparameter=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]

    L1 =LSTM(128,input_shape=(timesteps,num_features))
    L2 =Dropout(0.1)(L1)
    L3 =RepeatVector(30)(L2)
    L4 =LSTM(128,return_sequences=True)(L3)
    L5 =Dropout(0.1)(L4)
    output = TimeDistributed(Dense(num_features))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_model_beta(inputs, hyperparameter=None):
    #inputs = Input(shape=(X.shape[1], X.shape[2]))
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(timesteps,num_features)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(timesteps)(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(num_features))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model