from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

def autoencoder_model_alpha(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]

    model.add(LSTM(128,input_shape=(timesteps,num_features)))
    model.add(Dropout(0.2))
    model.add(RepeatVector(30))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_features)))
    #model = Model(inputs=inputs, outputs=output)
    return model
def autoencoder_model_beta(model, inputs, hyperparameters=None):
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(LSTM(16, activation='relu', return_sequences=True,kernel_regularizer=regularizers.l2(0.00))(timesteps,num_features))
    model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model

def autoencoder_model_gamma(model, inputs, hyperparameters=None):
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    encoder_input = Input(shape=(timesteps, num_features), name='encoder_input')
    encoder_L1 = LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='encoder_L1')(
        encoder_input)
    encoder_L2 = LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_L2')(encoder_L1)
    encoder_L3 = LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_L3')(encoder_L2)

    # bridge: input (None, 16), output (None, 4, 16)
    # not part of encoder or decoder; will be necessary to connect the two
    encoder_decoder_bridge = RepeatVector(timesteps, name='encoder_decoder_bridge')(encoder_L3)

    # decoder stuff: input (None, 4, 16), output (None, 4, feats)
    decoder_input = Input(shape=(timesteps, 16), name='decoder_input')
    decoder_L1 = LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L1')(
        decoder_input)
    decoder_L2 = LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L2')(decoder_L1)
    decoder_L3 = LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_L3')(decoder_L2)
    decoder_output = TimeDistributed(Dense(num_features))(decoder_L3)

    # encoder
    encoder = Model(inputs=[encoder_input], outputs=[encoder_L3])  # output must be passed through repeat vector in order to fed to decoder

    # decoder
    decoder = Model(inputs=[decoder_input], outputs=[decoder_output])

def autoencoder_model_delta(model, inputs, hyperparameters=None):
    # encoder stuff: input (None, 4, feats), output (None, 16)
    timesteps = inputs.shape[1]
    num_features = inputs.shape[2]
    model.add(Input(shape=(timesteps, num_features), name='encoder_input'))
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