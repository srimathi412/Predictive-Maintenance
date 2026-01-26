import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def build_stacked_lstm(input_shape, dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    # First LSTM layer with return_sequences=True to stack LSTM
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)

    # Second LSTM layer
    x = LSTM(64)(x)
    x = Dropout(dropout_rate)(x)

    
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

   
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model



