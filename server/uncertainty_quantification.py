import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import numpy as np

def build_stacked_lstm_mc_dropout(input_shape, dropout_rate=0.3):
    """
    Builds a Stacked LSTM model with Monte Carlo Dropout enabled at inference time.
    """
    inputs = Input(shape=input_shape)

    # First LSTM layer with Return Sequences and MC Dropout
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x, training=True)  # training=True enables MC Dropout during inference

    # Second LSTM layer
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x, training=True)

    # Dense layers for regression
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x, training=True)

    # Output layer (1 unit for RUL regression)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def predict_with_uncertainty(f_model, X, n_iter=50):
    """
    Performs Monte Carlo Dropout inference to estimate Mean RUL and Uncertainty (Std Dev).
    """
    # X shape: (1, time_steps, features)
    # Run the model n_iter times (due to Dropout, each run gives different result)
    # Using batch prediction for speed if possible, or loop
    # Replicate X n_iter times to run in one batch for speed
    X_repeated = np.repeat(X, n_iter, axis=0)
    
    # Predict
    preds = f_model.predict(X_repeated, verbose=0)
    
    # Calculate stats
    mean_pred = float(preds.mean())
    std_pred = float(preds.std())
    
    return mean_pred, std_pred
