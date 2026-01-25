import pandas as pd
import numpy as np

def create_time_series_features(df, sensor_cols, window=20, lags=[1, 5]):
    """
    Generates Rolling Means, Rolling Std Devs, and Lag Features.
    Ensures data does not bleed between units by grouping by 'unit_nr'.
    """
    # Sort by unit and time to ensure correct rolling and lag calculations
    df = df.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)

    # Rolling features grouped by unit_nr
    for col in sensor_cols:
        # Rolling Mean
        df[f'{col}_rolling_mean'] = df.groupby('unit_nr')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Rolling Std Deviation (Uncertainty proxy)
        df[f'{col}_rolling_std'] = df.groupby('unit_nr')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )

    # Lag features grouped by unit_nr
    for lag in lags:
        for col in sensor_cols:
            df[f'{col}_lag_{lag}'] = df.groupby('unit_nr')[col].shift(lag)

    # Fill NaN values created by lags with 0 or forward fill (method='bfill' to avoid data leakage is risky, sticking to 0 for stability or drop)
    df = df.fillna(0)
    
    return df

def create_lstm_sequences(X, y=None, time_steps=20):
    """
    Generates 3D sequences (samples, time_steps, features) for LSTM.
    X: DataFrame or 2D array of features.
    y: Series or array of targets (optional).
    time_steps: The lookback window size.
    """
    Xs, ys = [], []
    if isinstance(X, pd.DataFrame):
        X = X.values
    if y is not None and isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values

    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        if y is not None:
            ys.append(y[i + time_steps - 1])
    
    if y is not None:
        return np.array(Xs), np.array(ys)
    else:
        return np.array(Xs)

