import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from server.feature_engineering import create_time_series_features, create_lstm_sequences
from server.uncertainty_quantification import build_stacked_lstm_mc_dropout

def main():
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train_FD002_with_RUL.csv")
    test_df = pd.read_csv("data/processed/test_FD002_clean.csv")

    # 2. Feature Engineering
    print("Generating features...")
    # Define features to use
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    op_cols = [col for col in train_df.columns if col.startswith('op_setting')]
    
    # Apply rolling and lags
    train_df = create_time_series_features(train_df, sensor_cols)
    test_df = create_time_series_features(test_df, sensor_cols)

    # List of all feature columns after engineering
    feature_cols = [c for c in train_df.columns if c not in ['unit_nr', 'time_cycles', 'RUL', 'engine_id', 'cycle']]
    
    # 3. Scaling
    print("Scaling data...")
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save Scaler and feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_elite.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    # 4. Sequence Generation
    print("Generating sequences...")
    time_steps = 20
    
    # For training, we need to iterate over units to create sequences without bleeding
    X_train_seq = []
    y_train_seq = []
    
    for unit in train_df['unit_nr'].unique():
        unit_data = train_df[train_df['unit_nr'] == unit]
        X_unit, y_unit = create_lstm_sequences(unit_data[feature_cols], unit_data['RUL'], time_steps)
        if len(X_unit) > 0:
            X_train_seq.append(X_unit)
            y_train_seq.append(y_unit)
            
    X_train = np.concatenate(X_train_seq)
    y_train = np.concatenate(y_train_seq)
    
    print(f"Training Data Implemented: {X_train.shape}")

    # 5. Model Training
    print("Training Model...")
    model = build_stacked_lstm_mc_dropout(input_shape=(time_steps, len(feature_cols)))
    
    # Train
    model.fit(
        X_train, y_train,
        epochs=10, # Kept low for demo speed, increase for production
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # 6. Save Model
    print("Saving Model...")
    model.save("models/lstm_elite.h5")
    print("Done!")

    # 7. Generate Background Data for SHAP (Save a small sample)
    background_data = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    np.save("models/background_data.npy", background_data)

if __name__ == "__main__":
    main()
