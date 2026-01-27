from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json

# Try importing standard optional libs
try:
    import plotly
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("NOTE: Plotly not installed. Charts will be disabled (install 'plotly' for charts).")

# Try importing elite components, handle missing libs for dev safety
try:
    import tensorflow as tf
    from uncertainty_quantification import predict_with_uncertainty, build_stacked_lstm_mc_dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("INFO: TensorFlow not found. Starting in ELITE DEMO MODE (Synthetic Data Enabled).")

# Import visualization independently of TF
plot_predicted_rul_and_sensors = None
try:
    from plotly_visualization import plot_predicted_rul_and_sensors
except ImportError as e:
    print(f"Warning: Could not import plotly_visualization. Error: {e}")

app = Flask(__name__)

# Paths
MODEL_PATH = "models/lstm_elite.h5"
SCALER_PATH = "models/scaler_elite.pkl"
FEATURE_COLS_PATH = "models/feature_cols.pkl"
DATA_PATH = "../data/processed/test_FD002_clean.csv"

# Global vars
model = None
scaler = None
feature_cols = None
test_df = None

# Default Sensor Specs for Demo and Monitoring
S_SPECS = {
    'sensor_2':  (640, 642, 648),    # Inlet Temp
    'sensor_3':  (1580, 1590, 1610), # Outlet Temp
    'sensor_4':  (1400, 1410, 1430), # Turbine Temp
    'sensor_7':  (550, 553, 558),    # Outlet Press
    'sensor_8':  (2388, 2388, 2389), # Fan Speed
    'sensor_9':  (9040, 9050, 9100), # Core Speed
    'sensor_11': (47, 47.5, 48.5),   # Static Press
    'sensor_12': (521, 522, 525),    # Fuel Flow
    'sensor_13': (2388, 2388, 2389), # Corrected Fan
    'sensor_14': (8120, 8135, 8160), # Corrected Core
    'sensor_15': (8.4, 8.45, 8.6),   # Bypass Ratio
    'sensor_17': (392, 393, 396),    # Bleed Air
    'sensor_20': (38.8, 39.0, 39.4), # HPT Cooling
    'sensor_21': (23.3, 23.4, 23.6), # LPT Cooling
}

def load_resources():
    global model, scaler, feature_cols, test_df
    
    # Load Data (needed for history lookups)
    if os.path.exists(DATA_PATH):
        test_df = pd.read_csv(DATA_PATH)
        print(f"Loaded Test Data: {test_df.shape}")
    else:
        print("Warning: Test data not found.")

    if not TF_AVAILABLE:
        return

    # Load Scaler & Features
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    else:
        print("Warning: Scaler/Feature cols not found.")
        return

    # Load Model
    if os.path.exists(MODEL_PATH):
        # We need to compile False or provide custom_objects for custom loss/metric if needed
        # Since we just used 'mse', it's standard.
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Elite LSTM Model Loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: Elite Model not found. Run train_elite.py first.")

# Initialize on startup
load_resources()

def get_engine_status(rul):
    """
    Determines engine health status based on RUL value.
    Thresholds aligned with bidirectional stress metric RUL ranges:
    - Healthy: RUL > 100 (stress < 0 to ~0.3)
    - Degrading: 40 < RUL ≤ 100 (stress ~0.3 to ~0.9)
    - Critical: RUL ≤ 40 (stress > 0.9)
    """
    if rul > 100: 
        return "Healthy"
    elif rul > 40: 
        return "Degrading"
    else: 
        return "Critical"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Parse Input
    unit_nr = 1 # Default
    if request.form.get('unit_nr'):
        unit_nr = int(request.form.get('unit_nr'))

    def extract_form_data():
        """Extracts sensor and op_setting values from the request form."""
        data = {}
        # Op Settings
        for i in range(1, 4):
            val = request.form.get(f'op_setting_{i}')
            if val: data[f'op_setting_{i}'] = float(val)
        
        # Sensors
        for i in range(1, 27): # Cover all potential sensors
            val = request.form.get(f'sensor_{i}')
            if val: data[f'sensor_{i}'] = float(val)
        return data

    user_inputs = extract_form_data()

    # DEMO MODE: If TF is not available or model incomplete, run simulation
    if not TF_AVAILABLE or model is None:
        print(f"Running in DEMO MODE for Unit {unit_nr}")
        
        # 1. Generate Base Simulation History (99 steps)
        # Use a combination of unit_nr and user input hash for seeds to ensure variety
        seed_basis = unit_nr
        if user_inputs:
            # Simple hash of values to vary the "base" if they change inputs significantly
            seed_basis += int(sum(user_inputs.values()) % 1000)
            
        np.random.seed(seed_basis)
        time_steps = np.arange(100)
        
        unit_data = pd.DataFrame({'time_cycles': time_steps, 'unit_nr': unit_nr})
        
        # Fill baseline trends
        for s, (low, mid, high) in S_SPECS.items():
            # Trend slightly upwards/downwards depending on sensor
            trend = np.linspace(low, mid, 100)
            unit_data[s] = trend + np.random.normal(0, (mid-low)*0.1, 100)
        
        # 2. Apply ALL User Inputs to the LAST ROW
        last_idx = unit_data.index[-1]
        for col, val in user_inputs.items():
            if col in unit_data.columns:
                unit_data.at[last_idx, col] = val
        
        # 3. Calculate Dynamic RUL based on Bidirectional Stress Heuristic
        # Stress score: negative = healthier than normal, 0 = normal, positive = degrading/critical
        stress_scores = []
        stress_dict = {}
        for s, (low, mid, high) in S_SPECS.items():
            current_val = unit_data.at[last_idx, s]
            
            if current_val < mid:
                # Healthy zone: negative stress (better than normal)
                # At low: stress = -1.0, at mid: stress = 0
                s_stress = (current_val - mid) / (mid - low) if mid > low else 0
            else:
                # Degrading/Critical zone: positive stress (worse than normal)
                # At mid: stress = 0, at high: stress = 1.0
                s_stress = (current_val - mid) / (high - mid) if high > mid else 0
            
            stress_scores.append(s_stress)
            stress_dict[s] = s_stress
        
        # Heuristic: Take the maximum absolute stress as a primary driver, but also consider the mean
        # For safety, we prioritize the worst sensor (highest positive stress)
        # But also consider average to get overall health picture
        max_stress = max(stress_scores) if stress_scores else 0
        min_stress = min(stress_scores) if stress_scores else 0
        avg_stress = np.mean(stress_scores) if stress_scores else 0
        
        # Combined stress score (max stress has higher weight for safety)
        # If max_stress is positive (degrading), it dominates
        # If all negative (healthy), use average
        if max_stress > 0.1:
            total_stress = (max_stress * 0.7) + (avg_stress * 0.3)
        else:
            total_stress = avg_stress
        
        total_stress = max(-1.5, min(2.0, total_stress)) # Range: -1.5 (very healthy) to 2.0 (super critical)

        # Map Stress to RUL (Bidirectional)
        # total_stress=-1.5 -> RUL=300+ (Exceptionally Healthy)
        # total_stress=-1.0 -> RUL=250  (Very Healthy)
        # total_stress=-0.5 -> RUL=200  (Healthy)
        # total_stress=0    -> RUL=150  (Normal)
        # total_stress=0.5  -> RUL=80   (Early Degrading)
        # total_stress=1.0  -> RUL=35   (Degrading/Warning)
        # total_stress=1.5  -> RUL=10   (Critical)
        # total_stress=2.0  -> RUL=1    (Imminent Failure)
        
        if total_stress < -1.0:
            # Exceptionally healthy: -1.5 to -1.0 -> 300 to 250
            sim_rul = 250 + ((total_stress + 1.0) * -100)
        elif total_stress < -0.5:
            # Very healthy: -1.0 to -0.5 -> 250 to 200
            sim_rul = 200 + ((total_stress + 0.5) * -100)
        elif total_stress < 0:
            # Healthy: -0.5 to 0 -> 200 to 150
            sim_rul = 150 + ((total_stress) * -100)
        elif total_stress < 0.5:
            # Normal to early degrading: 0 to 0.5 -> 150 to 80
            sim_rul = 150 - (total_stress * 140)
        elif total_stress < 1.0:
            # Degrading: 0.5 to 1.0 -> 80 to 35
            sim_rul = 80 - ((total_stress - 0.5) * 90)
        elif total_stress < 1.5:
            # Critical: 1.0 to 1.5 -> 35 to 10
            sim_rul = 35 - ((total_stress - 1.0) * 50)
        else:
            # Imminent failure: 1.5 to 2.0 -> 10 to 1
            sim_rul = 10 - ((total_stress - 1.5) * 18)

        sim_rul = max(1, int(sim_rul + np.random.normal(0, 2)))
        mean_rul = float(sim_rul)
        std_rul = float(1.0 + total_stress * 5.0) # Uncertainty increases with stress

        # 4. Determine Top Contributor for Demo
        top_s = max(stress_dict, key=stress_dict.get)
        top_contributor_str = f"{top_s} (Stress: {stress_dict[top_s]:.2f})"
        
        # Add synthetic predicted RUL column for plot
        # Create a trend that leads to current predicted RUL
        unit_data['predicted_rul'] = np.linspace(mean_rul + 100, mean_rul, 100)

    # REAL MODE: If system is ready
    else:
        # Fetch history from test_df
        if test_df is None: return "Test data unavailable.", 500
        
        # Get data for this unit
        unit_data = test_df[test_df['unit_nr'] == unit_nr].copy()
        
        if len(unit_data) < 20:
             return f"Not enough history for Unit {unit_nr} (Need 20 cycles).", 400
        
        # Apply User Inputs to the LAST ROW for Prediction
        last_idx = unit_data.index[-1]
        for col, val in user_inputs.items():
            if col in unit_data.columns:
                unit_data.at[last_idx, col] = val
             
        # Initialize predicted_rul column for visualization (mostly empty since we only predict last step)
        unit_data['predicted_rul'] = np.nan
        
        # Feature Engineering 
        sensor_cols = [c for c in unit_data.columns if c.startswith('sensor_')]
        unit_data_eng = create_time_series_features(unit_data.copy(), sensor_cols)
        
        # Scale
        try:
            unit_data_eng[feature_cols] = scaler.transform(unit_data_eng[feature_cols])
        except Exception as e:
            return f"Scaling Error (Feature mismatch?): {e}", 500
        
        # Get last sequence
        X_seq, _ = create_lstm_sequences(unit_data_eng[feature_cols], time_steps=20)
        if len(X_seq) == 0: return "Insufficient data sequence.", 400
        X_input = X_seq[-1:] 

        # Prediction & Uncertainty
        mean_rul, std_rul = predict_with_uncertainty(model, X_input)
        
        # Update visualization dataframe with the prediction
        unit_data.iloc[-1, unit_data.columns.get_loc('predicted_rul')] = mean_rul
        
        # Explainability (SHAP)
        try:
            if os.path.exists("models/background_data.npy"):
                background = np.load("models/background_data.npy")
            else:
                background = np.zeros((1, 20, len(feature_cols))) 
            
            shap_values = explain_model_prediction(model, X_input, background)
            top_features = get_top_contributors(shap_values, feature_cols)
            top_contributor_str = f"{top_features[0][0]} ({top_features[0][1]:.4f})"
        except Exception as e:
            print(f"SHAP Error: {e}")
            top_contributor_str = "N/A"

    # Visualization (Works for both Demo and Real)
    graphJSON = None
    plot_error = None
    if PLOTLY_AVAILABLE and plot_predicted_rul_and_sensors:
        try:
            # Using the unit_data we have (real or mock)
            fig = plot_predicted_rul_and_sensors(unit_data, sensor_cols=['sensor_11', 'sensor_4'], unit_nr=unit_nr)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            plot_error = str(e)
            print(f"Plotting Error: {e}")

    # 5. response & Insights
    
    # Generate tailored recommendation (aligned with new status thresholds)
    if mean_rul > 100:
        rec_text = "System Healthy. Continue standard monitoring schedule."
        alert_text = "None. All sensors within nominal range."
        status_icon = "🟢"
    elif mean_rul > 40:
        rec_text = "Schedule maintenance within next 20-40 cycles. Monitor sensor trends closely."
        alert_text = f"Degradation detected in {top_contributor_str.split('(')[0]}."
        status_icon = "🟡"
    else:
        rec_text = "IMMEDIATE INSPECTION REQUIRED. Halt operation if possible."
        alert_text = f"CRITICAL VARIANCE: {top_contributor_str.split('(')[0]} exceeding safe thresholds."
        status_icon = "🔴"

    prediction_data = {
        "rul": int(mean_rul),
        "margin": int(std_rul * 1.96), # 95% CI roughly
        "status": get_engine_status(mean_rul),
        "status_icon": status_icon,
        "top_contributor": top_contributor_str,
        "confidence": round(max(0, 100 - (std_rul / 0.5)), 2) if mean_rul > 0 else 0, # Scaled for 'High' confidence
        "alert_trigger": alert_text,
        "recommendation": rec_text,
        "plot_json": graphJSON,
        "plot_error": plot_error,
        "demo_mode": not TF_AVAILABLE, # Flag to show in UI
        "no_plot": not PLOTLY_AVAILABLE, # Flag for UI to hide chart area
        
        # Data for PDF Audit
        "input_summary": [
            {'sensor': 'sensor_4', 'label': 'Turbine Temp', 'value': f"{unit_data.iloc[-1].get('sensor_4', 0):.2f}", 'normal': f"{S_SPECS['sensor_4'][0]} - {S_SPECS['sensor_4'][1]}"},
            {'sensor': 'sensor_11', 'label': 'Comp. Static Press', 'value': f"{unit_data.iloc[-1].get('sensor_11', 0):.2f}", 'normal': f"{S_SPECS['sensor_11'][0]} - {S_SPECS['sensor_11'][1]}"},
            {'sensor': 'sensor_14', 'label': 'Corrected Core Speed', 'value': f"{unit_data.iloc[-1].get('sensor_14', 0):.2f}", 'normal': f"{S_SPECS['sensor_14'][0]} - {S_SPECS['sensor_14'][1]}"}
        ],
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Debug: Print data to console to verify insights are generated
    print("DEBUG: Sending Prediction Data:", prediction_data)

    return render_template('result.html', data=prediction_data)

if __name__ == "__main__":
    app.run(debug=True)
