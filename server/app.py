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
    if rul > 80: return "Healthy"
    elif rul > 30: return "Degrading"
    else: return "Critical"

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
        np.random.seed(unit_nr)
        time_steps = np.arange(100)
        
        # Define ranges for "Normal" vs "High" for key sensors
        # Sensor: (Normal_Min, Normal_Max, Critical_Max)
        S_SPECS = {
            'sensor_4':  (1100, 1300, 1500), # Turbine Temp
            'sensor_11': (550, 600, 650),    # Static Press
            'sensor_14': (140, 150, 165),    # Core Speed
            'sensor_7':  (550, 600, 640),    # Compressor Press
            'sensor_2':  (640, 645, 655),    # Inlet Temp
        }
        
        unit_data = pd.DataFrame({'time_cycles': time_steps, 'unit_nr': unit_nr})
        
        # Fill baseline trends
        for s, (low, mid, high) in S_SPECS.items():
            unit_data[s] = np.linspace(low, mid, 100) + np.random.normal(0, (mid-low)*0.05, 100)
        
        # 2. Apply ALL User Inputs to the LAST ROW
        last_idx = unit_data.index[-1]
        for col, val in user_inputs.items():
            # If the column exists in our specs or DataFrame, update it
            if col in S_SPECS:
                unit_data.at[last_idx, col] = val
        
        # 3. Calculate Dynamic RUL based on Stress Heuristic
        # We calculate "Stress" score from 0 (Healthy) to 1.0 (Critical Failure)
        stress_scores = []
        for s, (low, mid, high) in S_SPECS.items():
            current_val = unit_data.at[last_idx, s]
            # Normalize: mid is "normal/degrading boundary", high is "critical"
            # score = 0 at mid, score = 1 at high
            s_stress = max(0, (current_val - mid) / (high - mid)) if high > mid else 0
            stress_scores.append(s_stress)
        
        total_stress = np.mean(stress_scores) if stress_scores else 0
        total_stress = min(1.0, total_stress) # Cap at 1.0

        # Map Stress to RUL
        # total_stress=0   -> RUL=150 (Healthy)
        # total_stress=0.5 -> RUL=50  (Degrading)
        # total_stress=0.8 -> RUL=15  (Critical)
        if total_stress < 0.2:
            sim_rul = 150 - (total_stress * 150) # Mostly Healthy
        elif total_stress < 0.7:
            sim_rul = 80 - ((total_stress - 0.2) * 100) # Degrading
        else:
            sim_rul = 25 - ((total_stress - 0.7) * 50) # Critical

        sim_rul = max(0, int(sim_rul + np.random.normal(0, 3)))
        mean_rul = float(sim_rul)
        std_rul = float(1.2 + (150 - mean_rul) * 0.01) # Reduced uncertainty for high confidence scores

        # 4. Determine Top Contributor for Demo
        # Find sensor with highest normalized stress
        stress_dict = {s: max(0, (unit_data.at[last_idx, s] - mid) / (high - mid)) 
                       for s, (low, mid, high) in S_SPECS.items()}
        top_s = max(stress_dict, key=stress_dict.get)
        top_contributor_str = f"{top_s} ({stress_dict[top_s]:.4f})"
        
        # Add synthetic predicted RUL column for plot
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
    
    # Generate tailored recommendation
    if mean_rul > 80:
        rec_text = "System Healthy. Continue standard monitoring schedule."
        alert_text = "None. All sensors within nominal range."
        status_icon = "🟢"
    elif mean_rul > 30:
        rec_text = "Schedule maintenance within next 20 cycles."
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
            {'sensor': 'sensor_4', 'label': 'Turbine Temp', 'value': f"{unit_data.iloc[-1].get('sensor_4', 0):.2f}", 'normal': '1100 - 1450'},
            {'sensor': 'sensor_11', 'label': 'High Pressure', 'value': f"{unit_data.iloc[-1].get('sensor_11', 0):.2f}", 'normal': '550 - 650'},
             {'sensor': 'sensor_14', 'label': 'Coolant Flow', 'value': f"{unit_data.iloc[-1].get('sensor_14', 0):.2f}", 'normal': '120 - 160'}
        ],
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Debug: Print data to console to verify insights are generated
    print("DEBUG: Sending Prediction Data:", prediction_data)

    return render_template('result.html', data=prediction_data)

if __name__ == "__main__":
    app.run(debug=True)
