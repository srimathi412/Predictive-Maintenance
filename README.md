#  Predictive Maintenance System

## Project Overview
Predictive Maintenance aims to predict equipment failure before it happens by analyzing sensor data.
In this project, we estimate the Remaining Useful Life (RUL) of jet engines using machine learning models trained on the NASA Turbofan Engine Dataset.

This helps industries reduce maintenance cost, avoid unexpected breakdowns, and optimize maintenance schedules.

##  Dataset Used
NASA Turbofan Jet Engine Dataset (FD001)

Dataset Files:
- train_FD001.txt – Training data
- test_FD001.txt – Test data
- RUL_FD001.txt – Remaining Useful Life labels

## Problem Statement
Given historical sensor data of an engine, predict how many cycles remain before the engine fails.

This is a regression problem where the target variable is Remaining Useful Life (RUL).

##  Project Workflow
1. Load and explore sensor data
2. Generate RUL labels for training data
3. Data preprocessing and feature scaling
4. Train machine learning regression model
5. Evaluate model performance
6. Predict RUL for a selected engine

##  Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

##  Project Structure
```
Predictive_Maintenance/
│
├── README.md                          # Project documentation
├── train_elite.py                     # Main training script for LSTM model
│
├── data/                              # Dataset directory
│   ├── raw/                          # Original NASA C-MAPSS data
│   │   ├── train_FD002.txt
│   │   ├── test_FD002.txt
│   │   └── RUL_FD002.txt
│   │
│   ├── processed/                    # Cleaned and processed data
│   │   ├── train_FD002_with_RUL.csv
│   │   └── test_FD002_clean.csv
│   │
│   └── final/                        # Numpy arrays for training
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── X_val.npy
│       ├── y_train.npy
│       └── y_val.npy
│
├── models/                            # Saved models
│   ├── rf_fd002_rul_model.pkl        # Random Forest model (old)
│   └── scaler_fd002.pkl              # Feature scaler (old)
│   # Note: LSTM model files will be created here after training:
│   # - lstm_elite.h5
│   # - scaler_elite.pkl
│   # - feature_cols.pkl
│   # - background_data.npy
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_and_RUL.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── results/                           # Analysis results
│   └── fd002_engine_status.csv
│
└── server/                            # Flask web application
    ├── app.py                        # Main Flask application
    ├── feature_engineering.py        # Time-series feature creation
    ├── uncertainty_quantification.py # MC Dropout LSTM model
    ├── shap_explainability.py        # SHAP explanations
    ├── plotly_visualization.py       # Interactive charts
    ├── model_lstm.py                 # LSTM model utilities
    │
    ├── templates/                    # HTML templates
    │   ├── index.html               # Input form (warm UI design)
    │   └── result.html              # Results dashboard (warm UI design)
    │
    └── __pycache__/                  # Python cache (auto-generated)
        ├── feature_engineering.cpython-314.pyc
        ├── plotly_visualization.cpython-314.pyc
        └── shap_explainability.cpython-314.pyc
```

##  How to Run the Project
1. Clone the repository
git clone https://github.com/srimathi412/Predictive_Maintenance.git

2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

3. Run notebooks in order
01_data_and_RUL.ipynb
02_preprocessing.ipynb
03_model_training.ipynb

##  Note on Model Files
Trained model files (.pkl) are not included in the repository due to GitHub size limitations.
They will be generated when the training notebook is executed.

## Output
- Predicts Remaining Useful Life (RUL)
- Helps plan maintenance schedules
- Reduces unexpected failure






