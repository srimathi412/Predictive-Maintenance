# 🚀 Predictive Maintenance System

## 📌 Project Overview
Predictive Maintenance aims to predict equipment failure before it happens by analyzing sensor data.
In this project, we estimate the Remaining Useful Life (RUL) of jet engines using machine learning models trained on the NASA Turbofan Engine Dataset.

This helps industries reduce maintenance cost, avoid unexpected breakdowns, and optimize maintenance schedules.

## 📊 Dataset Used
NASA Turbofan Jet Engine Dataset (FD001)

Dataset Files:
- train_FD001.txt – Training data
- test_FD001.txt – Test data
- RUL_FD001.txt – Remaining Useful Life labels

## 🧠 Problem Statement
Given historical sensor data of an engine, predict how many cycles remain before the engine fails.

This is a regression problem where the target variable is Remaining Useful Life (RUL).

## ⚙️ Project Workflow
1. Load and explore sensor data
2. Generate RUL labels for training data
3. Data preprocessing and feature scaling
4. Train machine learning regression model
5. Evaluate model performance
6. Predict RUL for a selected engine

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📂 Project Structure
Predictive_Maintenance/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── notebooks/
│   ├── 01_data_and_RUL.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── server/
│   └── app.py
│
├── client/
│   └── index.html
│
├── model/
│   └── (trained model files generated after training)
│
└── README.md

## ▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/srimathi412/Predictive_Maintenance.git

2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

3. Run notebooks in order
01_data_and_RUL.ipynb
02_preprocessing.ipynb
03_model_training.ipynb

## 📌 Note on Model Files
Trained model files (.pkl) are not included in the repository due to GitHub size limitations.
They will be generated when the training notebook is executed.

## 📈 Output
- Predicts Remaining Useful Life (RUL)
- Helps plan maintenance schedules
- Reduces unexpected failu

