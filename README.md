# 🏢 Saher – Smart Building Health Monitor

Saher is an AI-powered system for monitoring building structural health using sensor data.
The project applies Machine Learning and Deep Learning techniques to predict building conditions and identify risks.

## 📌 Project Idea

The system predicts building safety conditions based on real-time sensor data:

Accelerometer (X, Y, Z)

Strain (μɛ)

Temperature (°C)

(Optional) Seconds / Timestamp

The model classifies buildings into three categories:

✅ Safe (0)

⚠️ Warning (1)

🚨 Critical (2)

## 📂 Project Structure
```
saher-building-health-monitor/
│
├── saher_app.py # Streamlit application
├── requirements.txt # Project dependencies
│
├── models/ # Trained models
│ ├── saher_model.pkl # Scikit-learn model
│ └── saher_model.h5 # Keras model (optional)
│
├── scalers/
│ └── scaler.pkl # MinMaxScaler used during training
│
├── data/
│ └── building_health.csv # Cleaned dataset
│
├── notebooks/
│ └── Saher_Training.ipynb # Jupyter notebook for training & analysis
│
└── README.md # Documentation
```

## ⚙️ Setup & Run Locally

Clone the repository:

git clone https://github.com/MahmoudAttia111/saher-building-health-monitor.git
cd saher-building-health-monitor


(Optional) Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Linux/Mac


### Install dependencies:

pip install -r requirements.txt


### Run the Streamlit app:

streamlit run saher_app.py

## 📊 Model Training

The training pipeline is provided in:
notebooks/Saher_Training.ipynb

Steps include:

Data exploration & preprocessing

Handling missing values and outliers

Feature scaling using MinMaxScaler

Handling class imbalance using:

class_weight (for Neural Networks)

SMOTE / ADASYN (for classical ML models)

Training multiple ML models (Decision Trees, Random Forest, SVM, Naive Bayes, etc.)

Training a Neural Network with Keras

Saving the best model and scaler:

import joblib, pickle

### Save model
joblib.dump(model, "models/saher_model.pkl")

### Save scaler
with open("scalers/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

## 🌐 Deployment
You can try the live app here:
🔗  https://saher-building-health-monitor-6fx8hpvvkpuzni5sihzehu.streamlit.app/

The app can be deployed on:

Streamlit Cloud

## 📦 requirements.txt
streamlit
scikit-learn
numpy
pandas
joblib
matplotlib
seaborn
tensorflow    

 

### Streamlit app UI

Confusion Matrix from Jupyter notebook

 
