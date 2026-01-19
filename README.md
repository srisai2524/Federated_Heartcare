📌 Project Overview

Federated HeartCare is a privacy-preserving and adaptive heart disease prediction system that combines Federated Learning and Concept Drift Detection to provide reliable and personalized cardiac risk assessment.
Instead of collecting sensitive patient data on a central server, the system trains models locally on distributed clients and shares only model parameters, ensuring data confidentiality.

The system dynamically adapts to lifestyle changes (typical, athletic, diver) using a drift-aware model swapping mechanism, maintaining high prediction accuracy over time.

🎯 Objectives

Preserve patient data privacy using federated learning

Detect physiological changes using concept drift detection

Adapt models dynamically for different user profiles

Provide real-time heart disease prediction through a UI

🧠 System Architecture
Wearable / Client Devices → Local Training → Federated Aggregation
        ↓
Continuous Monitoring → Concept Drift Detection
        ↓
Model Swapping → Accurate Prediction → Streamlit UI

🛠 Technologies Used
Category	Tools
Programming Language	Python
Machine Learning	Scikit-learn
Federated Learning	Custom FedAvg Simulation
Drift Detection	River (ADWIN)
UI	Streamlit
Visualization	Matplotlib
Dataset	UCI Heart Disease Dataset
📂 Project Structure
Federated_HeartCare/
│
├── heart.csv
├── module1_data_preparation.py
├── module2_centralized_model.py
├── module3_federated_learning.py
├── module4_drift_detection.py
├── module5_model_swapping.py
├── module6_evaluation.py
├── app.py
├── model_typical.pkl
├── model_athletic.pkl
├── model_diver.pkl
├── scaler_typical.pkl
├── scaler_athletic.pkl
├── scaler_diver.pkl
└── README.md

▶️ How to Run the Project
1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2. Install Dependencies
pip install pandas numpy scikit-learn matplotlib river streamlit joblib

3. Run Modules
python module1_data_preparation.py
python module2_centralized_model.py
python module3_federated_learning.py
python module4_drift_detection.py
python module5_model_swapping.py
python module6_evaluation.py

4. Launch UI
streamlit run app.py

📈 Results

Centralized Model Accuracy: 86.8%

Federated Learning preserves privacy

Drift-aware model swapping improves adaptability

UI demonstrates real-time prediction

🔐 Privacy & Security

No raw patient data is shared

Only model parameters are transmitted

Predictions occur locally

🧪 Sample Prediction
Input	Output
Healthy values	✅ Low Risk
Risky values	⚠ High Risk


