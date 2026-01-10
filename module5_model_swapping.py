import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
datasets = {
    "Typical": "typical.csv",
    "Athletic": "athletic.csv",
    "Diver": "diver.csv"
}

models = {}

# Train and save models
for user_type, file in datasets.items():
    data = pd.read_csv(file)
    X = data.drop(["target", "user_type"], axis=1)
    y = data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, f"model_{user_type.lower()}.pkl")
    models[user_type] = model

    print(f"✔ {user_type} model trained and saved")

print("\n🔄 Simulating model swapping...")

# Simulate drift-based model swapping
current_model = "Typical"
print(f"Current active model: {current_model}")

# Assume drift detected → user becomes Athletic
new_state = "Athletic"
current_model = new_state
print(f"🔁 Model swapped to: {current_model}")
