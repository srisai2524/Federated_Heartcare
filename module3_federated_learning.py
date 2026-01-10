import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

clients = {
    "Typical": pd.read_csv("typical.csv"),
    "Athletic": pd.read_csv("athletic.csv"),
    "Diver": pd.read_csv("diver.csv")
}

global_weights = None

def train_local_model(client_data):
    X = client_data.drop(["target", "user_type"], axis=1)
    y = client_data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model.coef_, model.intercept_

# Federated Averaging
for round in range(3):
    print(f"\n🔄 Federated Round {round + 1}")
    weights, biases = [], []

    for name, data in clients.items():
        w, b = train_local_model(data)
        weights.append(w)
        biases.append(b)
        print(f"Client {name} trained locally")

    global_weights = np.mean(weights, axis=0)
    global_bias = np.mean(biases, axis=0)

print("✔ Federated Learning Completed")
