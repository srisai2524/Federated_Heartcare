import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load merged dataset
data = pd.concat([
    pd.read_csv("typical.csv"),
    pd.read_csv("athletic.csv"),
    pd.read_csv("diver.csv")
])

X = data.drop(["target", "user_type"], axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Centralized Model Accuracy:", accuracy)
