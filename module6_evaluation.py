import matplotlib.pyplot as plt

# Accuracy values (example results)
accuracy_before_drift = [0.82, 0.81, 0.80, 0.79]
accuracy_after_swap = [0.82, 0.88, 0.90, 0.91]

plt.plot(accuracy_before_drift, marker='o', label='Before Drift Handling')
plt.plot(accuracy_after_swap, marker='o', label='After Model Swap')

plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("Federated HeartCare Performance Improvement")
plt.legend()
plt.grid(True)
plt.show()
