from river.drift import ADWIN
import numpy as np

drift_detector = ADWIN()

# Simulated heart rate stream
heart_rate_stream = np.random.normal(70, 2, 100)

# Strong concept drift
heart_rate_stream[40:] += 30   # FORCE clear drift

drift_found = False

for i, rate in enumerate(heart_rate_stream):
    drift = drift_detector.update(rate)

    if drift:
        print(f"⚠ Drift detected at index {i}")
        drift_found = True
        break

if not drift_found:
    print("✔ Stream processed — no drift detected (normal behavior)")
