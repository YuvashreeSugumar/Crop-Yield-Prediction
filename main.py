# crop_yield_prediction.py
# Author: Yuvashree Sugumar

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/large_sample_dataset.csv")

# Preprocessing
df = pd.get_dummies(df, columns=['soil_type', 'crop_type'])

# Features and target
X = df.drop("yield", axis=1)
y = df["yield"]

# Train model on full dataset
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Evaluation
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.2f}")

# Plot results
plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred, color='green')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Full Data: Actual vs Predicted Crop Yield")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/full_data_prediction_plot.png")
plt.show()
