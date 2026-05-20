import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load sensor data
df = pd.read_csv("../sensor_simulation/sensor_data.csv")

# Create AI labels
# 1 = Irrigation Needed
# 0 = No Irrigation Needed

df["irrigation_needed"] = df["soil_moisture"].apply(
    lambda x: 1 if x < 40 else 0
)

# Features
X = df[[
    "soil_moisture",
    "temperature",
    "humidity",
    "soil_pH",
    "light_intensity"
]]

# Target
y = df["irrigation_needed"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "irrigation_model.pkl")

print("\nAI Model Trained Successfully!")