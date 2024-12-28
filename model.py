import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv("student_lifestyle_dataset.csv")

# Features and target
X = data[[
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]]
y = data["GPA"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Function to classify GPA
def classify_gpa(gpa):
    if gpa >= 3.5:
        return "High"
    elif gpa >= 2.5:
        return "Moderate"
    else:
        return "Low"

# Save the classify function
with open('classify_gpa.pkl', 'wb') as file:
    pickle.dump(classify_gpa, file)
