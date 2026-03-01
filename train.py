import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\sharm\OneDrive\Documents\weatherAUS.csv")

# Drop missing target
df = df.dropna(subset=["RainTomorrow"])

# Convert target + RainToday
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

# Select features
features = [
    "Humidity3pm",
    "Pressure3pm",
    "Temp3pm",
    "WindSpeed3pm",
    "RainToday"
]

df_model = df[features + ["RainTomorrow"]]
df_model = df_model.fillna(df_model.median())

X = df_model[features]
y = df_model["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Balanced logistic regression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rain_model.pkl")

print("Model trained and saved successfully.")