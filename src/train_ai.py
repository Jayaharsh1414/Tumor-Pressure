import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = pd.read_csv(RESULTS_DIR / "AI_dataset.csv")

print("\nDataset Preview:\n")
print(df.head())

# ----------------------------------------------------
# Features and Target
# ----------------------------------------------------

X = df[["ParticleSize", "Time"]]
y = df["PenetrationDepth"]

# ----------------------------------------------------
# Train-Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ----------------------------------------------------
# Models
# ----------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
}

best_model = None
best_score = -999

print("\n==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print("------------------------------")
    print(f"MAE  = {mae:.5f}")
    print(f"RMSE = {rmse:.5f}")
    print(f"R2   = {r2:.5f}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_predictions = pred

print("\n==============================")
print("Best Model Saved")

joblib.dump(
    best_model,
    MODELS_DIR / "penetration_model.pkl"
)

# ----------------------------------------------------
# Actual vs Predicted
# ----------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    best_predictions,
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.savefig(
    RESULTS_DIR / "AI_actual_vs_predicted.png"
)

plt.close()

print("\nFinished Successfully!")