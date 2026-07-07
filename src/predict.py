import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "penetration_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

print("=" * 50)
print("Tumor Nanoparticle Penetration Predictor")
print("=" * 50)

while True:

    try:
        particle_size = float(input("\nEnter Particle Size (nm): "))
        time = float(input("Enter Time: "))

        sample = pd.DataFrame({
            "ParticleSize": [particle_size],
            "Time": [time]
        })

        prediction = model.predict(sample)[0]

        print("\nPredicted Penetration Depth = {:.4f}".format(prediction))

    except Exception as e:
        print("Error:", e)

    choice = input("\nDo you want another prediction? (y/n): ")

    if choice.lower() != "y":
        break

print("\nThank you.")