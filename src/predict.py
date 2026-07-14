"""
Multi-output Drug Delivery Prediction
Predicts:
- Penetration Depth
- Maximum Concentration
- Drug Coverage
- Delivery Time
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Load trained model and scaler
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please run src/train_ai.py first to train the model.")
    exit(1)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

feature_names = ['ParticleSize', 'Pressure', 'HydraulicConductivity', 'UptakeRate', 'Diffusion']
output_names = ['PenetrationDepth', 'MaxConcentration', 'DrugCoverage', 'DeliveryTime']

print("=" * 70)
print("TUMOR NANOPARTICLE DRUG DELIVERY PREDICTOR - PHASE 2")
print("=" * 70)
print("\nThis tool predicts multiple drug delivery parameters:")
print("  1. Penetration Depth (mm)")
print("  2. Maximum Concentration")
print("  3. Drug Coverage (%)")
print("  4. Delivery Time (seconds)")

while True:
    try:
        print("\n" + "-" * 70)
        print("Enter Nanoparticle Parameters:")
        print("-" * 70)
        
        particle_size = float(input("  Particle Size (nm) [20-200]: "))
        pressure = float(input("  Vessel Pressure (mmHg) [15-25]: "))
        hydraulic_cond = float(input("  Hydraulic Conductivity (e-6) [0.8-1.2]: ")) * 1e-6
        uptake_rate = float(input("  Cellular Uptake Rate [0.02-0.10]: "))
        
        # Compute diffusion coefficient using Stokes-Einstein equation,
        # converted from m^2/s to mm^2/s to match training units
        # (generate_dataset.py) - without this the model sees D 1e6x
        # outside its training distribution.
        kB = 1.380649e-23
        T = 310
        mu = 1e-3
        r = particle_size * 1e-9
        D = (kB * T / (3 * np.pi * mu * r)) * 1e6
        
        # Create input dataframe
        sample = pd.DataFrame({
            "ParticleSize": [particle_size],
            "Pressure": [pressure],
            "HydraulicConductivity": [hydraulic_cond],
            "UptakeRate": [uptake_rate],
            "Diffusion": [D]
        })
        
        # Scale features
        sample_scaled = scaler.transform(sample)
        
        # Make prediction
        predictions = model.predict(sample_scaled)[0]
        
        # Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        
        print(f"\nInput Parameters:")
        print(f"  Particle Size: {particle_size:.1f} nm")
        print(f"  Vessel Pressure: {pressure:.1f} mmHg")
        print(f"  Hydraulic Conductivity: {hydraulic_cond:.2e} m²/(Pa·s)")
        print(f"  Cellular Uptake Rate: {uptake_rate:.3f} s⁻¹")
        print(f"  Diffusion Coefficient: {D:.2e} m²/s")
        
        print(f"\nPredicted Outcomes:")
        print(f"  Penetration Depth: {predictions[0]:.4f} mm")
        print(f"  Maximum Concentration: {predictions[1]:.4f}")
        print(f"  Drug Coverage: {predictions[2]:.2f}%")
        print(f"  Delivery Time: {predictions[3]:.2f} seconds")
        
        print("\n" + "=" * 70)
        
    except ValueError as e:
        print(f"Error: Invalid input. {e}")
        continue
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    choice = input("\nMake another prediction? (y/n): ")
    if choice.lower() != "y":
        break

print("\nThank you for using the Drug Delivery Predictor!")
