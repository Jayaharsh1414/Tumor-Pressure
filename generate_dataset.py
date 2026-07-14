"""
Phase 2: Generate Comprehensive AI Dataset
Generates 500-1000 simulations by varying:
- Particle size (20, 50, 100, 150, 200 nm)
- Vessel pressure (15, 20, 25 mmHg)
- Hydraulic conductivity (Low, Medium, High)
- Cellular uptake rate (0.02-0.10)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed

from parameters import *
from fluid_model import solve_pressure, compute_velocity
from transport import transport_step, penetration_depth

# Setup paths
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Precompute baseline pressure field once globally (grid size N=100)
print("Precomputing baseline pressure field...")
BASE_PRESSURE = solve_pressure(100)

# ============================================
# Parameter Ranges for Dataset Generation
# ============================================

# Expanded parameter ranges for 500-1000 simulations
PARTICLE_SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # nm (10 values)
VESSEL_PRESSURES = [15, 17.5, 20, 22.5, 25]  # mmHg (5 values)
HYDRAULIC_CONDUCTIVITY = {
    'Low': 0.8e-6,
    'Medium': 1.0e-6,
    'High': 1.2e-6
}
UPTAKE_RATES = np.linspace(0.02, 0.10, 7)  # 7 values: [0.02, 0.0367, 0.0533, 0.07, 0.0867, 0.1033, 0.12]

# ============================================
# Helper Functions
# ============================================

def compute_diffusion_coefficient(particle_size_nm, T=310, mu=1e-3, kB=1.380649e-23):
    """
    Stokes-Einstein equation for diffusion coefficient
    D = kB*T / (3*pi*mu*r), converted from m^2/s to mm^2/s to match the
    domain (defined in mm in parameters.py) - without this conversion D is
    9-10 orders of magnitude too small to have any visible effect on the
    simulated transport (see TASKS.md Task 2).
    """
    r = particle_size_nm * 1e-9  # Convert nm to m
    D_si = kB * T / (3 * np.pi * mu * r)  # m^2/s
    return D_si * 1e6  # mm^2/s

def compute_drug_coverage(C, threshold):
    """
    Compute the percentage of domain with concentration above threshold
    """
    total_cells = C.size
    covered_cells = np.sum(C > threshold)
    coverage = (covered_cells / total_cells) * 100
    return coverage

def compute_delivery_time(C, threshold, dt, target_penetration=3.0):
    """
    Estimate time to reach target penetration depth
    """
    # This is computed during simulation
    return None  # Will be updated during simulation

def run_simulation(particle_size, pressure_factor, K_value, uptake_rate, N=100, dt=dt):
    """
    Run a single simulation and return output metrics
    """
    dx = L / (N - 1)

    # Use precomputed pressure field, scaled by pressure_factor
    P = BASE_PRESSURE * pressure_factor
    vx, vy = compute_velocity(P, dx, K_value)
    D = compute_diffusion_coefficient(particle_size)

    # Initialize concentration
    C = np.zeros((N, N))
    C[:, 0] = 1.0

    delivery_time = None

    # Simulation loop
    max_steps = 600  # 120 seconds at dt=0.2 (calibrated, see TASKS.md Task 2)
    for step in range(max_steps):
        C = transport_step(C, vx, vy, D, uptake_rate, dx, dt)
        
        # Check target penetration every 10 steps to find delivery_time
        if step % 10 == 0 and delivery_time is None:
            depth = penetration_depth(C, threshold, dx)
            if depth >= target_penetration:
                delivery_time = step * dt
    
    # If target not reached, use final time
    if delivery_time is None:
        delivery_time = max_steps * dt

    # Extract metrics at final simulation time for consistent outputs
    final_depth = penetration_depth(C, threshold, dx)
    final_max_conc = C[:, 1:].max()  # Exclude inlet boundary where C is fixed to 1.0
    final_coverage = compute_drug_coverage(C, threshold)

    return {
        'ParticleSize': particle_size,
        'Pressure': pressure_factor * 20,  # Scale back to mmHg (normalized from 0-1)
        'HydraulicConductivity': K_value,
        'UptakeRate': uptake_rate,
        'Diffusion': D,
        'PenetrationDepth': final_depth,
        'MaxConcentration': final_max_conc,
        'DrugCoverage': final_coverage,
        'DeliveryTime': delivery_time
    }

# ============================================
# Generate Dataset
# ============================================

print("=" * 70)
print("PHASE 2: GENERATING COMPREHENSIVE AI DATASET")
print("=" * 70)

# Prepare task list for parallel execution
tasks = []
for ps, vp, (hc_name, hc_val), ur in product(
    PARTICLE_SIZES, VESSEL_PRESSURES, HYDRAULIC_CONDUCTIVITY.items(), UPTAKE_RATES
):
    pressure_factor = vp / 20.0
    tasks.append((ps, pressure_factor, hc_val, ur))

print(f"\nRunning {len(tasks)} simulations in parallel using all available cores...")

try:
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(ps, pf, hc, ur) for ps, pf, hc, ur in tasks
    )
    dataset = [r for r in results if r is not None]
except Exception as e:
    print(f"Error during parallel simulation execution: {e}")
    dataset = []

print(f"\nCompleted {len(dataset)} simulations successfully!")

# ============================================
# Save Dataset
# ============================================

df = pd.DataFrame(dataset)

# Save full dataset
output_file = RESULTS_DIR / "AI_dataset_comprehensive.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved to: {output_file}")

# Display statistics
print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("DATASET PREVIEW")
print("=" * 70)
print(df.head(10))

# ============================================
# Visualize Dataset Distribution
# ============================================

print("\nGenerating dataset visualization plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Penetration Depth vs Particle Size
axes[0, 0].scatter(df['ParticleSize'], df['PenetrationDepth'], alpha=0.5)
axes[0, 0].set_xlabel('Particle Size (nm)')
axes[0, 0].set_ylabel('Penetration Depth (mm)')
axes[0, 0].set_title('Penetration Depth vs Particle Size')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Max Concentration vs Particle Size
axes[0, 1].scatter(df['ParticleSize'], df['MaxConcentration'], alpha=0.5, color='orange')
axes[0, 1].set_xlabel('Particle Size (nm)')
axes[0, 1].set_ylabel('Max Concentration')
axes[0, 1].set_title('Max Concentration vs Particle Size')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Delivery Time vs Uptake Rate
axes[1, 0].scatter(df['UptakeRate'], df['DeliveryTime'], alpha=0.5, color='green')
axes[1, 0].set_xlabel('Uptake Rate')
axes[1, 0].set_ylabel('Delivery Time (s)')
axes[1, 0].set_title('Delivery Time vs Uptake Rate')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Drug Coverage vs Pressure
axes[1, 1].scatter(df['Pressure'], df['DrugCoverage'], alpha=0.5, color='red')
axes[1, 1].set_xlabel('Pressure (mmHg)')
axes[1, 1].set_ylabel('Drug Coverage (%)')
axes[1, 1].set_title('Drug Coverage vs Pressure')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "dataset_visualization.png", dpi=150, bbox_inches='tight')
print("Visualization saved to: results/dataset_visualization.png")

print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETE")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Features: {df.columns.tolist()}")
