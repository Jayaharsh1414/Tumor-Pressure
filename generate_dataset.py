"""
Phase 2: Generate Comprehensive AI Dataset
Generates ~1050 simulations using Latin Hypercube Sampling over 5 independent
parameters (Review 2 comment: "use random or Latin Hypercube sampling instead
of testing only a few fixed parameter values"):
- Particle size (20-200 nm)
- Vessel pressure (15-25 mmHg)
- Hydraulic conductivity (0.8e-6 - 1.2e-6)
- Cellular uptake rate (0.02-0.10)
- Drug release rate (0.005-0.10 /s) - first-order release kinetics, new in
  Review 2: the vessel-wall boundary concentration ramps up over time as
  C(0,t) = 1 - exp(-k_release*t) instead of being instantly fixed at 1,
  modeling a nanoparticle carrier releasing its payload over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import qmc

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
# Parameter Ranges (Latin Hypercube Sampling)
# ============================================

N_SAMPLES = 1050  # matches the previous full-factorial dataset size

PARAM_RANGES = {
    'ParticleSize': (20.0, 200.0),          # nm
    'Pressure': (15.0, 25.0),               # mmHg
    'HydraulicConductivity': (0.8e-6, 1.2e-6),
    'UptakeRate': (0.02, 0.10),             # 1/s
    'DrugReleaseRate': (0.005, 0.10),       # 1/s (first-order release kinetics)
}
PARAM_NAMES = list(PARAM_RANGES.keys())

sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=42)
unit_samples = sampler.random(n=N_SAMPLES)
lower_bounds = [PARAM_RANGES[p][0] for p in PARAM_NAMES]
upper_bounds = [PARAM_RANGES[p][1] for p in PARAM_NAMES]
scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

param_df = pd.DataFrame(scaled_samples, columns=PARAM_NAMES)

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

def run_simulation(particle_size, pressure_mmhg, K_value, uptake_rate, release_rate, N=100, dt=dt):
    """
    Run a single simulation and return output metrics. The vessel-wall
    boundary concentration ramps up over time following first-order release
    kinetics (release_rate = k_release, units 1/s):
        C(0, t) = 1 - exp(-k_release * t)
    """
    dx = L / (N - 1)
    pressure_factor = pressure_mmhg / 20.0

    # Use precomputed pressure field, scaled by pressure_factor
    P = BASE_PRESSURE * pressure_factor
    vx, vy = compute_velocity(P, dx, K_value)
    D = compute_diffusion_coefficient(particle_size)

    # Initialize concentration
    C = np.zeros((N, N))
    C[:, 0] = 1 - np.exp(-release_rate * 0.0)

    delivery_time = None

    # Simulation loop
    max_steps = 600  # 120 seconds at dt=0.2 (calibrated, see TASKS.md Task 2)
    for step in range(max_steps):
        t_next = (step + 1) * dt
        boundary_value = 1 - np.exp(-release_rate * t_next)
        C = transport_step(C, vx, vy, D, uptake_rate, dx, dt, boundary_value=boundary_value)

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
        'Pressure': pressure_mmhg,
        'HydraulicConductivity': K_value,
        'UptakeRate': uptake_rate,
        'DrugReleaseRate': release_rate,
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
print("PHASE 2: GENERATING COMPREHENSIVE AI DATASET (Latin Hypercube Sampling)")
print("=" * 70)

tasks = list(param_df.itertuples(index=False, name=None))

print(f"\nRunning {len(tasks)} LHS-sampled simulations in parallel using all available cores...")
print(f"Parameter ranges: {PARAM_RANGES}")

try:
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(ps, pf, hc, ur, rr) for ps, pf, hc, ur, rr in tasks
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].scatter(df['ParticleSize'], df['PenetrationDepth'], alpha=0.5)
axes[0, 0].set_xlabel('Particle Size (nm)')
axes[0, 0].set_ylabel('Penetration Depth (mm)')
axes[0, 0].set_title('Penetration Depth vs Particle Size')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(df['ParticleSize'], df['MaxConcentration'], alpha=0.5, color='orange')
axes[0, 1].set_xlabel('Particle Size (nm)')
axes[0, 1].set_ylabel('Max Concentration')
axes[0, 1].set_title('Max Concentration vs Particle Size')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(df['DrugReleaseRate'], df['PenetrationDepth'], alpha=0.5, color='purple')
axes[0, 2].set_xlabel('Drug Release Rate (1/s)')
axes[0, 2].set_ylabel('Penetration Depth (mm)')
axes[0, 2].set_title('Penetration Depth vs Drug Release Rate')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].scatter(df['UptakeRate'], df['DeliveryTime'], alpha=0.5, color='green')
axes[1, 0].set_xlabel('Uptake Rate')
axes[1, 0].set_ylabel('Delivery Time (s)')
axes[1, 0].set_title('Delivery Time vs Uptake Rate')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(df['Pressure'], df['DrugCoverage'], alpha=0.5, color='red')
axes[1, 1].set_xlabel('Pressure (mmHg)')
axes[1, 1].set_ylabel('Drug Coverage (%)')
axes[1, 1].set_title('Drug Coverage vs Pressure')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].scatter(df['DrugReleaseRate'], df['MaxConcentration'], alpha=0.5, color='brown')
axes[1, 2].set_xlabel('Drug Release Rate (1/s)')
axes[1, 2].set_ylabel('Max Concentration')
axes[1, 2].set_title('Max Concentration vs Drug Release Rate')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "dataset_visualization.png", dpi=150, bbox_inches='tight')
print("Visualization saved to: results/dataset_visualization.png")

print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETE")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Features: {df.columns.tolist()}")
print(f"Sampling method: Latin Hypercube Sampling (5 independent dimensions)")
