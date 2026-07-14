
# Parameters
K = 1e-6  # Default hydraulic conductivity (m²/(Pa·s))
mu = 1e-3  # Dynamic viscosity (Pa·s)
ku = 0.005  # Default cellular uptake rate (1/s)

L = 0.5       # Domain length (mm) - interstitial region around a single tumor
              # microvessel (~500um), not a full 10mm tumor slab. At the
              # original 10mm/120s scale, diffusion and pressure-driven
              # advection were both too small to move the concentration front
              # more than one grid cell (see TASKS.md Task 2 for the
              # calibration that led to this value).
N = 100       # Grid points

dt = 0.2      # Time step (s)
threshold = 0.01  # Concentration threshold for penetration depth and coverage

target_penetration = 0.15  # Target penetration depth in mm for delivery time
                            # (was 1.0mm, impossible on a 0.5mm domain)

# Physical constants
kB = 1.380649e-23  # Boltzmann constant (J/K)
T = 310  # Temperature (K, body temperature)
