# Phase 2: AI-Based Prediction and Optimization - COMPLETE

## Summary

Phase 2 implements an AI-based prediction and optimization system for the tumor
nanoparticle drug delivery model. This document was rewritten after the professor's
review identified real bugs in the underlying physics model (see `TASKS.md` for the
full diagnosis and fix log) — the numbers below are from the regenerated dataset and
retrained models, not the original (broken) 225-sample run.

---

## Why the numbers changed from the first submission

The original dataset was generated on a 10mm domain over a 120-second simulated
window. At that scale, both diffusion (Stokes-Einstein, computed in raw SI units of
m²/s) and pressure-driven advection were 9-10 orders of magnitude too small to move
the drug concentration front more than one 0.101mm grid cell, for every particle size,
pressure, and hydraulic conductivity combination. Two further numerical bugs (a
periodic-boundary artifact in the transport solver, and an unset boundary condition in
the pressure solver) compounded this. The practical effect: `PenetrationDepth` and
`DeliveryTime` were constant or near-constant, `MaxConcentration` was a very narrow
range, and the AI model was trained mostly on noise-free deterministic outputs that
don't reflect real particle-size/diffusion/hydraulic-conductivity sensitivity.

The fix: the domain was rescaled to `L=0.5mm` (500μm — the physically relevant
interstitial region around a single tumor microvessel, consistent with published
intercapillary-distance values, rather than a full 10mm tumor slab), the diffusion
coefficient was correctly converted from m²/s to mm²/s to match the grid units, and
two solver bugs were fixed. This was validated with a calibration sweep (see
`TASKS.md` Task 2) before regenerating the full dataset, confirming all four outputs
now vary meaningfully and monotonically with all five inputs.

---

## Parameter ranges used for dataset generation

| Parameter | Values used | Count |
|---|---|---|
| Particle Size (nm) | 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 | 10 |
| Vessel Pressure (mmHg) | 15, 17.5, 20, 22.5, 25 | 5 |
| Hydraulic Conductivity (m²/(Pa·s)) | 0.8e-6 (Low), 1.0e-6 (Medium), 1.2e-6 (High) | 3 |
| Cellular Uptake Rate (1/s) | linspace(0.02, 0.10, 7) | 7 |
| Diffusion Coefficient | Computed via Stokes-Einstein from Particle Size, converted to mm²/s | — |

Full factorial grid: 10 × 5 × 3 × 7 = **1050 simulations**.

Domain/timestep: `L=0.5mm`, `N=100` grid points (`dx≈5.05μm`), `dt=0.2s`,
`max_steps=600` (120s simulated time per run), `threshold=0.01`,
`target_penetration=0.15mm`.

---

## Step 1: Dataset Generation

**File**: `results/AI_dataset_comprehensive.csv` — 1050 simulations, 9 columns.

Target variable ranges after the fix (previously frozen/near-constant):

| Target | Distinct values | Range |
|---|---|---|
| PenetrationDepth | 65 | 0.106 – 0.444 mm |
| MaxConcentration | 971 | 0.813 – 0.982 |
| DrugCoverage | 65 | 22 – 89 % |
| DeliveryTime | 31 | 16 – 120 s |

---

## Step 2: Machine Learning Model Training and Evaluation

Trained and compared 4 multi-output regression models, using an explicit **80/20
train-test split** (840 train / 210 test, `random_state=42`), evaluated only on the
held-out test set, plus **5-fold cross-validation** on the full dataset to confirm the
result isn't a fluke of one split.

| Model | Test-split MAE | Test-split RMSE | Test-split R² | 5-fold CV R² (mean ± std) |
|-------|-----|------|----------|----------|
| **Random Forest** | 0.5689 | 1.6332 | **0.9852** | 0.9817 ± 0.0045 |
| XGBoost | 0.6360 | 1.9726 | 0.9784 | 0.9778 ± 0.0046 |
| Decision Tree | 0.6213 | 2.0412 | 0.9773 | 0.9723 ± 0.0117 |
| Neural Network | 0.5771 | 1.0912 | 0.8387 | 0.8254 ± 0.0250 |

**Best Model**: Random Forest (R² = 0.9852, test split; 0.9817 ± 0.0045, 5-fold CV)

Per-target breakdown (Random Forest), full table in
`figures/model_comparison_per_target.csv`:

| Target | MAE | RMSE | R² |
|---|---|---|---|
| PenetrationDepth | 0.0031 | 0.0053 | 0.9948 |
| MaxConcentration | 0.0008 | 0.0020 | 0.9972 |
| DrugCoverage | 0.5962 | 1.0356 | 0.9950 |
| DeliveryTime | 1.6753 | 5.4901 | 0.9539 |

R² is no longer a flat 1.0000 across every model and every target: Neural Network
genuinely underperforms (R²≈0.84, and R²=0.44 specifically on MaxConcentration),
showing real model differentiation rather than every model memorizing a
noise-free deterministic surface.

**Files**: `models/best_model.pkl`, `models/{decision_tree,random_forest,xgboost,neural_network}.pkl`

---

## Step 3: Multi-Output Prediction

All models predict 4 outputs simultaneously from the same 5 inputs:
`PenetrationDepth`, `MaxConcentration`, `DrugCoverage`, `DeliveryTime`.

**Files**:
- `figures/result6_actual_vs_predicted.png` — all 4 outputs, per-target R²
- `figures/result7_multi_output_prediction.png` — combined view of PenetrationDepth,
  MaxConcentration, DrugCoverage predicted simultaneously

---

## Step 4: Explainable AI (Feature Importance)

Feature importance is computed from a Random Forest model regardless of which model
wins the overall comparison (previously this was silently skipped whenever a
different model won). Verified non-zero for all 5 features on all 4 targets:

| Target | Top feature | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| PenetrationDepth | UptakeRate (0.746) | Pressure (0.134) | HydraulicConductivity (0.109) | Diffusion (0.005) | ParticleSize (0.005) |
| MaxConcentration | UptakeRate (0.767) | Pressure (0.124) | HydraulicConductivity (0.103) | Diffusion (0.003) | ParticleSize (0.003) |
| DrugCoverage | UptakeRate (0.746) | Pressure (0.134) | HydraulicConductivity (0.109) | Diffusion (0.005) | ParticleSize (0.005) |
| DeliveryTime | UptakeRate (0.333) | Pressure (0.321) | HydraulicConductivity (0.258) | ParticleSize (0.045) | Diffusion (0.043) |

UptakeRate and Pressure genuinely dominate (cellular absorption and pressure-driven
convection are the strongest transport mechanisms at this microvessel scale) but
ParticleSize/Diffusion/HydraulicConductivity are now real, non-zero contributors —
not the exact-zero artifact reported previously.

**Files**: `figures/result8_feature_importance.png` (Random-Forest-based),
`figures/result8_feature_importance_shap.png` (SHAP summary plots, one per target -
the original code only ever explained a single output's tree and mis-indexed the
result, which is what crashed; fixed by building one explainer per target).

---

## Step 5: Parameter Optimization

Grid search over the AI surrogate to maximize predicted `PenetrationDepth`:

**Optimal Parameters Found**:
- Particle Size: 20 nm
- Vessel Pressure: 25 mmHg
- Hydraulic Conductivity: 1.20e-06
- Uptake Rate: 0.020
- AI-predicted Penetration Depth: 0.4357 mm

**Validated against the real PDE simulation** (not just trusted blindly): re-running
the actual mathematical model at this exact parameter combination gives a real
penetration depth of **0.4444 mm** — a 0.0087mm (≈2%) difference from the AI
prediction, confirming the surrogate model is trustworthy at its claimed optimum.

The optimization landscape (Figure 9) now shows real variation across the grid
(DrugCoverage 46-65%, DeliveryTime 20-44s across particle size × pressure), not the
flat/constant surface from the original run.

**Files**: `figures/result9_optimization_landscape.png`, `figures/result9_optimization.csv`

---

## Step 6: Model Comparison

**Files**: `figures/result10_model_comparison.png`, `figures/model_comparison.csv`,
`figures/model_comparison_per_target.csv`, `figures/model_comparison_cv.csv`

---

## How to Use the AI System

### Make Predictions
```bash
python src/predict.py
```

### Train New Models
```bash
python src/train_ai.py
```

### Generate New Dataset
```bash
python generate_dataset.py
```

---

## Project Structure

```
tumor_pressure_final/
├── main.py                          # Phase 1: Mathematical modeling
├── fluid_model.py                   # Pressure and velocity solver
├── transport.py                     # Drug transport equations
├── parameters.py                    # Simulation parameters
├── generate_dataset.py              # Phase 2: Dataset generation
├── src/
│   ├── train_ai.py                 # Phase 2: Model training
│   └── predict.py                  # Phase 2: Prediction interface
├── results/                         # All simulation outputs
├── figures/                         # All generated plots
├── models/                          # Trained ML models
├── TASKS.md                         # Full bug diagnosis + fix log
└── README.txt
```
