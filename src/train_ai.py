"""
Phase 2: Comprehensive AI Training and Optimization
Implements:
- Multi-output regression
- Model training and comparison
- SHAP explainability
- Parameter optimization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fluid_model import solve_pressure, compute_velocity
from transport import transport_step, penetration_depth as sim_penetration_depth
import parameters as sim_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# ============================================
# Setup
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================
# Step 1: Load and Prepare Dataset
# ============================================

print("=" * 70)
print("PHASE 2: AI-BASED PREDICTION AND OPTIMIZATION")
print("=" * 70)

# Load dataset
dataset_file = RESULTS_DIR / "AI_dataset_comprehensive.csv"
if not dataset_file.exists():
    print(f"ERROR: Dataset not found at {dataset_file}")
    print("Please run generate_dataset.py first to create the comprehensive dataset.")
    exit(1)

df = pd.read_csv(dataset_file)
print(f"\nLoaded dataset: {len(df)} samples")
print(f"Features: {df.columns.tolist()}")
print("\nDataset Preview:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# ============================================
# Step 2: Prepare Features and Targets
# ============================================

# Features (inputs)
feature_cols = ['ParticleSize', 'Pressure', 'HydraulicConductivity', 'UptakeRate', 'Diffusion']
X = df[feature_cols].copy()

# Targets (multi-output regression)
target_cols = ['PenetrationDepth', 'MaxConcentration', 'DrugCoverage', 'DeliveryTime']
y = df[target_cols].copy()

print(f"\nFeatures: {feature_cols}")
print(f"Targets: {target_cols}")

# Handle any missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Save scaler for later use
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# ============================================
# Step 3: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================================
# Step 4: Define Models
# ============================================

models = {}

# Random Forest (Multi-output)
models['Random Forest'] = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
)

# Decision Tree (Multi-output)
models['Decision Tree'] = MultiOutputRegressor(
    DecisionTreeRegressor(random_state=42, max_depth=15)
)

# Neural Network (Multi-output)
models['Neural Network'] = MultiOutputRegressor(
    MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42)
)

# XGBoost (if available)
if HAS_XGBOOST:
    models['XGBoost'] = MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    )

# ============================================
# Step 5: Train Models
# ============================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

trained_models = {}
predictions = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    predictions[model_name] = {'train': y_pred_train, 'test': y_pred_test}
    
    print(f"✓ {model_name} trained successfully")

# ============================================
# Step 6: Evaluate Models
# ============================================

print("\n" + "=" * 70)
print("MODEL EVALUATION - MULTI-OUTPUT METRICS")
print("=" * 70)

results = []
per_target_results = []

for model_name, model in trained_models.items():
    print(f"\n{model_name}")
    print("-" * 70)

    y_pred = predictions[model_name]['test']

    # Calculate metrics for each output
    mae_list = []
    rmse_list = []
    r2_list = []

    for i, target in enumerate(target_cols):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        per_target_results.append({
            'Model': model_name, 'Target': target,
            'MAE': mae, 'RMSE': rmse, 'R2': r2
        })

        print(f"  {target:20s}: MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")

    # Overall metrics (average across outputs)
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)

    print(f"  {'Average':20s}: MAE={avg_mae:.5f}, RMSE={avg_rmse:.5f}, R²={avg_r2:.5f}")

    results.append({
        'Model': model_name,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'R2': avg_r2
    })

# Create comparison dataframe
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('R2', ascending=False)

print("\n" + "=" * 70)
print("MODEL COMPARISON (SORTED BY R²)")
print("=" * 70)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(FIGURES_DIR / "model_comparison.csv", index=False)

# Save the full per-target metrics table (not just the 4-model average) so a
# reviewer can see MAE/RMSE/R2 for each individual output, not just an average.
per_target_df = pd.DataFrame(per_target_results)
per_target_df.to_csv(FIGURES_DIR / "model_comparison_per_target.csv", index=False)
print("\n✓ Saved per-target metrics: figures/model_comparison_per_target.csv")

# ============================================
# Step 6b: 5-Fold Cross-Validation
# ============================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION (R², full dataset)")
print("=" * 70)

from sklearn.model_selection import KFold, cross_validate

cv_results = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for model_name, model in models.items():
    cv_scores = cross_validate(
        model, X_scaled, y, cv=kfold, scoring='r2', n_jobs=-1
    )
    r2_mean = cv_scores['test_score'].mean()
    r2_std = cv_scores['test_score'].std()
    print(f"  {model_name:20s}: R² = {r2_mean:.5f} ± {r2_std:.5f}")
    cv_results.append({'Model': model_name, 'CV_R2_mean': r2_mean, 'CV_R2_std': r2_std})

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(FIGURES_DIR / "model_comparison_cv.csv", index=False)
print("✓ Saved cross-validation results: figures/model_comparison_cv.csv")

# ============================================
# Step 7: Visualize Predictions
# ============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
y_pred_best = predictions[best_model_name]['test']

print(f"\nBest Model: {best_model_name} (R² = {comparison_df.iloc[0]['R2']:.5f})")

# Plot 1: Actual vs Predicted for each output
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, target in enumerate(target_cols):
    ax = axes[i]
    ax.scatter(y_test.iloc[:, i], y_pred_best[:, i], alpha=0.5)
    
    # Add diagonal line
    min_val = min(y_test.iloc[:, i].min(), y_pred_best[:, i].min())
    max_val = max(y_test.iloc[:, i].max(), y_pred_best[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    r2 = r2_score(y_test.iloc[:, i], y_pred_best[:, i])
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target} (R² = {r2:.4f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "result6_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
print("✓ Saved: result6_actual_vs_predicted.png")

# ============================================
# Result 7: Multi-Output Prediction Summary
# ============================================
# A single combined view of the model predicting PenetrationDepth,
# MaxConcentration, and DrugCoverage simultaneously from the same 5 inputs -
# distinct from Result 6's per-target breakdown (which includes DeliveryTime).

multi_output_targets = ['PenetrationDepth', 'MaxConcentration', 'DrugCoverage']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, target in zip(axes, multi_output_targets):
    i = target_cols.index(target)
    ax.scatter(y_test.iloc[:, i], y_pred_best[:, i], alpha=0.5, c='teal')
    min_val = min(y_test.iloc[:, i].min(), y_pred_best[:, i].min())
    max_val = max(y_test.iloc[:, i].max(), y_pred_best[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    r2 = r2_score(y_test.iloc[:, i], y_pred_best[:, i])
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target}\n(R² = {r2:.4f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(f'Result 7: Multi-Output Prediction ({best_model_name})', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "result7_multi_output_prediction.png", dpi=150, bbox_inches='tight')
print("✓ Saved: result7_multi_output_prediction.png")

# Plot 2: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

metrics = ['MAE', 'RMSE', 'R2']
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric])
    ax.set_ylabel(metric)
    ax.set_title(f'Model Comparison - {metric}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color the best one differently
    bars[0].set_color('green')
    bars[0].set_alpha(0.7)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "result10_model_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Saved: result10_model_comparison.png")

# ============================================
# Step 8: Feature Importance (SHAP)
# ============================================

print("\n" + "=" * 70)
print("STEP 4: EXPLAINABLE AI (SHAP)")
print("=" * 70)

# Feature importance/SHAP are always computed from the Random Forest model
# specifically (it's always trained above, regardless of which model wins the
# comparison) rather than being gated on "Random Forest happens to be best" -
# that gate meant this figure was silently skipped whenever another model won.
importance_model = trained_models['Random Forest']

if HAS_SHAP:
    print("\nGenerating SHAP explanations...")

    try:
        # One explainer per output: importance_model.estimators_[i] is a
        # single-output tree for target_cols[i]. The original code only ever
        # explained estimators_[0] (PenetrationDepth) once and then indexed
        # into its 2D (samples x features) SHAP matrix as if it held one
        # matrix per target - that's what crashed ("vector not matrix").
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, target in enumerate(target_cols):
            base_model = importance_model.estimators_[i]
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_test)

            ax = axes[i]
            plt.sca(ax)
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                            plot_type="bar", show=False)
            ax.set_title(f'{target} - Feature Importance')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "result8_feature_importance_shap.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: result8_feature_importance_shap.png")

    except Exception as e:
        print(f"SHAP visualization error: {e}")
        print("Creating alternative feature importance plot...")

print("Generating feature importance from Random Forest...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, target in enumerate(target_cols):
    ax = axes[i]
    importances = importance_model.estimators_[i].feature_importances_

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_cols[j] for j in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title(f'{target} - Feature Importance')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "result8_feature_importance.png", dpi=150, bbox_inches='tight')
print("✓ Saved: result8_feature_importance.png")

# ============================================
# Step 9: Optimization
# ============================================

print("\n" + "=" * 70)
print("STEP 5: AI OPTIMIZATION")
print("=" * 70)

print("\nPerforming grid search optimization...")

# Grid search for optimal parameters
particle_sizes = np.linspace(20, 200, 10)
pressures = np.linspace(15, 25, 5)
uptake_rates = np.linspace(0.02, 0.10, 5)
hydraulic_cond = np.linspace(0.8e-6, 1.2e-6, 3)

# Target: Maximum penetration depth
best_penetration = -1
best_params = None

for ps in particle_sizes:
    for p in pressures:
        for ur in uptake_rates:
            for hc in hydraulic_cond:
                # Compute diffusion using Stokes-Einstein, converted to mm^2/s
                # to match the units the model was trained on (generate_dataset.py) -
                # without this conversion D is fed to the model 1e6x out of its
                # training distribution and the optimizer effectively ignores it.
                kB = 1.380649e-23
                T = 310
                mu = 1e-3
                r = ps * 1e-9
                D = (kB * T / (3 * np.pi * mu * r)) * 1e6

                # Create input
                input_data = np.array([[ps, p, hc, ur, D]])
                input_scaled = scaler.transform(input_data)
                
                # Predict
                pred = best_model.predict(input_scaled)[0]
                penetration = pred[0]  # Penetration depth
                
                if penetration > best_penetration:
                    best_penetration = penetration
                    best_params = {
                        'ParticleSize': ps,
                        'Pressure': p,
                        'HydraulicConductivity': hc,
                        'UptakeRate': ur,
                        'Diffusion': D
                    }

print(f"\nOptimal Parameters (Grid Search):")
print(f"  Particle Size: {best_params['ParticleSize']:.1f} nm")
print(f"  Vessel Pressure: {best_params['Pressure']:.1f} mmHg")
print(f"  Hydraulic Conductivity: {best_params['HydraulicConductivity']:.2e}")
print(f"  Uptake Rate: {best_params['UptakeRate']:.3f}")
print(f"  Expected Penetration Depth: {best_penetration:.4f} mm")

# ============================================
# Step 5b: Validate the AI-predicted optimum against the real simulator
# ============================================
# The AI surrogate is only useful if it agrees with the actual PDE model at
# the point it claims is optimal - re-run the real simulation there and
# compare, rather than trusting the surrogate's prediction blindly.

print("\n" + "=" * 70)
print("VALIDATING AI OPTIMUM AGAINST THE REAL PDE SIMULATION")
print("=" * 70)

dx_sim = sim_params.L / (sim_params.N - 1)
base_pressure_sim = solve_pressure(sim_params.N)
pressure_factor_sim = best_params['Pressure'] / 20.0
P_sim = base_pressure_sim * pressure_factor_sim
vx_sim, vy_sim = compute_velocity(P_sim, dx_sim, best_params['HydraulicConductivity'])

C_sim = np.zeros((sim_params.N, sim_params.N))
C_sim[:, 0] = 1.0
max_steps_sim = 600
for step in range(max_steps_sim):
    C_sim = transport_step(
        C_sim, vx_sim, vy_sim, best_params['Diffusion'],
        best_params['UptakeRate'], dx_sim, sim_params.dt
    )
real_depth = sim_penetration_depth(C_sim, sim_params.threshold, dx_sim)

print(f"  AI-predicted PenetrationDepth:   {best_penetration:.4f} mm")
print(f"  Real simulation PenetrationDepth: {real_depth:.4f} mm")
print(f"  Absolute difference:             {abs(best_penetration - real_depth):.4f} mm")

# Create optimization result table
opt_results = []
for ps in [20, 50, 100, 150, 200]:
    for p in [15, 20, 25]:
        kB = 1.380649e-23
        T = 310
        mu = 1e-3
        r = ps * 1e-9
        D = (kB * T / (3 * np.pi * mu * r)) * 1e6  # mm^2/s, matches training units

        input_data = np.array([[ps, p, 1.0e-6, 0.05, D]])
        input_scaled = scaler.transform(input_data)
        pred = best_model.predict(input_scaled)[0]
        
        opt_results.append({
            'ParticleSize': ps,
            'Pressure': p,
            'PredictedPenetration': pred[0],
            'PredictedMaxConc': pred[1],
            'PredictedCoverage': pred[2],
            'DeliveryTime': pred[3]
        })

opt_df = pd.DataFrame(opt_results)

print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS (Sample Grid)")
print("=" * 70)
print(opt_df.head(15))

# Save optimization results
opt_df.to_csv(FIGURES_DIR / "result9_optimization.csv", index=False)

# Visualize optimization landscape
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Penetration depth heatmap
pivot_penet = opt_df.pivot_table(values='PredictedPenetration', 
                                  index='ParticleSize', 
                                  columns='Pressure')
sns.heatmap(pivot_penet, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
axes[0].set_title('Predicted Penetration Depth vs Parameters')
axes[0].set_ylabel('Particle Size (nm)')
axes[0].set_xlabel('Vessel Pressure (mmHg)')

# Delivery time heatmap
pivot_time = opt_df.pivot_table(values='DeliveryTime', 
                                 index='ParticleSize', 
                                 columns='Pressure')
sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='plasma', ax=axes[1])
axes[1].set_title('Predicted Delivery Time vs Parameters')
axes[1].set_ylabel('Particle Size (nm)')
axes[1].set_xlabel('Vessel Pressure (mmHg)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "result9_optimization_landscape.png", dpi=150, bbox_inches='tight')
print("✓ Saved: result9_optimization_landscape.png")

# ============================================
# Step 10: Save Best Model
# ============================================

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
print(f"✓ Saved best model: {best_model_name}")

# Save all models
for name, model in trained_models.items():
    joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    print(f"✓ Saved: {name}")

# ============================================
# Summary Report
# ============================================

print("\n" + "=" * 70)
print("PHASE 2 SUMMARY REPORT")
print("=" * 70)

print(f"\n✓ Dataset generated: {len(df)} simulations")
print(f"✓ Features: {', '.join(feature_cols)}")
print(f"✓ Outputs: {', '.join(target_cols)}")
print(f"\n✓ Models trained: {len(trained_models)}")
print(f"✓ Best model: {best_model_name} (R² = {comparison_df.iloc[0]['R2']:.5f})")
print(f"\n✓ Optimal parameters found:")
print(f"  - Particle Size: {best_params['ParticleSize']:.1f} nm")
print(f"  - Vessel Pressure: {best_params['Pressure']:.1f} mmHg")
print(f"  - Uptake Rate: {best_params['UptakeRate']:.3f}")
print(f"  - Expected Penetration: {best_penetration:.4f} mm")

print(f"\n✓ Outputs generated:")
print(f"  - Result 6: Actual vs Predicted (result6_actual_vs_predicted.png)")
print(f"  - Result 8: Feature Importance (result8_feature_importance.png)")
print(f"  - Result 9: Optimization (result9_optimization_landscape.png)")
print(f"  - Result 10: Model Comparison (result10_model_comparison.png)")

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE!")
print("=" * 70)
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.savefig(
    RESULTS_DIR / "AI_actual_vs_predicted.png"
)

plt.close()

print("\nFinished Successfully!")