"""
Phase 2: Complete Workflow Execution Guide
==========================================

This script runs the complete Phase 2 workflow:
1. Generate comprehensive dataset
2. Train ML models
3. Generate visualizations
4. Optimize parameters
5. Compare models
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_command(cmd, description):
    """Run a command and report results"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print("PHASE 2: AI-BASED PREDICTION AND OPTIMIZATION")
    print("=" * 70)
    
    # Check required packages
    print("\nChecking required packages...")
    packages_needed = {
        'xgboost': 'pip install xgboost',
        'shap': 'pip install shap',
        'seaborn': 'pip install seaborn'
    }
    
    for package, install_cmd in packages_needed.items():
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            print(f"  Install with: {install_cmd}")
    
    # Step 1: Generate Dataset
    success = run_command(
        f"{sys.executable} generate_dataset.py",
        "Step 1: Generate Comprehensive Dataset"
    )
    
    if not success:
        print("Failed to generate dataset. Please check the error above.")
        return False
    
    # Step 2: Train Models
    success = run_command(
        f"{sys.executable} src/train_ai.py",
        "Step 2: Train ML Models and Optimize"
    )
    
    if not success:
        print("Failed to train models. Please check the error above.")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 EXECUTION COMPLETE")
    print("=" * 70)
    print("\nGenerated Outputs:")
    print("  ✓ results/AI_dataset_comprehensive.csv - Full dataset")
    print("  ✓ figures/result6_actual_vs_predicted.png")
    print("  ✓ figures/result8_feature_importance.png")
    print("  ✓ figures/result9_optimization_landscape.png")
    print("  ✓ figures/result10_model_comparison.png")
    print("  ✓ models/best_model.pkl - Trained model")
    print("  ✓ models/scaler.pkl - Feature scaler")
    print("\nTo make predictions, run:")
    print(f"  {sys.executable} src/predict.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
