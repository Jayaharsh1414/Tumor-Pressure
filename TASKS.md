# Phase 2 AI Project Task Plan

## Goal
Fix the dataset generation, training, evaluation, feature importance, and optimization pipeline so the AI results are valid, reproducible, and correctly reported.

## Tasks

1. Fix dataset generation in `generate_dataset.py`
   - Ensure parameter ranges are explicit and documented.
   - Confirm `MaxConcentration`, `PenetrationDepth`, `DrugCoverage`, and `DeliveryTime` vary properly.
   - Remove artificial constant forcing of outputs.
   - Generate at least 500–1000 valid samples.
   - Save dataset and dataset statistics.

2. Regenerate the dataset and verify output variation
   - Run `generate_dataset.py`.
   - Analyze the resulting `results/AI_dataset_comprehensive.csv`.
   - Confirm target and input ranges are valid.

3. Update training script in `src/train_ai.py`
   - Use explicit 80/20 train-test split.
   - Compare at least three models: Random Forest, Decision Tree, XGBoost (if available) or another strong regressor.
   - Calculate MAE, RMSE, and R² for each target and overall.
   - Add 5-fold cross-validation results.
   - Save model comparison table.

4. Fix feature importance and SHAP explanation
   - Ensure SHAP is computed correctly for each output.
   - Generate feature importance plots and verify they are not empty.
   - Add a parameter-range table to the report.

5. Fix optimization code in `src/train_ai.py`
   - Verify input scaling and feature order for prediction.
   - Use a wider grid of parameter combinations.
   - Report the best parameter set and predicted outputs.
   - Validate the predicted optimum with the original simulation.

6. Document the workflow
   - Add a parameter range table.
   - Explain dataset generation and model evaluation steps.
   - Save summary of changes and expected results.

## Progress
- [x] Task file created
- [x] Task 1: Fix dataset generation
- [x] Task 2: Regenerate and verify dataset
- [x] Task 3: Update training script
- [x] Task 4: Fix feature importance and SHAP
- [x] Task 5: Fix optimization code
- [x] Task 6: Document workflow

