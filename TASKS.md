# Phase 2 AI Project — Fix Plan (Professor Review, due Thu 2026-07-16)

## Root cause (confirmed empirically, not guessed)

1. **Pressure solver BC bug** — `fluid_model.py:solve_pressure()` never sets or updates
   the top/bottom rows (`P[0,:]`, `P[-1,:]`) except at the two corners. They stay frozen
   at their initial value of 0, which injects a spurious, spatially-concentrated velocity
   spike at the domain edges instead of a smooth physical flow field.
   Confirmed by direct test: `vx` max before fix = 0.0099, entirely from the artifact rows;
   after adding a no-flux (Neumann) condition on those rows, `vx` drops to a smooth
   3e-5–1.8e-4 range.

2. **Diffusion/advection are numerically invisible on the current grid+timescale** —
   `parameters.py` defines the domain in millimetres (`L=10.0`, `dx≈0.101mm`) but
   `compute_diffusion_coefficient()` (Stokes–Einstein) and hydraulic conductivity are
   computed in raw SI units (m²/s, m²/(Pa·s)). Even after fixing bug #1, a direct
   simulation test (20/100/200nm particles, 2400 steps, 120s) shows `PenetrationDepth`
   frozen at exactly one grid cell (0.101mm) for every particle size, forever, because
   `D ~ 1e-12–1e-11` and `v ~ 1e-4 mm/s` are both far too small to cross even one more
   0.101mm grid cell within 120 simulated seconds.
   This single fact is the direct cause of:
   - `PenetrationDepth` having only 2 distinct values across all 1050 rows
   - `DeliveryTime` being **exactly** 120.0 for all 1050 rows (the 1mm target in
     `parameters.py: target_penetration=1.0` is never reached, so it always falls back
     to `max_steps*dt`)
   - Zero feature importance for `ParticleSize`, `Diffusion`, `HydraulicConductivity`
     (they have no real effect on the broken outputs)
   - `Pressure`/`UptakeRate` "dominating" everything (uptake decay is the only
     transport term operating at a normal magnitude)
   - Trivial/near-perfect R² on 2 of 4 targets, since they're constant or binary

3. `results/AI_dataset.csv` (Phase 1, timestamped Jul 7) shows smooth, physically
   sensible depth-vs-size variation, but it is **not reproducible by the current
   `main.py`** (git-diff confirms `main.py` is unchanged since the first commit, yet
   directly re-running it freezes exactly like the Phase 2 data). It's orphaned data
   from an earlier/different script, not a working reference — don't use it as a
   sanity check.

4. `src/train_ai.py` — SHAP + feature-importance plotting block is gated on
   `best_model_name == 'Random Forest'` (lines ~293, ~325). Last run picked Decision
   Tree as best, so **no feature-importance figure is generated at all** in that case.

5. Last training run crashed outright (`results/_train_ai_log.txt`):
   `ModuleNotFoundError: No module named 'seaborn'`. The figures/models currently in
   `figures/` and `models/` are stale, from the old 225-sample dataset your professor
   already reviewed — the regenerated 1050-row dataset has never actually been trained on.

6. No `Result 7` (multi-output prediction summary) is generated anywhere, despite being
   referenced in the professor's figure numbering.

---

## Tasks

- [ ] **Task 1 — Fix the pressure-solver boundary condition**
  - File: `fluid_model.py`
  - In `solve_pressure`, after each Jacobi update (or after the loop, each iteration),
    add no-flux boundary rows: `P[0,:] = P[1,:]` and `P[-1,:] = P[-2,:]`, then re-apply
    `P[:,0] = 1.0` / `P[:,-1] = 0.0`.
  - Validate: re-run the calibration snippet used above; confirm `vx`/`vy` are smooth
    (no order-of-magnitude spike at edges) and `grid_independence.csv`-style check
    (max velocity vs grid size) still looks monotonic/sane.

- [x] **Task 2 — Rescale the domain/timescale so transport is observable**

  **Calibration result (confirmed by direct simulation, not guessed):**
  - Final parameters: `L=0.5` mm (500 μm microvessel-scale domain), `N=100`
    (`dx≈5.05μm`), `dt=0.2s`, `max_steps=600` (120s total — same duration as the
    original design, so `UPTAKE_RATES` did **not** need rescaling), `threshold=0.01`
    (unchanged), `target_penetration=0.15` mm (was 1.0mm, impossible on this domain).
  - Stokes-Einstein `D` converted from SI (m²/s) to mm²/s (×1e6) at point of use.
    `HydraulicConductivity` needed **no** conversion — empirically already gives
    sensible mm/s-scale velocities against the new `dx`.
  - Two additional real bugs found and fixed along the way (not present in the
    original diagnosis, found via direct testing):
    1. `fluid_model.py: solve_pressure` never updated/set the top/bottom rows —
       they stayed frozen at 0, creating a spurious velocity spike at the domain
       edges. Fixed with a no-flux (Neumann) condition on those rows each iteration.
    2. `transport.py: transport_step` used `np.roll` for the x-direction, making the
       domain periodic — the fixed inlet (`C[:,0]=1`) wrapped around and instantly
       "leaked" onto the far edge regardless of real transport, and separately the
       central-difference advection term oscillates/overshoots C above 1.0 once the
       grid Péclet number exceeds ~2. Fixed with non-periodic (Neumann far edge)
       differencing and upwind advection.
  - Calibration sweep results (120s, `target_penetration=0.15mm`):

    | Sweep | Result |
    |---|---|
    | Particle size 20→200nm (P=1.0, K=1e-6, ku=0.05) | depth 0.283→0.263mm, delivT 24→32s (monotonic, smaller=deeper, correct) |
    | Pressure factor 0.75→1.25 (size=100nm) | depth 0.222→0.303mm, delivT 42→26s (monotonic) |
    | Uptake rate 0.02→0.10 (size=100nm, P=1.0) | depth 0.343→0.162mm (strong monotonic decrease, correct) |
    | Hydraulic conductivity 0.8e-6→1.2e-6 | depth 0.232→0.298mm (monotonic) |

  All 4 inputs now produce real, monotonic, physically-sensible, non-constant
  output variation — confirmed before touching the full dataset.

- [ ] **Task 3 — Regenerate the dataset**
  - File: `generate_dataset.py`
  - Apply the calibrated parameters from Task 2 (`L=0.5`, `dt=0.2`, `max_steps=600`,
    `target_penetration=0.15`, `D` converted ×1e6 to mm²/s) inside `run_simulation`.
  - Re-run and confirm via `df.nunique()` / `df.describe()` that all 4 targets now have
    meaningfully many distinct values (not 1-2), and that `results/dataset_visualization.png`
    shows real spread (not a single visible point) for `MaxConcentration`.
  - Check wall-clock time for the full 1050-run sweep; `max_steps` dropped from 2400
    to 600 so this should be faster than the original run, not slower.

- [x] **Task 4 — Fix environment / retrain**
  - `seaborn`/`shap`/`xgboost` were all already installed; the earlier crash log was
    stale. Retrained end-to-end on the regenerated dataset with no exceptions.

- [x] **Task 5 — Report real per-target metrics + cross-validation**
  - Per-target MAE/RMSE/R² saved to `figures/model_comparison_per_target.csv`.
  - 5-fold CV added (`figures/model_comparison_cv.csv`). Random Forest: single-split
    R²=0.98524, CV R²=0.98168 ± 0.00447 — consistent, not a fluke of one split.
    Neural Network genuinely underperforms (R²≈0.84) — real model differentiation,
    not everything pinned to 1.0000.

- [x] **Task 6 — Fix the feature-importance/SHAP gating bug**
  - Feature importance now always computed from the Random Forest model regardless of
    which model wins. Verified non-zero for all 5 features on all 4 targets (was
    exactly 0.0 for ParticleSize/Diffusion/HydraulicConductivity before):
    UptakeRate/Pressure still dominate (physically genuine — matches the Task 2
    calibration sweep), but ParticleSize/Diffusion/HydraulicConductivity are now real,
    non-zero, not an artifact.
  - SHAP was crashing on a real bug, not a version issue: it only ever explained
    `estimators_[0]` (the PenetrationDepth-only tree) once, then indexed into its
    single 2D SHAP matrix as if it held one matrix per target. Fixed by building one
    `TreeExplainer` per target's own estimator. `figures/result8_feature_importance_shap.png`
    now generates successfully (professor explicitly asked for SHAP in addition to
    plain feature importance).

- [x] **Task 7 — Fix / validate the optimization landscape (Figure 9)**
  - Found and fixed a second critical bug while validating: both optimization grids
    computed the Stokes-Einstein diffusion coefficient WITHOUT the ×1e6 mm²/s
    conversion added in Task 2/3 — feeding the model D values 1e6x outside its
    training distribution at prediction time. Fixed both spots.
  - Figure 9 now shows real variation: DrugCoverage 46-65%, DeliveryTime 20-44s across
    the particle-size × pressure grid (previously flat).
  - Validated AI optimum against the real PDE simulation: AI predicted 0.4357mm,
    actual simulation gave 0.4444mm at the same parameters (2% relative difference).

- [x] **Task 8 — Add the missing Result 7 (multi-output prediction)**
  - Added `figures/result7_multi_output_prediction.png` (PenetrationDepth,
    MaxConcentration, DrugCoverage actual-vs-predicted in one view).

- [ ] **Task 9 — Update documentation**
  - Files: `PHASE2_COMPLETE.md`, `docs/report_notes.txt`
  - Replace the old (stale, 225-sample, R²=1.0000) numbers with the real numbers from
    the regenerated pipeline.
  - Add an explicit parameter-range table (all 5 inputs, min/max/values used) and a
    short paragraph stating the 80/20 train/test split and 5-fold CV explicitly, since
    the professor could not tell either was happening from the report alone.
  - Document the domain/unit rescaling from Task 2 and why it was necessary — this is
    the one substantive modeling change and should be explained, not hidden.

- [x] **Task 10 — Final validation pass**

  Every professor comment checked against the regenerated artifacts:

  | Professor's comment | Status |
  |---|---|
  | R²=1.0000 unrealistic, verify | Fixed — R² now 0.84–0.99, varies by model |
  | Verify evaluated on unseen test data, not training data | Already correct in code (80/20 split); now explicitly documented |
  | Clearly state train-test split in code + report | Documented in `PHASE2_COMPLETE.md` |
  | Report MAE/RMSE, not just R² | Per-target CSV + printed table added |
  | Increase dataset to 500-1000 samples | Already 1050 — no change needed |
  | `MaxConcentration` plot is a single point / constant | Fixed — 971 unique values, 0.813-0.982 range |
  | Compare ≥3 models (RF, XGBoost, Decision Tree) | Already 4 models compared — no change needed |
  | Explain dataset generation from the math model, reproducibly | Documented in `PHASE2_COMPLETE.md` + this file |
  | Feature importance incomplete/inconsistent, `MaxConcentration` importance empty | Fixed — gating bug removed, all 4 targets populated |
  | Zero importance for ParticleSize/Diffusion/HydraulicConductivity | Fixed — all now non-zero |
  | Pressure/UptakeRate dominate — check for dataset bias | Confirmed genuine (calibration sweep), documented, not a bug |
  | Inputs must vary independently over a documented range | Confirmed — full factorial grid, range table added |
  | Use SHAP in addition to feature importance | Fixed — found and fixed the actual crash (mis-indexed single-output explainer), now generates successfully |
  | Figure 9: predictions constant across particle size/pressure | Fixed — real variation now (46-65% coverage, 20-44s delivery); also fixed a diffusion-unit bug in the optimization code itself that would have kept this broken even after the dataset fix |
  | Validate AI optimum against real simulation | Done — 0.4357mm (AI) vs 0.4444mm (real sim), ~2% difference |
  | Figure 10: model comparison, near-zero MAE/RMSE bars suspicious | Fixed — real, non-trivial MAE/RMSE values now |
  | Confirm no data leakage between train/test | Confirmed — `train_test_split` used correctly, no shared rows |
  | Repeat model comparison with 5-fold CV | Added — `figures/model_comparison_cv.csv` |
  | Result 7 (multi-output prediction) missing | Added — `figures/result7_multi_output_prediction.png` |

  **Additional bug found during this verification pass (not in the original list, found by
  grepping for every copy of the Stokes-Einstein formula across the codebase):**
  `src/predict.py` had the exact same missing ×1e6 diffusion-unit conversion as the
  optimization code in `train_ai.py` — fixed. Numeric impact was small for typical
  inputs (Diffusion has low feature importance in this model) but it was feeding the
  model values 1e6x outside its training distribution, which is undefined behavior for
  a tree-based model.

  **Update — main.py (Phase 1) fixed too, on request:**
  - Same ×1e6 diffusion-unit fix applied.
  - `main.py` also ran for 1200 steps, which at the new `dt=0.2` is 240s — this
    saturates the domain (depth ≈0.48-0.50mm, the full 0.5mm domain) for every
    particle size, flattening the size-dependence Result 5 is meant to show. Reduced
    to 600 steps (120s), matching the validated Phase 2 calibration.
  - Regenerated `results/AI_dataset.csv`, `grid_independence.csv`, and
    `result1-5*.png`. Verified: final PenetrationDepth now monotonic by size
    (20nm=0.419mm → 200nm=0.359mm, correct direction), grid-independence max
    velocity scales sensibly with grid resolution (50→0.00207, 100→0.00353,
    200→0.0071).
