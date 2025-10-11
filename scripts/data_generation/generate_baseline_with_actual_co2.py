"""
Quick run of Economic AL to extract ACTUAL best CO2 per iteration
(Not predictions, but actual values from the database)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.active_learning import EconomicActiveLearner
from src.cost.estimator import MOFCostEstimator

print("="*70)
print("GENERATING ECONOMIC AL BASELINE WITH ACTUAL CO2 VALUES")
print("="*70)

# Load data
mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
df = pd.read_csv(mof_file)

print(f"\nLoaded {len(df)} MOFs")

# Prepare split (same as original)
train_size = 100
train_idx = list(range(train_size))
pool_idx = list(range(train_size, len(df)))

feature_cols = ['cell_a', 'cell_b', 'cell_c', 'volume']
X_train = df.iloc[train_idx][feature_cols]
y_train = df.iloc[train_idx]['co2_uptake_mean']
X_pool = df.iloc[pool_idx][feature_cols]
y_pool = df.iloc[pool_idx]['co2_uptake_mean']

# Initialize
cost_estimator = MOFCostEstimator()
learner = EconomicActiveLearner(
    X_train=X_train,
    y_train=y_train,
    X_pool=X_pool,
    y_pool=y_pool,
    cost_estimator=cost_estimator,
    pool_compositions=[{'metal': df.iloc[i]['metal']} for i in pool_idx]
)

print(f"Initial training: {len(X_train)} MOFs")
print(f"Initial pool: {len(X_pool)} MOFs")

# Track actual best CO2 across iterations
baseline_progression = []

# Initial best (from training set)
initial_best = y_train.max()
cumulative_best = initial_best

print(f"\nInitial best (from training set): {initial_best:.2f} mol/kg\n")

# Run 3 iterations
n_iterations = 3
budget_per_iteration = 50.0

for i in range(n_iterations):
    print(f"Iteration {i+1}...")

    # Run iteration
    metrics = learner.run_iteration(
        budget=budget_per_iteration,
        strategy='expected_value'
    )

    # Get the actual CO2 values of what was just validated
    # These are now in the training set
    # The learner adds selected MOFs to y_train

    # Track current best from ALL validated MOFs so far
    current_best = learner.y_train.max()

    # Update cumulative best
    cumulative_best = max(cumulative_best, current_best)

    baseline_progression.append({
        'iteration': i + 1,
        'n_validated': metrics['n_validated'],
        'cost': metrics['iteration_cost'],
        'best_this_iter': current_best,
        'cumulative_best': cumulative_best
    })

    print(f"  Validated: {metrics['n_validated']} MOFs")
    print(f"  Cost: ${metrics['iteration_cost']:.2f}")
    print(f"  Best in training set now: {current_best:.2f} mol/kg")
    print(f"  Cumulative best: {cumulative_best:.2f} mol/kg\n")

# Save results
baseline_df = pd.DataFrame(baseline_progression)
output_file = project_root / "results/economic_al_baseline_actual_co2.csv"
baseline_df.to_csv(output_file, index=False)

print("="*70)
print("BASELINE PROGRESSION")
print("="*70)
print(baseline_df.to_string(index=False))

print(f"\n✓ Saved to: {output_file}")

print("\n" + "="*70)
print("COMPARISON DATA READY FOR FIGURE")
print("="*70)
print("\nEconomic AL (Exploitation baseline):")
for _, row in baseline_df.iterrows():
    print(f"  Iteration {int(row['iteration'])}: {row['cumulative_best']:.2f} mol/kg")

print("\nActive Generative Discovery:")
print("  Iteration 1: 9.03 mol/kg")
print("  Iteration 2: 10.43 mol/kg")
print("  Iteration 3: 11.07 mol/kg")

print("\n🎉 Ready to update Panel A!")
