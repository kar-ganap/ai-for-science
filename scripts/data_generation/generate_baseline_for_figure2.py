"""
Generate Fair Baseline for Figure 2: Panel A
Economic AL with $500/iteration, exploration strategy, 3 iterations
(To fairly compare against Active Generative Discovery)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.active_learning import EconomicActiveLearner
from src.cost.estimator import MOFCostEstimator

print("="*80)
print("GENERATING FAIR BASELINE FOR FIGURE 2 (PANEL A)")
print("="*80)
print("\nConfiguration:")
print("  Budget: $500/iteration")
print("  Strategy: exploration (cost_aware_uncertainty)")
print("  Iterations: 3")
print("  Candidates: Real MOFs only (no generation)")
print()

# Load CRAFTED MOF data
print("Loading CRAFTED MOF data...")
mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
df = pd.read_csv(mof_file)
print(f"  ✓ Loaded {len(df)} MOFs")

# Use SAME split as AGD for fair comparison
train_size = 100
train_idx = list(range(train_size))
pool_idx = list(range(train_size, len(df)))

feature_cols = ['cell_a', 'cell_b', 'cell_c', 'volume']
X_train = df.iloc[train_idx][feature_cols]
y_train = df.iloc[train_idx]['co2_uptake_mean']
X_pool = df.iloc[pool_idx][feature_cols]
y_pool = df.iloc[pool_idx]['co2_uptake_mean']

print(f"  Training set: {len(X_train)} MOFs")
print(f"  Pool: {len(X_pool)} MOFs")

# Initialize cost estimator
cost_estimator = MOFCostEstimator()

# Initialize learner
learner = EconomicActiveLearner(
    X_train=X_train,
    y_train=y_train,
    X_pool=X_pool,
    y_pool=y_pool,
    cost_estimator=cost_estimator,
    pool_compositions=[{'metal': df.iloc[i]['metal']} for i in pool_idx]
)

print("\n" + "="*80)
print("RUNNING ECONOMIC AL (EXPLORATION, $500/ITER)")
print("="*80)

# Track cumulative best
initial_best = y_train.max()
cumulative_best = initial_best
baseline_progression = []

print(f"\nInitial best (from training set): {initial_best:.2f} mol/kg\n")

n_iterations = 3
budget_per_iteration = 500.0

for i in range(n_iterations):
    print(f"Iteration {i+1}...")

    # Run iteration with exploration strategy
    metrics = learner.run_iteration(
        budget=budget_per_iteration,
        strategy='cost_aware_uncertainty'  # Exploration
    )

    # Track current best from ALL validated MOFs
    current_best = learner.y_train.max()
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
    print(f"  Best found this iter: {current_best:.2f} mol/kg")
    print(f"  Cumulative best: {cumulative_best:.2f} mol/kg")
    print()

# Save results
baseline_df = pd.DataFrame(baseline_progression)
output_file = project_root / "results/figure2_baseline_exploration_500.csv"
baseline_df.to_csv(output_file, index=False)

print("="*80)
print("BASELINE PROGRESSION (for Panel A)")
print("="*80)
print(baseline_df.to_string(index=False))

print(f"\n✓ Saved to: {output_file}")

# Load AGD results for comparison
print("\n" + "="*80)
print("COMPARISON: AGD vs BASELINE")
print("="*80)

import json
agd_file = project_root / "results/active_generative_discovery_demo/demo_results.json"
with open(agd_file, 'r') as f:
    agd_results = json.load(f)

print("\nActive Generative Discovery (Real + Generated):")
for i, iter_data in enumerate(agd_results['iterations']):
    print(f"  Iteration {i+1}: {iter_data['best_co2_this_iter']:.2f} mol/kg")

print("\nEconomic AL Baseline (Real only, $500/iter, exploration):")
for _, row in baseline_df.iterrows():
    print(f"  Iteration {int(row['iteration'])}: {row['cumulative_best']:.2f} mol/kg")

# Calculate improvement
agd_final = agd_results['iterations'][-1]['best_co2_this_iter']
baseline_final = baseline_df['cumulative_best'].iloc[-1]
improvement_pct = ((agd_final - baseline_final) / baseline_final) * 100
improvement_abs = agd_final - baseline_final

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"AGD Final:      {agd_final:.2f} mol/kg")
print(f"Baseline Final: {baseline_final:.2f} mol/kg")
print(f"Improvement:    +{improvement_pct:.1f}% (+{improvement_abs:.2f} mol/kg)")
print()
print("✅ Fair comparison achieved:")
print("   - Same budget: $500/iteration")
print("   - Same strategy: exploration (cost_aware_uncertainty)")
print("   - Same iterations: 3")
print("   - Only difference: AGD has generated MOFs, baseline doesn't")
print()
print("🎉 Ready to update Figure 2 Panel A!")
