"""
Extract actual Economic AL baseline progression from CSV results
"""

import pandas as pd
import json
import ast
from pathlib import Path

# Load Economic AL results
project_root = Path(__file__).parent
al_results = pd.read_csv(project_root / "results/economic_al_expected_value.csv")

# Load original MOF database with actual CO2 values
mof_data = pd.read_csv(project_root / "data/processed/crafted_mofs_co2_with_costs.csv")

print("="*70)
print("EXTRACTING ECONOMIC AL BASELINE PROGRESSION")
print("="*70 + "\n")

print(f"Economic AL iterations: {len(al_results)}")
print(f"MOF database size: {len(mof_data)}\n")

# Track cumulative best
cumulative_best = 0
baseline_progression = []

for idx, row in al_results.iterrows():
    iteration = row['iteration']

    # Parse selected MOFs JSON (uses Python dict format, not JSON)
    selected_mofs_json = row['selected_mofs']
    selected_mofs = ast.literal_eval(selected_mofs_json)

    # Extract pool indices
    pool_indices = [mof['pool_index'] for mof in selected_mofs]

    # Get actual CO2 values for selected MOFs
    # Note: pool_index refers to position in the candidate pool, not mof_id
    # We need to reconstruct the pool from iteration state

    # For now, let's use a simpler approach:
    # Get the indices from the unvalidated pool and look up CO2 values

    print(f"\nIteration {iteration}:")
    print(f"  Selected {len(pool_indices)} MOFs")

    # The pool_index refers to the unvalidated pool at that iteration
    # We need to figure out which MOFs those were

    # For iterations 1-3, we can work backwards:
    # - Start with full database (687 MOFs)
    # - Iteration 1 validated some MOFs (removing them from pool)
    # - Iteration 2 sees smaller pool, etc.

    # Let's try a different approach: look at n_train and n_pool
    n_train = row['n_train']
    n_pool = row['n_pool']

    print(f"  Training set size: {n_train}")
    print(f"  Pool size: {n_pool}")

    # Actually, the best_predicted_performance might not be actual CO2
    # Let me check if there's another way to get this...

print("\n" + "="*70)
print("ISSUE IDENTIFIED")
print("="*70)
print("\nThe Economic AL CSV doesn't store actual CO2 values of selected MOFs.")
print("It only has predicted values and pool indices.")
print("\nOptions:")
print("1. Re-run Economic AL to capture actual CO2 values")
print("2. Use predicted values as proxy (not ideal)")
print("3. Reconstruct pool state and map indices (complex)")
print("\nLet's check what's in baseline_comparison.json...")

# Load baseline comparison
with open(project_root / "results/baseline_comparison.json", 'r') as f:
    baseline = json.load(f)

print("\nBaseline comparison data:")
print(f"  AL (Exploitation) best: {baseline['AL (Exploitation)']['best_performance']:.2f} mol/kg")
print(f"  AL (Exploration) best: {baseline['AL (Exploration)']['best_performance']:.2f} mol/kg")
print(f"  Expert best: {baseline['Expert']['best_performance']:.2f} mol/kg")
print(f"  Random best (mean): {baseline['Random']['best_performance_mean']:.2f} mol/kg")

print("\nThis gives us final values but not iteration-by-iteration progression.")
