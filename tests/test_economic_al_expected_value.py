"""
Test Economic AL with Expected Value Strategy
(Exploitation-focused: predicted_value × uncertainty / cost)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.active_learning import EconomicActiveLearner
from src.cost.estimator import MOFCostEstimator

def test_economic_al_expected_value():
    """Run Economic AL with expected_value strategy"""

    print("="*80)
    print("ECONOMIC ACTIVE LEARNING - EXPECTED VALUE STRATEGY")
    print("="*80)

    # Load CRAFTED MOF data
    print("\n[1/6] Loading CRAFTED MOF data...")
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"

    if not mof_file.exists():
        print(f"❌ Missing {mof_file}")
        print(f"   Run: uv run python tests/test_economic_al_crafted.py first")
        return None

    df = pd.read_csv(mof_file)
    print(f"  ✓ Loaded {len(df)} MOFs")
    print(f"  ✓ CO2 uptake range: {df['co2_uptake_mean'].min():.2f} - {df['co2_uptake_mean'].max():.2f} mol/kg")

    # Initialize cost estimator
    print("\n[2/6] Initializing cost estimator...")
    cost_estimator = MOFCostEstimator()
    print(f"  ✓ Cost range: ${df['synthesis_cost'].min():.2f} - ${df['synthesis_cost'].max():.2f} per gram")

    # Prepare data
    print("\n[3/6] Preparing train/pool split...")
    train_size = 100

    # Use same split as exploration run for fair comparison
    train_idx = list(range(train_size))
    pool_idx = list(range(train_size, len(df)))

    # Features
    feature_cols = ['cell_a', 'cell_b', 'cell_c', 'volume']
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['co2_uptake_mean']
    X_pool = df.iloc[pool_idx][feature_cols]
    y_pool = df.iloc[pool_idx]['co2_uptake_mean']

    print(f"  ✓ Training set: {len(X_train)} MOFs")
    print(f"  ✓ Pool (unlabeled): {len(X_pool)} MOFs")

    # Initialize learner
    print("\n[4/6] Initializing Economic Active Learner...")
    learner = EconomicActiveLearner(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        cost_estimator=cost_estimator,
        pool_compositions=[{'metal': df.iloc[i]['metal']} for i in pool_idx]
    )
    print(f"  ✓ Learner initialized")

    # Compute initial pool uncertainties
    print(f"\n  Computing initial pool uncertainties...")
    pool_mean, pool_std, pool_costs = learner.compute_pool_uncertainties()

    pool_uncertainty_data = []
    for i in range(len(learner.X_pool)):
        original_idx = pool_idx[i]
        pool_uncertainty_data.append({
            'mof_id': df.iloc[original_idx]['mof_id'],
            'original_index': original_idx,
            'co2_uptake_mean': df.iloc[original_idx]['co2_uptake_mean'],
            'synthesis_cost': df.iloc[original_idx]['synthesis_cost'],
            'metal': df.iloc[original_idx]['metal'],
            'uncertainty': pool_std[i],
            'predicted_performance': pool_mean[i],
            'validation_cost': pool_costs[i]
        })

    pool_uncertainty_df = pd.DataFrame(pool_uncertainty_data)
    pool_uncertainty_file = project_root / "results" / "pool_uncertainties_expected_value.csv"
    pool_uncertainty_df.to_csv(pool_uncertainty_file, index=False)
    print(f"  ✓ Saved {len(pool_uncertainty_df)} pool MOF uncertainties")

    # Run 5 iterations
    print("\n[5/6] Running Economic Active Learning (Expected Value)...")
    print("-"*80)

    n_iterations = 5
    budget_per_iteration = 50.0
    history = []

    for i in range(n_iterations):
        metrics = learner.run_iteration(
            budget=budget_per_iteration,
            strategy='expected_value'  # 🔥 KEY CHANGE
        )
        history.append(metrics)

        print(f"\n  Iteration {metrics['iteration']}:")
        print(f"    Validated:    {metrics['n_validated']} MOFs")
        print(f"    Cost:         ${metrics['iteration_cost']:.2f} (avg: ${metrics['avg_cost_per_sample']:.2f}/MOF)")
        print(f"    Cumulative:   ${metrics['cumulative_cost']:.2f}")
        print(f"    Uncertainty:  {metrics['mean_uncertainty']:.3f} (max: {metrics['max_uncertainty']:.3f})")
        print(f"    Best MOF:     {metrics['best_predicted_performance']:.2f} mol/kg CO2")
        print(f"    Training set: {metrics['n_train']} MOFs")

    # Save history
    history_df = learner.get_history_df()
    history_file = project_root / "results" / "economic_al_expected_value.csv"
    history_df.to_csv(history_file, index=False)

    print("\n" + "-"*80)
    print("[6/6] Validating results...")
    print("="*80)

    # Compute metrics
    initial_unc = history[0]['mean_uncertainty']
    final_unc = history[-1]['mean_uncertainty']
    unc_reduction = (initial_unc - final_unc) / initial_unc * 100

    print(f"\n✓ Uncertainty Reduction:")
    print(f"    Initial: {initial_unc:.3f}")
    print(f"    Final:   {final_unc:.3f}")
    print(f"    Change:  {unc_reduction:.1f}%")

    print(f"\n✓ Budget Compliance:")
    for i, metrics in enumerate(history, 1):
        status = "✅" if metrics['iteration_cost'] <= budget_per_iteration else "❌"
        print(f"    Iteration {i}: ${metrics['iteration_cost']:.2f} ≤ ${budget_per_iteration} {status}")

    print(f"\n✓ Active Learning Progress:")
    print(f"    Initial training: {train_size} MOFs")
    print(f"    Final training:   {history[-1]['n_train']} MOFs")
    print(f"    Validated:        {history[-1]['n_train'] - train_size} MOFs")
    print(f"    Total cost:       ${learner.cumulative_cost:.2f}")

    print(f"\n✓ MOF Performance Discovery:")
    print(f"    Best predicted:   {history[-1]['best_predicted_performance']:.2f} mol/kg")
    print(f"    Mean predicted:   {history[-1]['mean_predicted_performance']:.2f} mol/kg")

    print("\n" + "="*80)
    print("EXPECTED VALUE AL COMPLETE!")
    print("="*80)
    print(f"\n📊 Results saved to: {history_file}")

    return history


if __name__ == '__main__':
    test_economic_al_expected_value()
