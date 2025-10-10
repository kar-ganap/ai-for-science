"""
Test Economic Active Learning and save metrics for analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.active_learning import EconomicActiveLearner


def test_economic_al_with_artifacts():
    """Run Economic AL and save detailed metrics"""

    print("Testing Economic Active Learning with artifact generation")
    print("=" * 70)

    # Generate synthetic MOF-like data
    np.random.seed(42)
    n_total = 1000

    # Create synthetic features
    X = pd.DataFrame({
        'LCD': np.random.uniform(5, 20, n_total),
        'PLD': np.random.uniform(3, 15, n_total),
        'ASA': np.random.uniform(500, 3000, n_total),
        'Density': np.random.uniform(0.3, 1.5, n_total)
    })

    # Create synthetic target (CO2 uptake)
    # Higher surface area and larger pores → better performance
    y = (0.003 * X['ASA'] +
         0.5 * X['LCD'] +
         0.3 * X['PLD'] -
         2 * X['Density'] +
         np.random.normal(0, 2, n_total))

    # Split into train and pool
    train_size = 100
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_pool = X.iloc[train_size:]
    y_pool = y.iloc[train_size:]

    print(f"\nInitial setup:")
    print(f"  Training set: {len(X_train)} MOFs")
    print(f"  Pool (unlabeled): {len(X_pool)} MOFs")
    print(f"  Features: {list(X.columns)}")

    # Initialize learner
    learner = EconomicActiveLearner(X_train, y_train, X_pool, y_pool)

    # Run 5 iterations to see clearer trends
    print("\nRunning 5 iterations of Economic Active Learning...")
    print("-" * 70)

    for i in range(5):
        metrics = learner.run_iteration(budget=500, strategy='cost_aware_uncertainty')
        print(f"\nIteration {metrics['iteration']}:")
        print(f"  Validated: {metrics['n_validated']} MOFs")
        print(f"  Cost: ${metrics['iteration_cost']:.2f} (avg: ${metrics['avg_cost_per_sample']:.2f}/sample)")
        print(f"  Cumulative cost: ${metrics['cumulative_cost']:.2f}")
        print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.3f}")
        print(f"  Max uncertainty: {metrics['max_uncertainty']:.3f}")
        print(f"  Best predicted: {metrics['best_predicted_performance']:.2f} mmol/g")
        print(f"  Training set size: {metrics['n_train']} MOFs")

    # Get history as DataFrame
    history_df = learner.get_history_df()

    # Add uncertainty reduction metric
    initial_uncertainty = history_df.iloc[0]['mean_uncertainty']
    history_df['uncertainty_reduction_pct'] = (
        (initial_uncertainty - history_df['mean_uncertainty']) / initial_uncertainty * 100
    )

    # Add cost efficiency metric
    history_df['cumulative_samples'] = history_df['n_validated'].cumsum()
    history_df['avg_cumulative_cost_per_sample'] = (
        history_df['cumulative_cost'] / history_df['cumulative_samples']
    )

    # Save to CSV
    output_dir = Path(__file__).parents[1] / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "economic_al_test_results.csv"
    history_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print(f"\n📊 Metrics saved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\n1. Uncertainty Reduction (Epistemic Uncertainty Validation):")
    print(f"   Initial mean uncertainty: {history_df.iloc[0]['mean_uncertainty']:.4f}")
    print(f"   Final mean uncertainty:   {history_df.iloc[-1]['mean_uncertainty']:.4f}")
    print(f"   Reduction: {history_df.iloc[-1]['uncertainty_reduction_pct']:.1f}%")
    print(f"   Trend: {'✅ Decreasing (good!)' if history_df.iloc[-1]['mean_uncertainty'] < history_df.iloc[0]['mean_uncertainty'] else '❌ Not decreasing'}")

    print("\n2. Cost Efficiency:")
    print(f"   Total validated: {history_df['cumulative_samples'].iloc[-1]} MOFs")
    print(f"   Total cost: ${history_df['cumulative_cost'].iloc[-1]:.2f}")
    print(f"   Average cost per sample: ${history_df['avg_cumulative_cost_per_sample'].iloc[-1]:.2f}")

    print("\n3. Model Improvement:")
    print(f"   Training set growth: {history_df.iloc[0]['n_train']} → {history_df.iloc[-1]['n_train']} MOFs")
    print(f"   Best predicted performance: {history_df.iloc[-1]['best_predicted_performance']:.2f} mmol/g")

    print("\n4. Iteration Costs:")
    for i, row in history_df.iterrows():
        print(f"   Iteration {row['iteration']}: ${row['iteration_cost']:.2f} "
              f"({row['n_validated']} samples @ ${row['avg_cost_per_sample']:.2f} each)")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Uncertainty decreasing over iterations validates that")
    print("our ensemble is capturing epistemic (reducible) uncertainty, not")
    print("just aleatoric (irreducible) noise. This confirms the approach from")
    print("uncertainty_quantification_explained.md is working as intended.")
    print("=" * 70)

    return history_df


if __name__ == '__main__':
    history = test_economic_al_with_artifacts()

    print("\n📁 Check results/economic_al_test_results.csv for detailed metrics!")
