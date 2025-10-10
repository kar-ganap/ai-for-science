"""
Baseline Comparisons for Economic Active Learning

Compares Economic AL against:
1. Random sampling (uninformed baseline)
2. Expert intuition (pre-2019 domain knowledge)
3. Economic AL (our method)

All with same budget constraint.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


def random_baseline(mof_data: pd.DataFrame, budget: float = 150.0,
                   n_trials: int = 20, return_all: bool = False) -> pd.DataFrame:
    """
    Random sampling baseline - uninformed selection (averaged over multiple trials)

    Args:
        mof_data: Full MOF dataset with synthesis_cost
        budget: Available budget
        n_trials: Number of random trials to average over
        return_all: If True, return all trials; if False, return best trial

    Returns:
        DataFrame of randomly selected MOFs within budget
    """
    all_trials = []

    for seed in range(n_trials):
        # Shuffle with different seed each time
        shuffled = mof_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        selected_indices = []
        cumulative_cost = 0.0

        for idx, row in shuffled.iterrows():
            cost = row['synthesis_cost']
            if cumulative_cost + cost <= budget:
                selected_indices.append(idx)
                cumulative_cost += cost

        selected = shuffled.loc[selected_indices].copy()
        selected['selection_method'] = f'Random (seed={seed})'
        selected['cumulative_cost'] = cumulative_cost
        selected['trial'] = seed

        all_trials.append(selected)

    if return_all:
        return all_trials

    # Return the median performing trial as representative
    trial_perfs = [trial['co2_uptake_mean'].max() for trial in all_trials]
    median_idx = np.argsort(trial_perfs)[len(trial_perfs) // 2]
    best_trial = all_trials[median_idx]
    best_trial['selection_method'] = 'Random'
    return best_trial


def expert_baseline(mof_data: pd.DataFrame, budget: float = 150.0) -> pd.DataFrame:
    """
    Expert intuition baseline - pre-2019 domain knowledge

    Based on heuristics from literature BEFORE CRAFTED (2019):
    1. Larger pore volume → better CO2 uptake
    2. Preferred metals: Zn, Cu, Mg (known good CO2 affinity)
    3. Cost-conscious (don't overpay)

    NOTE: Does NOT use CO2 uptake labels (honest baseline)

    Args:
        mof_data: Full MOF dataset
        budget: Available budget

    Returns:
        DataFrame of expert-selected MOFs within budget
    """
    df = mof_data.copy()

    # Compute expert score based on observable features only
    scores = np.zeros(len(df))

    # Heuristic 1: Larger volume → better (40% weight)
    # Normalize volume to [0, 1] range
    volume_norm = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
    scores += 0.4 * volume_norm

    # Heuristic 2: Metal preference (30% weight)
    # Based on pre-2019 literature for CO2 capture
    metal_scores = {
        'Zn': 0.30,   # Well-known, good CO2 affinity
        'Cu': 0.30,   # Excellent CO2 binding
        'Mg': 0.30,   # Strong Lewis acid sites
        'Cr': 0.20,   # Good but toxic/less preferred
        'Al': 0.10,   # Mixed results
        'Fe': 0.10,   # Mixed results
        'Ca': 0.05,   # Less studied
        'Ti': 0.05,   # Less common
        'Unknown': 0.00
    }

    for i, metal in enumerate(df['metal']):
        scores[i] += metal_scores.get(metal, 0.0)

    # Heuristic 3: Cost efficiency (30% weight)
    # Prefer cheaper MOFs (more bang for buck)
    cost_norm = 1.0 - (df['synthesis_cost'] - df['synthesis_cost'].min()) / \
                (df['synthesis_cost'].max() - df['synthesis_cost'].min())
    scores += 0.3 * cost_norm

    # Sort by score (descending) and greedily select within budget
    df['expert_score'] = scores
    df_sorted = df.sort_values('expert_score', ascending=False).reset_index(drop=True)

    selected_indices = []
    cumulative_cost = 0.0

    for idx, row in df_sorted.iterrows():
        cost = row['synthesis_cost']
        if cumulative_cost + cost <= budget:
            selected_indices.append(idx)
            cumulative_cost += cost

    selected = df_sorted.loc[selected_indices].copy()
    selected['selection_method'] = 'Expert'
    selected['cumulative_cost'] = cumulative_cost

    return selected


def economic_al_results(mof_data: pd.DataFrame,
                        pool_uncertainty_file: Path,
                        history_file: Path,
                        budget: float = 150.0) -> pd.DataFrame:
    """
    Load actual Economic AL results from history

    Args:
        mof_data: Full MOF dataset
        pool_uncertainty_file: Path to initial pool uncertainties
        history_file: Path to AL history CSV with selected_mofs data
        budget: Budget used (for consistency)

    Returns:
        DataFrame of AL-selected MOFs (from actual run)
    """
    import ast

    # Load initial pool with original indices
    pool_df = pd.read_csv(pool_uncertainty_file)

    # Load AL history
    history_df = pd.read_csv(history_file)

    # Extract actually selected MOF IDs across all iterations
    selected_mof_ids = []
    current_pool = pool_df.copy()

    for _, row in history_df.iterrows():
        # Parse selected_mofs (Python dict representation)
        selected_mofs = ast.literal_eval(row['selected_mofs'])

        # Get pool indices for this iteration
        pool_indices = [mof['pool_index'] for mof in selected_mofs]

        # Get the MOF IDs at these positions in current pool
        selected_this_iter = current_pool.iloc[pool_indices]
        selected_mof_ids.extend(selected_this_iter['mof_id'].tolist())

        # Remove selected MOFs from pool for next iteration
        current_pool = current_pool.drop(current_pool.index[pool_indices]).reset_index(drop=True)

    # Get the selected MOFs from main dataset
    selected = mof_data[mof_data['mof_id'].isin(selected_mof_ids)].copy()
    selected['selection_method'] = 'Economic AL'
    selected['cumulative_cost'] = selected['synthesis_cost'].sum()

    return selected


def compare_baselines(mof_data: pd.DataFrame,
                      pool_uncertainty_file: Path,
                      history_file: Path,
                      budget: float = 150.0,
                      n_random_trials: int = 20,
                      expected_value_history_file: Optional[Path] = None,
                      expected_value_pool_file: Optional[Path] = None) -> dict:
    """
    Run all baselines and compare (including both AL strategies)

    Args:
        mof_data: Full MOF dataset
        pool_uncertainty_file: Path to pool uncertainties (exploration AL)
        history_file: Path to AL history with selected_mofs data (exploration AL)
        budget: Budget constraint
        n_random_trials: Number of random trials to average over
        expected_value_history_file: Path to expected_value AL history (optional)
        expected_value_pool_file: Path to expected_value pool uncertainties (optional)

    Returns:
        Dictionary with results for each method
    """
    print(f"\n{'='*80}")
    print(f"4-WAY BASELINE COMPARISON: Budget = ${budget:.0f}")
    print(f"{'='*80}\n")

    # Run baselines
    print(f"Running Random baseline ({n_random_trials} trials)...")
    random_trials = random_baseline(mof_data, budget, n_trials=n_random_trials, return_all=True)

    print("Running Expert baseline...")
    expert_mofs = expert_baseline(mof_data, budget)

    print("Loading Economic AL (Exploration) results...")
    al_exploration_mofs = economic_al_results(mof_data, pool_uncertainty_file, history_file, budget)

    # Load expected_value AL if provided
    al_exploitation_mofs = None
    if expected_value_history_file and expected_value_pool_file:
        print("Loading Economic AL (Exploitation) results...")
        al_exploitation_mofs = economic_al_results(mof_data, expected_value_pool_file, expected_value_history_file, budget)

    # Compute statistics
    results = {}

    # Random statistics (aggregate across trials)
    random_best_perfs = [trial['co2_uptake_mean'].max() for trial in random_trials]
    random_mean_perfs = [trial['co2_uptake_mean'].mean() for trial in random_trials]
    random_n_selected = [len(trial) for trial in random_trials]

    results['Random'] = {
        'n_trials': n_random_trials,
        'best_performance_mean': np.mean(random_best_perfs),
        'best_performance_std': np.std(random_best_perfs),
        'best_performance_min': np.min(random_best_perfs),
        'best_performance_max': np.max(random_best_perfs),
        'mean_performance_avg': np.mean(random_mean_perfs),
        'n_selected_avg': np.mean(random_n_selected),
        'all_trials': random_trials  # Keep for plotting
    }

    print(f"\nRandom (n={n_random_trials} trials):")
    print(f"  MOFs selected:     {results['Random']['n_selected_avg']:.0f} (average)")
    print(f"  Best performance:  {results['Random']['best_performance_mean']:.2f} ± {results['Random']['best_performance_std']:.2f} mol/kg")
    print(f"                     (range: {results['Random']['best_performance_min']:.2f} - {results['Random']['best_performance_max']:.2f})")
    print(f"  Mean performance:  {results['Random']['mean_performance_avg']:.2f} mol/kg")
    print()

    # Expert and AL statistics (single run each)
    methods_to_process = [
        ('Expert', expert_mofs),
        ('AL (Exploration)', al_exploration_mofs)
    ]

    if al_exploitation_mofs is not None:
        methods_to_process.append(('AL (Exploitation)', al_exploitation_mofs))

    for method, selected in methods_to_process:

        n_selected = len(selected)
        total_cost = selected['cumulative_cost'].iloc[0]
        avg_cost = total_cost / n_selected

        # Performance metrics
        best_co2 = selected['co2_uptake_mean'].max()
        mean_co2 = selected['co2_uptake_mean'].mean()
        median_co2 = selected['co2_uptake_mean'].median()

        # Pareto optimal count (rough estimate)
        # MOF is Pareto if no other MOF has lower cost AND higher performance
        is_pareto = np.ones(len(selected), dtype=bool)
        costs = selected['synthesis_cost'].values
        performance = selected['co2_uptake_mean'].values

        for i in range(len(selected)):
            is_dominated = np.any((costs < costs[i]) & (performance > performance[i]))
            is_pareto[i] = not is_dominated

        n_pareto = is_pareto.sum()

        results[method] = {
            'selected_mofs': selected,
            'n_selected': n_selected,
            'total_cost': total_cost,
            'avg_cost_per_mof': avg_cost,
            'best_performance': best_co2,
            'mean_performance': mean_co2,
            'median_performance': median_co2,
            'n_pareto_optimal': n_pareto,
            'pareto_fraction': n_pareto / n_selected
        }

        print(f"{method}:")
        print(f"  MOFs selected:     {n_selected}")
        print(f"  Total cost:        ${total_cost:.2f}")
        print(f"  Avg cost/MOF:      ${avg_cost:.2f}")
        print(f"  Best performance:  {best_co2:.2f} mol/kg")
        print(f"  Mean performance:  {mean_co2:.2f} mol/kg")
        print(f"  Pareto-optimal:    {n_pareto} ({n_pareto/n_selected*100:.1f}%)")
        print()

    # Statistical comparison
    random_mean = results['Random']['best_performance_mean']
    random_std = results['Random']['best_performance_std']
    expert_best = results['Expert']['best_performance']
    al_exploration_best = results['AL (Exploration)']['best_performance']

    print(f"{'='*80}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*80}")
    print(f"Random (avg):          {random_mean:.2f} ± {random_std:.2f} mol/kg")
    print(f"Expert:                {expert_best:.2f} mol/kg")
    print(f"AL (Exploration):      {al_exploration_best:.2f} mol/kg")

    if 'AL (Exploitation)' in results:
        al_exploitation_best = results['AL (Exploitation)']['best_performance']
        print(f"AL (Exploitation):     {al_exploitation_best:.2f} mol/kg")

    print()

    # Z-score for AL vs Random distribution
    z_score_exp = (al_exploration_best - random_mean) / random_std if random_std > 0 else 0
    print(f"AL (Exploration) vs Random avg: {(al_exploration_best - random_mean)/random_mean*100:+.1f}%")
    print(f"Z-score:                         {z_score_exp:+.2f}σ")

    if z_score_exp > 2:
        print(f"  → Significantly BETTER than Random (p < 0.05)")
    elif z_score_exp < -2:
        print(f"  → Significantly WORSE than Random (p < 0.05)")
    else:
        print(f"  → Statistically similar to Random")

    print()

    if 'AL (Exploitation)' in results:
        z_score_expl = (al_exploitation_best - random_mean) / random_std if random_std > 0 else 0
        print(f"AL (Exploitation) vs Random avg: {(al_exploitation_best - random_mean)/random_mean*100:+.1f}%")
        print(f"Z-score:                          {z_score_expl:+.2f}σ")

        if z_score_expl > 2:
            print(f"  → Significantly BETTER than Random (p < 0.05)")
        elif z_score_expl < -2:
            print(f"  → Significantly WORSE than Random (p < 0.05)")
        else:
            print(f"  → Statistically similar to Random")
        print()

    print(f"AL (Exploration) vs Expert:      {(al_exploration_best - expert_best)/expert_best*100:+.1f}%")
    if 'AL (Exploitation)' in results:
        print(f"AL (Exploitation) vs Expert:     {(al_exploitation_best - expert_best)/expert_best*100:+.1f}%")
    print(f"Expert vs Random avg:            {(expert_best - random_mean)/random_mean*100:+.1f}%")
    print()

    return results


if __name__ == '__main__':
    print("Testing Baseline Comparisons\n" + "="*60)

    # Load data
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"
    pool_file = project_root / "results" / "pool_uncertainties_initial.csv"
    history_file = project_root / "results" / "economic_al_crafted_integration.csv"

    if not mof_file.exists() or not pool_file.exists() or not history_file.exists():
        print(f"❌ Missing data files. Run:")
        print(f"   uv run python tests/test_economic_al_crafted.py")
        exit(1)

    mof_data = pd.read_csv(mof_file)

    # Check for expected_value files
    expected_value_history_file = project_root / "results" / "economic_al_expected_value.csv"
    expected_value_pool_file = project_root / "results" / "pool_uncertainties_expected_value.csv"

    # Run comparison (4-way if expected_value exists, else 3-way)
    results = compare_baselines(
        mof_data, pool_file, history_file,
        budget=150.0,
        n_random_trials=20,
        expected_value_history_file=expected_value_history_file if expected_value_history_file.exists() else None,
        expected_value_pool_file=expected_value_pool_file if expected_value_pool_file.exists() else None
    )

    # Save results
    output_file = project_root / "results" / "baseline_comparison.json"

    # Convert to JSON-serializable format
    import json
    json_results = {}

    # Random has different structure (aggregated stats)
    if 'Random' in results:
        json_results['Random'] = {
            'n_trials': int(results['Random']['n_trials']),
            'n_selected_avg': float(results['Random']['n_selected_avg']),
            'best_performance_mean': float(results['Random']['best_performance_mean']),
            'best_performance_std': float(results['Random']['best_performance_std']),
            'best_performance_min': float(results['Random']['best_performance_min']),
            'best_performance_max': float(results['Random']['best_performance_max']),
            'mean_performance_avg': float(results['Random']['mean_performance_avg'])
        }

    # Expert and AL have single-run structure
    for method in ['Expert', 'AL (Exploration)', 'AL (Exploitation)']:
        if method in results:
            res = results[method]
            json_results[method] = {
                'n_selected': int(res['n_selected']),
                'total_cost': float(res['total_cost']),
                'avg_cost_per_mof': float(res['avg_cost_per_mof']),
                'best_performance': float(res['best_performance']),
                'mean_performance': float(res['mean_performance']),
                'median_performance': float(res['median_performance']),
                'n_pareto_optimal': int(res['n_pareto_optimal']),
                'pareto_fraction': float(res['pareto_fraction'])
            }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"📊 Results saved to: {output_file}")
