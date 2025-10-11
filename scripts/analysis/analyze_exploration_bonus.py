"""
Principled Justification for Exploration Bonus Value

Analyzes acquisition score distributions to determine the bonus needed
to make generated MOFs competitive with real MOFs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(".") / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator


def featurize_with_geom(mof_dict, geom_dict=None):
    """Featurize MOF with geometric descriptors"""
    metals = ['Zn', 'Fe', 'Ca', 'Al', 'Ti', 'Cu', 'Zr', 'Cr', 'Unknown']
    linkers = ['terephthalic acid', 'trimesic acid',
               '2,6-naphthalenedicarboxylic acid',
               'biphenyl-4,4-dicarboxylic acid']

    features = []
    metal = mof_dict.get('metal', 'Unknown')
    for m in metals:
        features.append(1.0 if metal == m else 0.0)

    linker = mof_dict.get('linker', 'terephthalic acid')
    for l in linkers:
        features.append(1.0 if linker == l else 0.0)

    features.extend([
        mof_dict.get('cell_a', 10.0),
        mof_dict.get('cell_b', 10.0),
        mof_dict.get('cell_c', 10.0),
        mof_dict.get('volume', 1000.0),
        mof_dict.get('synthesis_cost', 0.8)
    ])

    features.extend([1.2, 18.0, 0.08, 0.92, 2.0])
    return np.array(features)


def analyze_acquisition_distributions():
    """
    Analyze acquisition score distributions to determine needed bonus
    """
    print("="*70)
    print("EMPIRICAL CALIBRATION OF EXPLORATION BONUS")
    print("="*70 + "\n")

    project_root = Path(".")

    # Load data
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    linker_file = project_root / "data/processed/crafted_mofs_linkers.csv"

    mof_data = pd.read_csv(mof_file)
    linker_data = pd.read_csv(linker_file)
    mof_data = mof_data.merge(linker_data[['mof_id', 'linker']], on='mof_id', how='left')

    # Simulate iteration 1 state: 30 validated, rest unvalidated
    np.random.seed(42)
    validated_idx = np.random.choice(len(mof_data), 30, replace=False)
    validated = mof_data.iloc[validated_idx]
    unvalidated = mof_data.drop(validated_idx)

    print(f"Validated: {len(validated)} MOFs")
    print(f"Unvalidated: {len(unvalidated)} MOFs\n")

    # Train GP surrogate
    print("Training GP surrogate...")
    X_train = np.array([featurize_with_geom(row.to_dict()) for _, row in validated.iterrows()])
    y_train = validated['co2_uptake_mean'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X_train_scaled, y_train)
    print("✓ GP trained\n")

    # Generate MOFs
    print("Generating MOFs...")
    vae_model = project_root / "models/dual_conditional_mof_vae_compositional.pt"
    vae = DualConditionalMOFGenerator(use_geom_features=False)
    vae.load(vae_model)

    target_co2 = np.percentile(validated['co2_uptake_mean'], 90)
    generated = vae.generate_candidates(target_co2=target_co2, target_cost=0.78, n_candidates=100)
    print(f"✓ Generated {len(generated)} MOFs\n")

    # Compute acquisition scores
    print("="*70)
    print("ACQUISITION SCORE ANALYSIS")
    print("="*70 + "\n")

    # Cost mapping
    metal_costs = {
        'Zn': 35, 'Fe': 38, 'Ca': 42, 'Al': 40,
        'Ti': 58, 'Cu': 45, 'Zr': 65, 'Cr': 55
    }

    # Real MOFs
    real_acquisitions = []
    for _, mof in unvalidated.iterrows():
        X = featurize_with_geom(mof.to_dict()).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred, std = gp.predict(X_scaled, return_std=True)

        ei = pred[0] + 1.96 * std[0]
        cost = metal_costs.get(mof['metal'], 40)
        acquisition = ei / cost
        real_acquisitions.append(acquisition)

    # Generated MOFs
    gen_acquisitions = []
    for mof in generated:
        X = featurize_with_geom(mof).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred, std = gp.predict(X_scaled, return_std=True)

        ei = pred[0] + 1.96 * std[0]
        cost = metal_costs.get(mof['metal'], 40)
        acquisition = ei / cost
        gen_acquisitions.append(acquisition)

    real_acquisitions = np.array(real_acquisitions)
    gen_acquisitions = np.array(gen_acquisitions)

    # Statistics
    print("Real MOFs:")
    print(f"  Count: {len(real_acquisitions)}")
    print(f"  Mean acquisition: {real_acquisitions.mean():.4f}")
    print(f"  Std: {real_acquisitions.std():.4f}")
    print(f"  Median: {np.median(real_acquisitions):.4f}")
    print(f"  90th percentile: {np.percentile(real_acquisitions, 90):.4f}")
    print(f"  95th percentile: {np.percentile(real_acquisitions, 95):.4f}")
    print(f"  Max: {real_acquisitions.max():.4f}\n")

    print("Generated MOFs:")
    print(f"  Count: {len(gen_acquisitions)}")
    print(f"  Mean acquisition: {gen_acquisitions.mean():.4f}")
    print(f"  Std: {gen_acquisitions.std():.4f}")
    print(f"  Median: {np.median(gen_acquisitions):.4f}")
    print(f"  90th percentile: {np.percentile(gen_acquisitions, 90):.4f}")
    print(f"  95th percentile: {np.percentile(gen_acquisitions, 95):.4f}")
    print(f"  Max: {gen_acquisitions.max():.4f}\n")

    # Derive needed bonus
    print("="*70)
    print("DERIVING EXPLORATION BONUS")
    print("="*70 + "\n")

    # Method 1: Make median generated > 95th percentile real
    real_95th = np.percentile(real_acquisitions, 95)
    gen_median = np.median(gen_acquisitions)
    bonus_method1 = real_95th - gen_median

    print("Method 1: Median Generated > 95th Percentile Real")
    print(f"  Real 95th percentile: {real_95th:.4f}")
    print(f"  Generated median: {gen_median:.4f}")
    print(f"  Needed bonus: {bonus_method1:.4f}\n")

    # Method 2: Make worst generated > best real
    real_max = real_acquisitions.max()
    gen_min = gen_acquisitions.min()
    bonus_method2 = real_max - gen_min + 0.01  # Small margin

    print("Method 2: Worst Generated > Best Real")
    print(f"  Real max: {real_max:.4f}")
    print(f"  Generated min: {gen_min:.4f}")
    print(f"  Needed bonus: {bonus_method2:.4f}\n")

    # Method 3: Make generated mean > real mean by N std
    real_mean = real_acquisitions.mean()
    real_std = real_acquisitions.std()
    gen_mean = gen_acquisitions.mean()

    # Want: gen_mean + bonus > real_mean + N * real_std
    # For N=2 (2 standard deviations above)
    N = 2
    bonus_method3 = real_mean + N * real_std - gen_mean

    print(f"Method 3: Generated Mean > Real Mean + {N}σ")
    print(f"  Real mean: {real_mean:.4f}")
    print(f"  Real std: {real_std:.4f}")
    print(f"  Real mean + {N}σ: {real_mean + N * real_std:.4f}")
    print(f"  Generated mean: {gen_mean:.4f}")
    print(f"  Needed bonus: {bonus_method3:.4f}\n")

    # Method 4: Expected number of generated in top-k
    budget = 500
    avg_cost = 40
    k = budget // avg_cost  # ~12 selections

    real_topk = np.sort(real_acquisitions)[-k:]
    real_topk_min = real_topk.min()

    bonus_method4 = real_topk_min - gen_acquisitions.min()

    print(f"Method 4: Guarantee Generated in Top-{k}")
    print(f"  Budget: ${budget}")
    print(f"  Expected selections: {k}")
    print(f"  Real top-{k} minimum: {real_topk_min:.4f}")
    print(f"  Generated minimum: {gen_acquisitions.min():.4f}")
    print(f"  Needed bonus: {bonus_method4:.4f}\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    bonuses = {
        'Method 1 (Median > 95th %ile)': bonus_method1,
        'Method 2 (Min > Max)': bonus_method2,
        'Method 3 (Mean > Mean+2σ)': bonus_method3,
        'Method 4 (Guarantee in top-k)': bonus_method4
    }

    for method, bonus in bonuses.items():
        print(f"  {method:40s}: {bonus:.4f}")

    avg_bonus = np.mean(list(bonuses.values()))
    print(f"\n  Average across methods: {avg_bonus:.4f}")
    print(f"  Our choice (2.0): {'✓ Reasonable' if 1.5 < avg_bonus < 3.0 else '⚠ Consider adjusting'}")

    # Sensitivity analysis
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS")
    print("="*70 + "\n")

    test_bonuses = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("Effect of different bonus values on selection rate:\n")
    print("Bonus  | % Generated in Top-12")
    print("-------|----------------------")

    for bonus in test_bonuses:
        gen_with_bonus = gen_acquisitions + bonus
        combined = np.concatenate([real_acquisitions, gen_with_bonus])
        combined_sources = ['real'] * len(real_acquisitions) + ['generated'] * len(gen_acquisitions)

        top_k_indices = np.argsort(combined)[-12:]
        top_k_sources = [combined_sources[i] for i in top_k_indices]
        gen_pct = 100 * sum(1 for s in top_k_sources if s == 'generated') / 12

        print(f"{bonus:5.1f}  | {gen_pct:5.1f}%")

    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    analyze_acquisition_distributions()
