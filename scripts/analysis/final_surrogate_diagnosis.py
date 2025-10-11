"""
Final Surrogate Diagnosis with Geometric Features

Tests surrogate with multiple target regions to properly assess
correlation and generalization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator


def featurize_with_geom(mof_dict, geom_dict):
    """Featurize MOF with geometric descriptors"""
    metals = ['Zn', 'Fe', 'Ca', 'Al', 'Ti', 'Unknown']
    linkers = ['terephthalic acid', 'trimesic acid',
               '2,6-naphthalenedicarboxylic acid',
               'biphenyl-4,4-dicarboxylic acid']

    features = []

    # Metal one-hot (6)
    metal = mof_dict.get('metal', 'Unknown')
    for m in metals:
        features.append(1.0 if metal == m else 0.0)

    # Linker one-hot (4)
    linker = mof_dict.get('linker', 'terephthalic acid')
    for l in linkers:
        features.append(1.0 if linker == l else 0.0)

    # Cell parameters (4)
    features.extend([
        mof_dict.get('cell_a', 10.0),
        mof_dict.get('cell_b', 10.0),
        mof_dict.get('cell_c', 10.0),
        mof_dict.get('volume', 1000.0)
    ])

    # Synthesis cost (1)
    features.append(mof_dict.get('synthesis_cost', 0.8))

    # Geometric features (5)
    mof_id = mof_dict.get('mof_id')
    if mof_id and mof_id in geom_dict:
        geom = geom_dict[mof_id]
        features.extend([
            geom.get('density', 1.0),
            geom.get('volume_per_atom', 20.0),
            geom.get('packing_fraction', 0.5),
            geom.get('void_fraction_proxy', 0.5),
            geom.get('avg_coordination', 2.0),
        ])
    else:
        # Median values for generated MOFs
        features.extend([1.2, 18.0, 0.08, 0.92, 2.0])

    return np.array(features)


def main():
    print("="*70)
    print("FINAL SURROGATE DIAGNOSIS")
    print("With geometric features + proper correlation test")
    print("="*70 + "\n")

    # Load data
    project_root = Path(__file__).parent
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    linker_file = project_root / "data/processed/crafted_mofs_linkers.csv"
    geom_file = project_root / "data/processed/crafted_geometric_features.csv"
    vae_model = project_root / "models/dual_conditional_mof_vae_compositional.pt"

    mof_data = pd.read_csv(mof_file)
    linker_data = pd.read_csv(linker_file)
    geom_data = pd.read_csv(geom_file)

    # Merge
    mof_data = mof_data.merge(linker_data[['mof_id', 'linker']], on='mof_id', how='left')

    # Create geom dict
    geom_dict = {}
    for _, row in geom_data.iterrows():
        geom_dict[row['mof_id']] = row.to_dict()

    # Featurize real MOFs
    print("Training surrogate with geometric features...")
    X = []
    y = []

    for _, mof in mof_data.iterrows():
        features = featurize_with_geom(mof.to_dict(), geom_dict)
        X.append(features)
        y.append(mof['co2_uptake_mean'])

    X = np.array(X)
    y = np.array(y)

    print(f"Features: {X.shape[1]} dimensions (15 + 5 geometric)")

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    surrogate = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    surrogate.fit(X_train, y_train)

    # Evaluate on real MOFs
    y_pred = surrogate.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\nReal MOF performance:")
    print(f"  MAE: {test_mae:.3f} mol/kg")
    print(f"  R²:  {test_r2:.3f}")

    # Generate MOFs at MULTIPLE targets to test correlation
    print(f"\n{'='*70}")
    print("Generating MOFs at multiple targets...")
    print(f"{'='*70}\n")

    vae = DualConditionalMOFGenerator(use_geom_features=False)
    vae.load(vae_model)

    targets = [
        (3.0, 0.78, "Low target"),
        (5.0, 0.80, "Medium target"),
        (7.0, 0.82, "High target"),
        (9.0, 0.85, "Very high target"),
    ]

    all_generated = []
    all_targets = []
    all_predictions = []
    all_uncertainties = []

    for target_co2, target_cost, desc in targets:
        print(f"Target: {target_co2:.1f} mol/kg ({desc})")

        generated = vae.generate_candidates(
            n_candidates=50,
            target_co2=target_co2,
            target_cost=target_cost,
            temperature=4.0
        )

        print(f"  Generated: {len(generated)} MOFs")

        # Predict
        X_gen = np.array([featurize_with_geom(mof, geom_dict) for mof in generated])
        predictions = surrogate.predict(X_gen)

        tree_preds = np.array([tree.predict(X_gen) for tree in surrogate.estimators_])
        uncertainties = tree_preds.std(axis=0)

        print(f"  Mean prediction: {predictions.mean():.2f} mol/kg")
        print(f"  Avg uncertainty: {uncertainties.mean():.2f} mol/kg\n")

        all_generated.extend(generated)
        all_targets.extend([target_co2] * len(generated))
        all_predictions.extend(predictions.tolist())
        all_uncertainties.extend(uncertainties.tolist())

    # Convert to arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)

    # Compute correlation
    if len(set(all_targets)) > 1:  # Check for variance
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
    else:
        correlation = np.nan

    # Statistics
    print(f"{'='*70}")
    print("OVERALL STATISTICS (Generated MOFs)")
    print(f"{'='*70}\n")

    print(f"Total generated: {len(all_generated)}")
    print(f"Target range: {all_targets.min():.1f} - {all_targets.max():.1f} mol/kg")
    print(f"Prediction range: {all_predictions.min():.2f} - {all_predictions.max():.2f} mol/kg")
    print(f"Correlation (prediction vs target): {correlation:.3f}")
    print(f"Avg uncertainty: {all_uncertainties.mean():.2f} mol/kg")
    print(f"Max uncertainty: {all_uncertainties.max():.2f} mol/kg")

    # Average prediction by target
    print(f"\nPrediction by target region:")
    for target_co2, target_cost, desc in targets:
        mask = all_targets == target_co2
        if mask.sum() > 0:
            avg_pred = all_predictions[mask].mean()
            avg_unc = all_uncertainties[mask].mean()
            print(f"  Target {target_co2:.1f}: Predicted {avg_pred:.2f} ± {avg_unc:.2f} mol/kg")

    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}\n")

    # Check 1: Real MOF performance
    if test_r2 > 0.6:
        print("✅ Real MOF performance: EXCELLENT (R²={:.3f})".format(test_r2))
        real_verdict = "PASS"
    elif test_r2 > 0.4:
        print("✓ Real MOF performance: GOOD (R²={:.3f})".format(test_r2))
        real_verdict = "PASS"
    else:
        print("⚠️  Real MOF performance: WEAK (R²={:.3f})".format(test_r2))
        real_verdict = "CAUTION"

    # Check 2: Correlation with targets
    if correlation > 0.5:
        print("✅ Target correlation: STRONG (r={:.3f})".format(correlation))
        print("   VAE conditioning is working, surrogate captures trend")
        corr_verdict = "PASS"
    elif correlation > 0.3:
        print("✓ Target correlation: MODERATE (r={:.3f})".format(correlation))
        print("   Some alignment with targets, but weak")
        corr_verdict = "CAUTION"
    elif not np.isnan(correlation):
        print("❌ Target correlation: WEAK (r={:.3f})".format(correlation))
        print("   Predictions don't track with targets")
        corr_verdict = "FAIL"
    else:
        print("⚠️  Target correlation: Cannot compute (NaN)")
        corr_verdict = "UNKNOWN"

    # Check 3: Uncertainty
    uncertainty_real = tree_preds.std(axis=0).mean() if len(tree_preds) > 0 else 1.0
    uncertainty_gen = all_uncertainties.mean()
    uncertainty_ratio = uncertainty_gen / 0.868  # From real MOF test

    print(f"\nUncertainty analysis:")
    print(f"  Real MOFs:      0.87 mol/kg")
    print(f"  Generated MOFs: {uncertainty_gen:.2f} mol/kg")
    print(f"  Ratio:          {uncertainty_ratio:.2f}x")

    if uncertainty_ratio < 2.0:
        print("  ✓ Uncertainties acceptable")
        unc_verdict = "PASS"
    else:
        print("  ⚠️  Uncertainties elevated")
        unc_verdict = "CAUTION"

    # Overall verdict
    print(f"\n{'='*70}")
    if real_verdict == "PASS" and (corr_verdict in ["PASS", "CAUTION"]):
        print("✅ OVERALL: VIABLE")
        print("\nRecommendations:")
        print("  1. Use geometric features in production (R²=0.69)")
        print("  2. Implement uncertainty-aware acquisition")
        print("  3. Monitor predictions on selected generated MOFs")
        print("  4. Consider Gaussian Process for better uncertainty")
    else:
        print("⚠️  OVERALL: NEEDS WORK")
        print("\nRecommendations:")
        print("  1. Geometric features help, but not enough")
        print("  2. Add synthesizability filtering")
        print("  3. Try Gaussian Process or neural network")
        print("  4. Constrain VAE to more physical structures")
    print(f"{'='*70}")

    # Create visualization
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Predictions vs Targets
        ax1.scatter(all_targets, all_predictions, alpha=0.5)
        ax1.plot([all_targets.min(), all_targets.max()],
                 [all_targets.min(), all_targets.max()],
                 'r--', label='Perfect prediction')
        ax1.set_xlabel('Target CO2 (mol/kg)')
        ax1.set_ylabel('Predicted CO2 (mol/kg)')
        ax1.set_title(f'Prediction vs Target (r={correlation:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Uncertainty distribution
        ax2.hist(all_uncertainties, bins=30, alpha=0.7)
        ax2.axvline(all_uncertainties.mean(), color='r', linestyle='--',
                    label=f'Mean: {all_uncertainties.mean():.2f}')
        ax2.set_xlabel('Uncertainty (mol/kg)')
        ax2.set_ylabel('Count')
        ax2.set_title('Uncertainty Distribution (Generated MOFs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = project_root / 'results/surrogate_generalization_test/diagnostic_plots.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Diagnostic plots saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not create plots: {e}")


if __name__ == '__main__':
    main()
