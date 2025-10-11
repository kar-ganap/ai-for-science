"""
Quick Test: Does adding geometric features fix surrogate generalization?

This tests Fix #1 - using the geometric descriptors we already have
to improve surrogate predictions on generated MOFs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator


def featurize_with_geom(mof_dict, geom_dict):
    """Featurize MOF with geometric descriptors"""

    metals = ['Zn', 'Fe', 'Ca', 'Al', 'Ti', 'Unknown']
    linkers = ['terephthalic acid', 'trimesic acid',
               '2,6-naphthalenedicarboxylic acid',
               'biphenyl-4,4-dicarboxylic acid']

    features = []

    # Metal one-hot
    metal = mof_dict.get('metal', 'Unknown')
    for m in metals:
        features.append(1.0 if metal == m else 0.0)

    # Linker one-hot
    linker = mof_dict.get('linker', 'terephthalic acid')
    for l in linkers:
        features.append(1.0 if linker == l else 0.0)

    # Cell parameters
    features.extend([
        mof_dict.get('cell_a', 10.0),
        mof_dict.get('cell_b', 10.0),
        mof_dict.get('cell_c', 10.0),
        mof_dict.get('volume', 1000.0)
    ])

    # Synthesis cost
    features.append(mof_dict.get('synthesis_cost', 0.8))

    # Geometric features (if available)
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
        # Use median values for generated MOFs
        features.extend([1.2, 18.0, 0.08, 0.92, 2.0])

    return np.array(features)


def main():
    print("="*70)
    print("QUICK FIX TEST: Geometric Features")
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
    print("Featurizing with geometric descriptors...")
    X = []
    y = []

    for _, mof in mof_data.iterrows():
        features = featurize_with_geom(mof.to_dict(), geom_dict)
        X.append(features)
        y.append(mof['co2_uptake_mean'])

    X = np.array(X)
    y = np.array(y)

    print(f"Features: {X.shape[1]} dimensions (was 15, now {X.shape[1]})")
    print(f"  Added: 5 geometric descriptors\n")

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest...")
    surrogate = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    surrogate.fit(X_train, y_train)

    # Test on real MOFs
    y_pred = surrogate.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    tree_preds = np.array([tree.predict(X_test) for tree in surrogate.estimators_])
    uncertainties_real = tree_preds.std(axis=0).mean()

    print(f"\nReal MOF performance:")
    print(f"  MAE: {test_mae:.3f} mol/kg")
    print(f"  R²:  {test_r2:.3f} (was 0.234)")
    print(f"  Avg uncertainty: {uncertainties_real:.3f} mol/kg\n")

    # Test on generated MOFs
    print("Generating test MOFs...")
    vae = DualConditionalMOFGenerator(use_geom_features=False)
    vae.load(vae_model)

    generated = vae.generate_candidates(
        n_candidates=100,
        target_co2=7.0,
        target_cost=0.80,
        temperature=4.0
    )

    print(f"Generated: {len(generated)} MOFs\n")

    # Featurize and predict
    X_gen = np.array([featurize_with_geom(mof, geom_dict) for mof in generated])
    predictions = surrogate.predict(X_gen)

    tree_preds_gen = np.array([tree.predict(X_gen) for tree in surrogate.estimators_])
    uncertainties_gen = tree_preds_gen.std(axis=0)

    # Get targets
    targets = np.array([mof.get('target_co2', 7.0) for mof in generated])

    # Statistics
    correlation = np.corrcoef(predictions, targets)[0, 1]
    avg_uncertainty_gen = uncertainties_gen.mean()
    uncertainty_ratio = avg_uncertainty_gen / uncertainties_real

    print(f"Generated MOF predictions:")
    print(f"  Mean: {predictions.mean():.3f} mol/kg (target: 7.0)")
    print(f"  Correlation with targets: {correlation:.3f} (was -0.04)")
    print(f"  Avg uncertainty: {avg_uncertainty_gen:.3f} mol/kg")
    print(f"  Uncertainty ratio: {uncertainty_ratio:.2f}x (was 1.71x)")

    # Verdict
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}\n")

    print(f"Without geometric features:")
    print(f"  R² on real MOFs:     0.234")
    print(f"  Correlation (gen):  -0.04")
    print(f"  Uncertainty ratio:   1.71x")

    print(f"\nWith geometric features:")
    print(f"  R² on real MOFs:     {test_r2:.3f}")
    print(f"  Correlation (gen):   {correlation:.3f}")
    print(f"  Uncertainty ratio:   {uncertainty_ratio:.2f}x")

    print(f"\n{'='*70}")
    if test_r2 > 0.4 and correlation > 0.3:
        print("✅ IMPROVEMENT: Geometric features help!")
        print("   Recommendation: Use this for production")
    elif test_r2 > 0.3 or correlation > 0.2:
        print("⚠️  PARTIAL IMPROVEMENT: Some help, but not enough")
        print("   Recommendation: Combine with other fixes")
    else:
        print("❌ NO IMPROVEMENT: Geometric features don't help enough")
        print("   Need more fundamental changes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
