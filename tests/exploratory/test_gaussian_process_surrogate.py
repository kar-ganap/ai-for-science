"""
Fix #2: Gaussian Process Surrogate

Test if GP with physics-informed kernels improves generalization.

Advantages of GP over Random Forest:
- Explicit uncertainty quantification
- Better extrapolation behavior
- Can incorporate domain knowledge via kernels
- More conservative (admits when uncertain)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

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

    # Geometric features
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
        features.extend([1.2, 18.0, 0.08, 0.92, 2.0])

    return np.array(features)


def main():
    print("="*70)
    print("FIX #2: GAUSSIAN PROCESS SURROGATE")
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

    # Featurize
    print("Preparing data...")
    X = []
    y = []

    for _, mof in mof_data.iterrows():
        features = featurize_with_geom(mof.to_dict(), geom_dict)
        X.append(features)
        y.append(mof['co2_uptake_mean'])

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (important for GP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {len(X_train)} MOFs")
    print(f"Test set: {len(X_test)} MOFs\n")

    # Define GP kernel
    # Matern(nu=2.5) is smoother than RBF but allows some flexibility
    # WhiteKernel captures noise
    print("Training Gaussian Process...")
    print("Kernel: Matern(nu=2.5) + WhiteKernel")
    print("(This may take 1-2 minutes...)\n")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,  # Noise is in the kernel
        n_restarts_optimizer=3,
        random_state=42
    )

    gp.fit(X_train_scaled, y_train)

    print("✓ Training complete!")
    print(f"Optimized kernel: {gp.kernel_}\n")

    # Evaluate on real MOFs
    y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"Real MOF performance:")
    print(f"  MAE: {test_mae:.3f} mol/kg")
    print(f"  R²:  {test_r2:.3f}")
    print(f"  Avg uncertainty: {y_std.mean():.3f} mol/kg")

    # Generate MOFs at multiple targets
    print(f"\n{'='*70}")
    print("Testing on generated MOFs...")
    print(f"{'='*70}\n")

    vae = DualConditionalMOFGenerator(use_geom_features=False)
    vae.load(vae_model)

    targets = [
        (3.0, 0.78),
        (5.0, 0.80),
        (7.0, 0.82),
        (9.0, 0.85),
    ]

    all_predictions = []
    all_uncertainties = []
    all_targets = []

    for target_co2, target_cost in targets:
        generated = vae.generate_candidates(
            n_candidates=50,
            target_co2=target_co2,
            target_cost=target_cost,
            temperature=4.0
        )

        # Featurize and predict
        X_gen = np.array([featurize_with_geom(mof, geom_dict) for mof in generated])
        X_gen_scaled = scaler.transform(X_gen)

        predictions, uncertainties = gp.predict(X_gen_scaled, return_std=True)

        print(f"Target {target_co2:.1f} mol/kg:")
        print(f"  Generated: {len(generated)} MOFs")
        print(f"  Predicted: {predictions.mean():.2f} ± {uncertainties.mean():.2f} mol/kg")
        print(f"  Uncertainty range: {uncertainties.min():.2f} - {uncertainties.max():.2f}\n")

        all_predictions.extend(predictions.tolist())
        all_uncertainties.extend(uncertainties.tolist())
        all_targets.extend([target_co2] * len(generated))

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_targets = np.array(all_targets)

    # Compute correlation
    correlation = np.corrcoef(all_predictions, all_targets)[0, 1] if len(set(all_targets)) > 1 else np.nan

    # Statistics
    print(f"{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")

    print(f"Prediction range: {all_predictions.min():.2f} - {all_predictions.max():.2f} mol/kg")
    print(f"Correlation (prediction vs target): {correlation:.3f}")
    print(f"Avg uncertainty: {all_uncertainties.mean():.2f} mol/kg")
    print(f"Max uncertainty: {all_uncertainties.max():.2f} mol/kg")
    print(f"Min uncertainty: {all_uncertainties.min():.2f} mol/kg")

    # Diagnosis
    print(f"\n{'='*70}")
    print("COMPARISON: Random Forest vs Gaussian Process")
    print(f"{'='*70}\n")

    print("Random Forest (from earlier):")
    print("  R² on real MOFs:     0.686")
    print("  Correlation (gen):   0.087")
    print("  Avg uncertainty:     1.41 mol/kg")

    print(f"\nGaussian Process:")
    print(f"  R² on real MOFs:     {test_r2:.3f}")
    print(f"  Correlation (gen):   {correlation:.3f}")
    print(f"  Avg uncertainty:     {all_uncertainties.mean():.2f} mol/kg")

    # Verdict
    print(f"\n{'='*70}")
    if correlation > 0.3:
        print("✅ IMPROVEMENT: GP shows better correlation!")
        if correlation > 0.5:
            print("   Strong enough for production use")
        else:
            print("   Moderate - combine with uncertainty penalties")
    elif correlation > 0.15:
        print("⚠️  SLIGHT IMPROVEMENT: GP marginally better")
        print("   Consider other approaches")
    else:
        print("❌ NO IMPROVEMENT: GP doesn't help")
        print("   Problem is deeper than model choice")
    print(f"{'='*70}")

    # Check uncertainty calibration
    print(f"\nUncertainty calibration:")
    print(f"  Real MOFs uncertainty: {y_std.mean():.2f} mol/kg")
    print(f"  Generated MOFs uncertainty: {all_uncertainties.mean():.2f} mol/kg")
    uncertainty_ratio = all_uncertainties.mean() / y_std.mean()
    print(f"  Ratio: {uncertainty_ratio:.2f}x")

    if uncertainty_ratio > 1.5:
        print("  ✓ GP is appropriately uncertain about generated MOFs")
        print("    (Good for safety - won't confidently make bad predictions)")
    else:
        print("  ⚠️  GP may be overconfident about generated MOFs")


if __name__ == '__main__':
    main()
