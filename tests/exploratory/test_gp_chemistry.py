"""
Test Gaussian Process with Chemistry Features

Goal: Beat current best (GP + geometric features: r=0.283)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(".") / "src" / "generation"))
sys.path.insert(0, str(Path(".") / "src" / "featurization"))

from dual_conditional_vae import DualConditionalMOFGenerator
from chemistry_features import ChemistryFeaturizer


def featurize_mof(mof_dict):
    """Simple featurization (compatible with generated MOFs)"""
    metals = ['Zn', 'Fe', 'Ca', 'Al', 'Ti', 'Cu', 'Zr', 'Cr', 'Unknown']
    linkers = ['terephthalic acid', 'trimesic acid',
               '2,6-naphthalenedicarboxylic acid',
               'biphenyl-4,4-dicarboxylic acid']

    features = []

    # Metal one-hot
    metal = mof_dict.get('metal', 'Unknown')
    for m in metals:
        features.append(1.0 if metal == m else 0.0)

    # Linker one-hot
    linker = mof_dict.get('linker', 'Unknown')
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

    return np.array(features)


def main():
    print("="*70)
    print("TESTING GP WITH CHEMISTRY FEATURES")
    print("="*70 + "\n")

    project_root = Path(".")

    # Load real labeled MOFs
    real_mofs = pd.read_csv(project_root / "data/processed/crafted_mofs_co2_with_costs.csv")
    linker_data = pd.read_csv(project_root / "data/processed/crafted_mofs_linkers.csv")
    real_mofs = real_mofs.merge(linker_data[['mof_id', 'linker']], on='mof_id', how='left')

    print(f"Real labeled MOFs: {len(real_mofs)}\n")

    # Featurize with basic + chemistry features
    print("Featurizing real MOFs...")
    X_basic = np.array([featurize_mof(mof) for mof in real_mofs.to_dict('records')])

    chem_feat = ChemistryFeaturizer(include_derived=True)
    X_chem = np.array([chem_feat.featurize(mof) for mof in real_mofs.to_dict('records')])

    X_combined = np.concatenate([X_basic, X_chem], axis=1)
    y = real_mofs['co2_uptake_mean'].values

    print(f"  Combined features: {X_combined.shape[1]} dimensions\n")

    # Standardize for GP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train GP with optimized kernel
    print("Training Gaussian Process with chemistry features...")
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.1,
        normalize_y=True,
        random_state=42
    )

    gp.fit(X_train, y_train)

    # Test on real MOFs
    y_pred, y_std = gp.predict(X_test, return_std=True)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"✓ Trained")
    print(f"  R² on real MOFs: {test_r2:.3f}")
    print(f"  MAE: {test_mae:.3f} mol/kg")
    print(f"  Avg uncertainty: {y_std.mean():.3f} mol/kg\n")

    # Test on generated MOFs
    print("="*70)
    print("TESTING ON GENERATED MOFs")
    print("="*70 + "\n")

    vae_model = project_root / "models/dual_conditional_mof_vae_compositional.pt"
    vae = DualConditionalMOFGenerator(use_geom_features=False)
    vae.load(vae_model)

    targets = [(3.0, 0.78), (5.0, 0.80), (7.0, 0.82), (9.0, 0.85)]

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

        # Featurize
        X_gen_basic = np.array([featurize_mof(mof) for mof in generated])
        X_gen_chem = np.array([chem_feat.featurize(mof) for mof in generated])
        X_gen = np.concatenate([X_gen_basic, X_gen_chem], axis=1)
        X_gen_scaled = scaler.transform(X_gen)

        predictions, uncertainties = gp.predict(X_gen_scaled, return_std=True)

        print(f"Target {target_co2:.1f} mol/kg:")
        print(f"  Generated: {len(generated)} MOFs")
        print(f"  Predicted: {predictions.mean():.2f} ± {predictions.std():.2f} mol/kg")
        print(f"  Avg uncertainty: {uncertainties.mean():.2f} mol/kg\n")

        all_predictions.extend(predictions.tolist())
        all_uncertainties.extend(uncertainties.tolist())
        all_targets.extend([target_co2] * len(generated))

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_targets = np.array(all_targets)

    correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    print(f"Correlation (predictions vs targets): {correlation:.3f}")
    print(f"Average uncertainty: {all_uncertainties.mean():.2f} mol/kg\n")

    print("COMPARISON:")
    print(f"  RF baseline:                       r = -0.040")
    print(f"  RF + geometric features:           r =  0.087")
    print(f"  RF + chemistry features:           r =  0.161")
    print(f"  GP + geometric features (prev):    r =  0.283")
    print(f"  GP + chemistry features (NEW):     r =  {correlation:.3f}\n")

    if correlation > 0.283:
        print("✅ NEW BEST! Chemistry features help GP!")
        print(f"   Improvement: {((correlation - 0.283) / 0.283 * 100):.1f}%")
    elif correlation > 0.161:
        print("✓ Better than RF+chem, but below prev GP")
    else:
        print("⚠️  Didn't improve over RF+chem")

    print(f"\n{'='*70}")
    print("✅ GP + CHEMISTRY FEATURES TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
