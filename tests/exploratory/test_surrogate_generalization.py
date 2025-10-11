"""
Surrogate Generalization Test: THE CRITICAL LYNCHPIN

This tests whether our surrogate model can make reliable predictions
for VAE-generated MOFs. If this fails, the whole Active Generative
Discovery system collapses.

Test Strategy:
1. Train surrogate on real MOFs (with cross-validation)
2. Generate novel MOFs using VAE
3. Compare surrogate performance on:
   - In-distribution: Real MOFs (cross-validation)
   - Out-of-distribution: Generated MOFs
4. Analyze uncertainty estimates
5. Determine if system is viable

Pass/Fail Criteria:
- PASS: Generated MOF predictions have reasonable uncertainty
        and correlate with targets
- FAIL: Generated MOF uncertainties are huge OR predictions random
        → System won't work, need fixes
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator


class SurrogateTester:
    """Test surrogate model generalization to generated MOFs"""

    def __init__(self):
        self.surrogate = None
        self.feature_names = None
        self.results = {}

    def featurize_mof(self, mof_dict, include_linker=True):
        """
        Convert MOF to feature vector

        Features:
        - Metal (one-hot)
        - Linker (one-hot, if include_linker=True)
        - Cell parameters (4 values)
        - Synthesis cost (1 value)
        """
        metals = ['Zn', 'Fe', 'Ca', 'Al', 'Ti', 'Unknown']
        linkers = ['terephthalic acid', 'trimesic acid',
                   '2,6-naphthalenedicarboxylic acid',
                   'biphenyl-4,4-dicarboxylic acid']

        features = []

        # Metal one-hot
        metal = mof_dict.get('metal', 'Unknown')
        for m in metals:
            features.append(1.0 if metal == m else 0.0)

        # Linker one-hot (if available)
        if include_linker:
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

        return np.array(features)

    def train_surrogate(self, mof_data, linker_data=None):
        """
        Train surrogate model on real MOFs

        Uses Random Forest with 100 trees for:
        - Decent performance
        - Built-in uncertainty (std of trees)
        - Fast training
        """
        print("="*70)
        print("TRAINING SURROGATE MODEL")
        print("="*70 + "\n")

        # Merge with linker data if available
        if linker_data is not None:
            mof_data = mof_data.merge(
                linker_data[['mof_id', 'linker']],
                on='mof_id',
                how='left'
            )
            include_linker = True
        else:
            include_linker = False

        # Featurize
        X = []
        y = []

        for _, mof in mof_data.iterrows():
            features = self.featurize_mof(mof.to_dict(), include_linker=include_linker)
            X.append(features)
            y.append(mof['co2_uptake_mean'])

        X = np.array(X)
        y = np.array(y)

        print(f"Dataset: {len(X)} MOFs")
        print(f"Features: {X.shape[1]} dimensions")
        print(f"Target range: {y.min():.2f} - {y.max():.2f} mol/kg\n")

        # Train-test split for held-out evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        print("Training Random Forest (100 trees)...")
        self.surrogate = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.surrogate.fit(X_train, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(
            self.surrogate, X_train, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        # Test set evaluation
        y_pred = self.surrogate.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        # Uncertainty estimation (std of tree predictions)
        tree_preds = np.array([tree.predict(X_test) for tree in self.surrogate.estimators_])
        uncertainties = tree_preds.std(axis=0)
        avg_uncertainty = uncertainties.mean()

        print(f"\n✓ Training complete!")
        print(f"\nPerformance on REAL MOFs:")
        print(f"  CV MAE:        {cv_mae:.3f} ± {cv_std:.3f} mol/kg")
        print(f"  Test MAE:      {test_mae:.3f} mol/kg")
        print(f"  Test R²:       {test_r2:.3f}")
        print(f"  Avg uncertainty: {avg_uncertainty:.3f} mol/kg")

        # Save results
        self.results['real_mof_performance'] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'avg_uncertainty': avg_uncertainty,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        return X_test, y_test, y_pred, uncertainties

    def predict_with_uncertainty(self, mof_dicts, include_linker=True):
        """
        Predict CO2 uptake with uncertainty for list of MOFs

        Returns:
            predictions: Mean predictions
            uncertainties: Standard deviation across trees
        """
        if self.surrogate is None:
            raise ValueError("Surrogate not trained! Call train_surrogate() first.")

        # Featurize
        X = np.array([self.featurize_mof(mof, include_linker) for mof in mof_dicts])

        # Predictions
        predictions = self.surrogate.predict(X)

        # Uncertainty from tree ensemble
        tree_preds = np.array([tree.predict(X) for tree in self.surrogate.estimators_])
        uncertainties = tree_preds.std(axis=0)

        return predictions, uncertainties

    def test_generated_mofs(self, vae_model_path, n_generate=100):
        """
        Test surrogate predictions on VAE-generated MOFs

        This is THE critical test:
        - If uncertainties are reasonable → system viable
        - If uncertainties huge → need better model or constraints
        - If predictions random → need physics constraints
        """
        print(f"\n{'='*70}")
        print("TESTING ON GENERATED MOFs")
        print(f"{'='*70}\n")

        # Load VAE
        vae = DualConditionalMOFGenerator(use_geom_features=False)
        vae.load(vae_model_path)

        # Generate MOFs at various targets
        targets = [
            (6.0, 0.80, "High CO2, Low Cost"),
            (8.0, 0.85, "Very High CO2, Med Cost"),
            (4.0, 0.78, "Medium CO2, Very Low Cost"),
        ]

        all_generated = []

        for target_co2, target_cost, desc in targets:
            print(f"Generating for: {desc}")
            candidates = vae.generate_candidates(
                n_candidates=n_generate // len(targets),
                target_co2=target_co2,
                target_cost=target_cost,
                temperature=4.0
            )
            all_generated.extend(candidates)
            print(f"  Generated {len(candidates)} candidates\n")

        print(f"Total generated: {len(all_generated)} MOFs")

        # Predict with uncertainty
        predictions, uncertainties = self.predict_with_uncertainty(all_generated)

        # Get target values for comparison
        targets_list = []
        for mof in all_generated:
            targets_list.append(mof.get('target_co2', 6.0))
        targets_array = np.array(targets_list)

        # Statistics
        avg_pred = predictions.mean()
        avg_uncertainty = uncertainties.mean()
        max_uncertainty = uncertainties.max()

        # Correlation with targets (should be positive if VAE conditioning works)
        correlation = np.corrcoef(predictions, targets_array)[0, 1]

        # Error relative to targets
        target_errors = np.abs(predictions - targets_array)
        avg_target_error = target_errors.mean()

        print(f"\nSurrogate predictions for GENERATED MOFs:")
        print(f"  Mean prediction:    {avg_pred:.3f} mol/kg")
        print(f"  Prediction range:   {predictions.min():.3f} - {predictions.max():.3f}")
        print(f"  Avg uncertainty:    {avg_uncertainty:.3f} mol/kg")
        print(f"  Max uncertainty:    {max_uncertainty:.3f} mol/kg")
        print(f"  Correlation with targets: {correlation:.3f}")
        print(f"  Avg error from target:    {avg_target_error:.3f} mol/kg")

        # Save results
        self.results['generated_mof_performance'] = {
            'n_generated': len(all_generated),
            'mean_prediction': avg_pred,
            'pred_range': (predictions.min(), predictions.max()),
            'avg_uncertainty': avg_uncertainty,
            'max_uncertainty': max_uncertainty,
            'correlation_with_targets': correlation,
            'avg_target_error': avg_target_error,
            'predictions': predictions.tolist(),
            'uncertainties': uncertainties.tolist(),
            'targets': targets_list
        }

        return all_generated, predictions, uncertainties

    def compare_distributions(self):
        """
        Compare real vs generated MOF prediction statistics

        Key metrics:
        - Uncertainty ratio: generated / real
        - Prediction stability
        """
        print(f"\n{'='*70}")
        print("REAL vs GENERATED COMPARISON")
        print(f"{'='*70}\n")

        real_perf = self.results['real_mof_performance']
        gen_perf = self.results['generated_mof_performance']

        # Uncertainty ratio
        uncertainty_ratio = gen_perf['avg_uncertainty'] / real_perf['avg_uncertainty']

        print(f"Real MOF uncertainty:      {real_perf['avg_uncertainty']:.3f} mol/kg")
        print(f"Generated MOF uncertainty: {gen_perf['avg_uncertainty']:.3f} mol/kg")
        print(f"Uncertainty ratio (gen/real): {uncertainty_ratio:.2f}x")

        # Interpretation
        print(f"\n{'='*70}")
        print("DIAGNOSIS")
        print(f"{'='*70}\n")

        diagnosis = []

        # Check 1: Uncertainty ratio
        if uncertainty_ratio < 1.5:
            diagnosis.append("✓ GOOD: Uncertainties comparable to real MOFs")
            verdict_uncertainty = "PASS"
        elif uncertainty_ratio < 3.0:
            diagnosis.append("⚠️  WARNING: Uncertainties 1.5-3× higher than real MOFs")
            diagnosis.append("   Recommendation: Use uncertainty-aware acquisition")
            verdict_uncertainty = "CAUTION"
        else:
            diagnosis.append("❌ CRITICAL: Uncertainties >3× higher than real MOFs")
            diagnosis.append("   Problem: Model not confident about generated MOFs")
            diagnosis.append("   Fix needed: Better features, constraints, or different model")
            verdict_uncertainty = "FAIL"

        # Check 2: Target correlation
        correlation = gen_perf['correlation_with_targets']
        if correlation > 0.5:
            diagnosis.append("\n✓ GOOD: Predictions correlate with VAE targets (r={:.2f})".format(correlation))
            diagnosis.append("   VAE conditioning is working")
            verdict_correlation = "PASS"
        elif correlation > 0.2:
            diagnosis.append("\n⚠️  WARNING: Weak correlation with targets (r={:.2f})".format(correlation))
            diagnosis.append("   VAE conditioning may need tuning")
            verdict_correlation = "CAUTION"
        else:
            diagnosis.append("\n❌ CRITICAL: No correlation with targets (r={:.2f})".format(correlation))
            diagnosis.append("   Problem: Predictions appear random")
            diagnosis.append("   Fix needed: Physics constraints or better surrogate")
            verdict_correlation = "FAIL"

        # Overall verdict
        if verdict_uncertainty == "PASS" and verdict_correlation == "PASS":
            overall = "PASS"
            diagnosis.append("\n🎉 OVERALL VERDICT: PASS")
            diagnosis.append("   System is viable for Active Generative Discovery")
            diagnosis.append("   Proceed with integration")
        elif "FAIL" in [verdict_uncertainty, verdict_correlation]:
            overall = "FAIL"
            diagnosis.append("\n🚨 OVERALL VERDICT: FAIL")
            diagnosis.append("   System has critical issues")
            diagnosis.append("   DO NOT proceed until fixes implemented")
        else:
            overall = "CAUTION"
            diagnosis.append("\n⚠️  OVERALL VERDICT: CAUTION")
            diagnosis.append("   System may work but needs improvements")
            diagnosis.append("   Recommend: Uncertainty-aware acquisition + monitoring")

        for line in diagnosis:
            print(line)

        # Save diagnosis
        self.results['diagnosis'] = {
            'uncertainty_ratio': uncertainty_ratio,
            'verdict_uncertainty': verdict_uncertainty,
            'verdict_correlation': verdict_correlation,
            'overall_verdict': overall,
            'recommendations': diagnosis
        }

        return overall


def main():
    """Run comprehensive surrogate generalization test"""

    print("="*70)
    print("SURROGATE GENERALIZATION TEST")
    print("Testing THE critical lynchpin for Active Generative Discovery")
    print("="*70 + "\n")

    print("Question: Can surrogate predict CO2 for VAE-generated MOFs?")
    print("If NO → System won't work\n")

    # Setup
    project_root = Path(__file__).parent
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    linker_file = project_root / "data/processed/crafted_mofs_linkers.csv"
    vae_model = project_root / "models/dual_conditional_mof_vae_compositional.pt"

    # Load data
    mof_data = pd.read_csv(mof_file)
    linker_data = pd.read_csv(linker_file)

    # Run test
    tester = SurrogateTester()

    # 1. Train on real MOFs
    X_test, y_test, y_pred, uncertainties = tester.train_surrogate(mof_data, linker_data)

    # 2. Test on generated MOFs
    generated_mofs, gen_predictions, gen_uncertainties = tester.test_generated_mofs(
        vae_model_path=vae_model,
        n_generate=100
    )

    # 3. Compare and diagnose
    verdict = tester.compare_distributions()

    # Save results
    output_dir = project_root / "results/surrogate_generalization_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_results.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'verdict': verdict,
            'results': tester.results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}/test_results.json")

    # Final summary
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")

    if verdict == "PASS":
        print("✅ System is ready for Active Generative Discovery")
        print("   Surrogate generalizes adequately to generated MOFs")
        return 0
    elif verdict == "CAUTION":
        print("⚠️  System needs improvements before deployment")
        print("   Implement uncertainty-aware acquisition")
        print("   Monitor predictions closely")
        return 1
    else:  # FAIL
        print("❌ System has critical issues")
        print("   DO NOT deploy without fixes")
        print("\nRecommended fixes:")
        print("  1. Add physics-based constraints to VAE")
        print("  2. Use better features (geometric descriptors)")
        print("  3. Try Gaussian Process instead of Random Forest")
        print("  4. Add synthesizability filtering")
        return 2


if __name__ == '__main__':
    exit(main())
