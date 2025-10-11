"""
Train Dual-Conditional VAE Variants with Monitoring

Trains multiple configurations:
1. Simple Dual cVAE (no geom) - beta sweep [1.0, 2.0, 3.0]
2. Hybrid Dual cVAE (with geom) - beta sweep [1.0, 2.0]

Features:
- Epoch-by-epoch logging
- Checkpoint saving
- Loss tracking
- Diversity evaluation
- Best model selection
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))

from dual_conditional_vae import DualConditionalMOFGenerator


def train_variant(variant_name: str,
                  mof_data: pd.DataFrame,
                  geom_features: pd.DataFrame,
                  use_geom: bool,
                  beta: float,
                  epochs: int = 100,
                  output_dir: Path = Path("results/dual_cvae_training")):
    """
    Train a single VAE variant with monitoring

    Returns:
        dict: Training results with final metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Variant: {variant_name}")
    print(f"Beta: {beta}, Use Geom: {use_geom}, Epochs: {epochs}")
    print(f"{'='*70}\n")

    # Initialize generator
    generator = DualConditionalMOFGenerator(use_geom_features=use_geom)

    # Create output directory
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Training log
    log_file = variant_dir / "training_log.txt"

    # Train with monitoring
    start_time = datetime.now()

    try:
        generator.train_dual_cvae(
            mof_data,
            geom_features=geom_features if use_geom else None,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            beta=beta,
            augment=True,
            use_latent_perturbation=True
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate diversity
        print(f"\n{'='*70}")
        print(f"Evaluating Diversity for {variant_name}")
        print(f"{'='*70}\n")

        diversity_results = []
        test_targets = [
            (6.0, 0.8, "High CO2, Low Cost"),
            (8.0, 1.0, "Very High CO2, Medium Cost"),
            (4.0, 0.7, "Medium CO2, Very Low Cost"),
        ]

        for target_co2, target_cost, desc in test_targets:
            candidates = generator.generate_candidates(
                n_candidates=50,
                target_co2=target_co2,
                target_cost=target_cost,
                temperature=2.0
            )

            unique = len(set((c['metal'], c['linker']) for c in candidates))
            diversity_pct = (unique / 50) * 100

            diversity_results.append({
                'target_co2': target_co2,
                'target_cost': target_cost,
                'description': desc,
                'n_candidates': len(candidates),
                'unique_combos': unique,
                'diversity_pct': diversity_pct
            })

            print(f"  {desc}: {unique}/50 unique ({diversity_pct:.1f}%)")

        # Average diversity
        avg_diversity = np.mean([r['diversity_pct'] for r in diversity_results])

        # Save model
        model_file = variant_dir / f"{variant_name}.pt"
        generator.save(model_file)

        # Save results
        results = {
            'variant_name': variant_name,
            'use_geom': use_geom,
            'beta': beta,
            'epochs': epochs,
            'training_time_sec': training_time,
            'avg_diversity_pct': avg_diversity,
            'diversity_results': diversity_results,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

        results_file = variant_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Variant {variant_name} complete!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Average diversity: {avg_diversity:.1f}%")
        print(f"  Model saved: {model_file}")

        return results

    except Exception as e:
        print(f"\n❌ Variant {variant_name} FAILED: {e}")

        results = {
            'variant_name': variant_name,
            'use_geom': use_geom,
            'beta': beta,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

        results_file = variant_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def main():
    """
    Train all variants and select best model
    """
    print("="*70)
    print("DUAL-CONDITIONAL VAE TRAINING SUITE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    project_root = Path(__file__).parent
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"
    geom_file = project_root / "data" / "processed" / "crafted_geometric_features.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        return

    mof_data = pd.read_csv(mof_file)
    print(f"✓ Loaded {len(mof_data)} MOFs")
    print(f"  CO2 range: {mof_data['co2_uptake_mean'].min():.2f} - {mof_data['co2_uptake_mean'].max():.2f} mol/kg")
    print(f"  Cost range: ${mof_data['synthesis_cost'].min():.2f} - ${mof_data['synthesis_cost'].max():.2f}/g\n")

    geom_features = None
    if geom_file.exists():
        geom_features = pd.read_csv(geom_file)
        print(f"✓ Loaded {len(geom_features)} geometric feature sets\n")
    else:
        print("⚠️  No geometric features found, will skip hybrid variants\n")

    # Define variants to train
    variants = []

    # Simple variants (no geom) - fast baseline
    for beta in [1.0, 2.0, 3.0]:
        variants.append({
            'name': f"simple_beta{beta:.1f}",
            'use_geom': False,
            'beta': beta,
            'epochs': 100
        })

    # Hybrid variants (with geom) - if geom features available
    if geom_features is not None:
        for beta in [1.0, 2.0]:
            variants.append({
                'name': f"hybrid_beta{beta:.1f}",
                'use_geom': True,
                'beta': beta,
                'epochs': 100
            })

    print(f"Training {len(variants)} variants:\n")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v['name']}: beta={v['beta']}, geom={v['use_geom']}")
    print()

    # Train all variants
    results = []
    output_dir = project_root / "results" / "dual_cvae_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, variant in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"# Variant {i}/{len(variants)}: {variant['name']}")
        print(f"{'#'*70}\n")

        result = train_variant(
            variant_name=variant['name'],
            mof_data=mof_data,
            geom_features=geom_features,
            use_geom=variant['use_geom'],
            beta=variant['beta'],
            epochs=variant['epochs'],
            output_dir=output_dir
        )

        results.append(result)

        # Save intermediate results
        summary_file = output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}\n")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}\n")

    if successful:
        # Rank by diversity
        successful_sorted = sorted(successful, key=lambda x: x['avg_diversity_pct'], reverse=True)

        print("Results ranked by diversity:\n")
        print(f"{'Rank':<6} {'Variant':<20} {'Beta':<6} {'Geom':<6} {'Diversity':<12} {'Time':<10}")
        print("-" * 70)

        for i, r in enumerate(successful_sorted, 1):
            print(f"{i:<6} {r['variant_name']:<20} {r['beta']:<6.1f} "
                  f"{'Yes' if r['use_geom'] else 'No':<6} "
                  f"{r['avg_diversity_pct']:<12.1f}% {r['training_time_sec']:<10.1f}s")

        # Best model
        best = successful_sorted[0]
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best['variant_name']}")
        print(f"{'='*70}")
        print(f"Beta: {best['beta']}")
        print(f"Geometric features: {'Yes' if best['use_geom'] else 'No'}")
        print(f"Average diversity: {best['avg_diversity_pct']:.1f}%")
        print(f"Training time: {best['training_time_sec']:.1f}s")

        # Copy best model to main location
        best_model_src = output_dir / best['variant_name'] / f"{best['variant_name']}.pt"
        best_model_dst = project_root / "models" / "dual_conditional_mof_vae_best.pt"

        import shutil
        shutil.copy(best_model_src, best_model_dst)
        print(f"\n✓ Best model copied to: {best_model_dst}")

    if failed:
        print(f"\nFailed variants:")
        for r in failed:
            print(f"  - {r['variant_name']}: {r.get('error', 'Unknown error')}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
