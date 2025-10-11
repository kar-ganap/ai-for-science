"""
Comprehensive VAE Hyperparameter Sweep

Phase 1: Beta sweep (training time ~3-4 hours)
- Simple: beta = [1.0, 1.5, 2.0] (fixed T=2.0 for evaluation)
- Hybrid: beta = [1.0, 1.5, 2.0, 3.0, 4.0] (fixed T=2.0 for evaluation)

Phase 2: Temperature sweep (evaluation only, ~5 min)
- Use best model from Phase 1
- Test T = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- NO retraining needed!

This will tell us:
1. Does lower beta → higher diversity? (hypothesis: yes)
2. Does temperature sampling help beyond beta tuning? (hypothesis: marginally)
3. What's the best configuration overall?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator

# Configuration
PHASE_1_CONFIGS = {
    'simple': {
        'use_geom': False,
        'beta_values': [1.0, 1.5, 2.0],  # Lower beta for simpler model
        'hidden_dim': 32
    },
    'hybrid': {
        'use_geom': True,
        'beta_values': [1.0, 1.5, 2.0, 3.0, 4.0],  # Full sweep
        'hidden_dim': 64
    }
}

PHASE_2_TEMPERATURES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

FIXED_PARAMS = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'augment': True,
    'use_latent_perturbation': True,
    'eval_temperature': 2.0,  # For Phase 1 evaluation
}

def train_one_variant(variant_name, config, output_dir):
    """Train a single variant"""
    print(f"\n{'='*70}")
    print(f"Training: {variant_name}")
    print(f"Beta: {config['beta']}, Geom: {config['use_geom']}")
    print(f"{'='*70}\n")

    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    project_root = Path(__file__).parent
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    geom_file = project_root / "data/processed/crafted_geometric_features.csv"

    mof_data = pd.read_csv(mof_file)
    geom_features = pd.read_csv(geom_file) if config['use_geom'] else None

    # Train
    generator = DualConditionalMOFGenerator(use_geom_features=config['use_geom'])

    start_time = datetime.now()
    try:
        generator.train_dual_cvae(
            mof_data,
            geom_features=geom_features,
            epochs=FIXED_PARAMS['epochs'],
            batch_size=FIXED_PARAMS['batch_size'],
            lr=FIXED_PARAMS['lr'],
            beta=config['beta'],
            augment=FIXED_PARAMS['augment'],
            use_latent_perturbation=FIXED_PARAMS['use_latent_perturbation']
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate at fixed temperature
        diversity_results = evaluate_diversity(
            generator,
            temperature=FIXED_PARAMS['eval_temperature']
        )
        avg_diversity = np.mean([r['diversity_pct'] for r in diversity_results])

        # Save model
        model_file = variant_dir / f"{variant_name}.pt"
        generator.save(model_file)

        results = {
            'variant_name': variant_name,
            'config': config,
            'training_time_sec': training_time,
            'eval_temperature': FIXED_PARAMS['eval_temperature'],
            'avg_diversity_pct': avg_diversity,
            'diversity_by_target': diversity_results,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

        with open(variant_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ {variant_name} complete!")
        print(f"  Time: {training_time/60:.1f} min")
        print(f"  Diversity @ T={FIXED_PARAMS['eval_temperature']}: {avg_diversity:.1f}%")

        return results

    except Exception as e:
        print(f"\n❌ {variant_name} FAILED: {e}")
        return {
            'variant_name': variant_name,
            'config': config,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def evaluate_diversity(generator, temperature=2.0):
    """Evaluate diversity at given temperature"""
    test_targets = [
        (6.0, 0.8, "High CO2, Low Cost"),
        (8.0, 1.0, "Very High CO2, Medium Cost"),
        (4.0, 0.7, "Medium CO2, Very Low Cost"),
    ]

    results = []
    for target_co2, target_cost, desc in test_targets:
        candidates = generator.generate_candidates(
            n_candidates=50,
            target_co2=target_co2,
            target_cost=target_cost,
            temperature=temperature
        )

        unique = len(set((c['metal'], c['linker']) for c in candidates))
        diversity_pct = (unique / 50) * 100

        results.append({
            'target_co2': target_co2,
            'target_cost': target_cost,
            'description': desc,
            'n_candidates': len(candidates),
            'unique_combos': unique,
            'diversity_pct': diversity_pct,
            'temperature': temperature
        })

    return results

def phase_1_beta_sweep():
    """Phase 1: Train models with different beta values"""
    print("="*70)
    print("PHASE 1: BETA SWEEP")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    output_dir = Path("results/comprehensive_vae_sweep/phase1_beta")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build variant list
    variants = []
    for model_type, type_config in PHASE_1_CONFIGS.items():
        for beta in type_config['beta_values']:
            variants.append({
                'name': f"{model_type}_beta{beta:.1f}",
                'type': model_type,
                'use_geom': type_config['use_geom'],
                'beta': beta,
                'hidden_dim': type_config['hidden_dim']
            })

    print(f"Training {len(variants)} variants:\n")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v['name']}: beta={v['beta']}, geom={v['use_geom']}")
    print()

    # Train all
    results = []
    for i, variant in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"# Variant {i}/{len(variants)}: {variant['name']}")
        print(f"{'#'*70}")

        result = train_one_variant(variant['name'], variant, output_dir)
        results.append(result)

        # Save intermediate
        with open(output_dir / "phase1_summary.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print_phase1_summary(results, output_dir)

    return results

def phase_2_temperature_sweep(best_model_path):
    """Phase 2: Temperature sweep on best model (NO retraining)"""
    print("\n" + "="*70)
    print("PHASE 2: TEMPERATURE SWEEP (Evaluation Only)")
    print("="*70)
    print(f"Best model: {best_model_path}")
    print(f"Temperatures: {PHASE_2_TEMPERATURES}\n")

    output_dir = Path("results/comprehensive_vae_sweep/phase2_temperature")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load best model
    generator = DualConditionalMOFGenerator(use_geom_features=True)
    generator.load(best_model_path)

    # Test each temperature
    results = []
    for temp in PHASE_2_TEMPERATURES:
        print(f"\nEvaluating T={temp:.1f}...")

        diversity_results = evaluate_diversity(generator, temperature=temp)
        avg_diversity = np.mean([r['diversity_pct'] for r in diversity_results])

        results.append({
            'temperature': temp,
            'avg_diversity_pct': avg_diversity,
            'diversity_by_target': diversity_results
        })

        print(f"  Diversity @ T={temp:.1f}: {avg_diversity:.1f}%")

    # Save
    with open(output_dir / "phase2_temperature_sweep.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*70)
    print("PHASE 2 SUMMARY: Temperature Impact")
    print("="*70 + "\n")

    print(f"{'Temperature':<12} {'Avg Diversity':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['temperature']:<12.1f} {r['avg_diversity_pct']:<15.1f}%")

    best_temp = max(results, key=lambda x: x['avg_diversity_pct'])
    print(f"\n✓ Best temperature: T={best_temp['temperature']:.1f} ({best_temp['avg_diversity_pct']:.1f}% diversity)")

    return results

def print_phase1_summary(results, output_dir):
    """Print Phase 1 summary"""
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY: Beta Impact")
    print("="*70 + "\n")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}\n")

    if successful:
        # Rank by diversity
        ranked = sorted(successful, key=lambda x: x['avg_diversity_pct'], reverse=True)

        print(f"{'Rank':<6} {'Variant':<18} {'Beta':<6} {'Geom':<6} {'Diversity':<12} {'Time':<10}")
        print("-"*70)

        for i, r in enumerate(ranked, 1):
            print(f"{i:<6} {r['variant_name']:<18} {r['config']['beta']:<6.1f} "
                  f"{'Yes' if r['config']['use_geom'] else 'No':<6} "
                  f"{r['avg_diversity_pct']:<12.1f}% {r['training_time_sec']/60:<10.1f}min")

        # Best model
        best = ranked[0]
        print(f"\n✓ BEST MODEL (Phase 1): {best['variant_name']}")
        print(f"  Beta: {best['config']['beta']}")
        print(f"  Diversity: {best['avg_diversity_pct']:.1f}% @ T={best['eval_temperature']}")

        # Copy best model
        import shutil
        src = output_dir / best['variant_name'] / f"{best['variant_name']}.pt"
        dst = Path("models/dual_conditional_mof_vae_phase1_best.pt")
        shutil.copy(src, dst)
        print(f"  Saved to: {dst}")

        return dst

    return None

def main():
    """Run comprehensive sweep"""
    print("="*70)
    print("COMPREHENSIVE VAE HYPERPARAMETER SWEEP")
    print("="*70 + "\n")

    # Phase 1: Beta sweep
    phase1_results = phase_1_beta_sweep()

    # Find best model from Phase 1
    successful = [r for r in phase1_results if r['success']]
    if not successful:
        print("\n❌ No successful models in Phase 1!")
        return

    best_phase1 = max(successful, key=lambda x: x['avg_diversity_pct'])
    best_model_path = Path("models/dual_conditional_mof_vae_phase1_best.pt")

    # Phase 2: Temperature sweep on best model
    print(f"\nProceeding to Phase 2 with best model: {best_phase1['variant_name']}")
    phase2_results = phase_2_temperature_sweep(best_model_path)

    # Final summary
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70 + "\n")

    print(f"Best Beta: {best_phase1['config']['beta']}")
    print(f"Best Diversity (Beta sweep): {best_phase1['avg_diversity_pct']:.1f}%")

    best_temp = max(phase2_results, key=lambda x: x['avg_diversity_pct'])
    print(f"\nBest Temperature: {best_temp['temperature']:.1f}")
    print(f"Best Diversity (Temp sweep): {best_temp['avg_diversity_pct']:.1f}%")

    improvement = best_temp['avg_diversity_pct'] - best_phase1['avg_diversity_pct']
    print(f"\nTemperature improvement: {improvement:+.1f}%")

    if improvement > 1.0:
        print("✓ Temperature sampling provides meaningful boost!")
    else:
        print("⚠️ Temperature sampling provides minimal improvement")

    print(f"\n✓ FINAL BEST CONFIG:")
    print(f"  Model: {best_phase1['variant_name']}")
    print(f"  Beta: {best_phase1['config']['beta']}")
    print(f"  Temperature: {best_temp['temperature']:.1f}")
    print(f"  Expected diversity: {best_temp['avg_diversity_pct']:.1f}%")

if __name__ == '__main__':
    main()
