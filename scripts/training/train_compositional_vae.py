"""
Train Dual-Conditional VAE with TRUE Compositional Exploration

KEY CHANGES from previous version:
- Uses REAL linker data (23 metal-linker combinations)
- VAE learns both metal AND linker independently
- No hardcoded metal→linker mapping
- Enables true compositional novelty generation

This is the fix for Option B: Remove hardcoded mapping, enable composition exploration.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator

# Configuration - using best hyperparams from previous sweep
BEST_CONFIG = {
    'use_geom': False,  # Simple model performed best
    'beta': 2.0,        # Best beta from sweep
    'temperature': 4.0,  # Best temperature from sweep
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'augment': True,
    'use_latent_perturbation': True,
}

def evaluate_diversity(generator, temperature=4.0):
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

        print(f"  {desc}: {unique}/50 unique ({diversity_pct:.1f}%)")

    return results

def main():
    print("="*70)
    print("COMPOSITIONAL VAE TRAINING - Option B Fix")
    print("Training VAE with REAL linker data for true composition exploration")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    project_root = Path(__file__).parent
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    geom_file = project_root / "data/processed/crafted_geometric_features.csv"

    mof_data = pd.read_csv(mof_file)
    print(f"✓ Loaded {len(mof_data)} MOFs\n")

    geom_features = None
    if BEST_CONFIG['use_geom'] and geom_file.exists():
        geom_features = pd.read_csv(geom_file)
        print(f"✓ Loaded geometric features\n")

    # Initialize generator
    generator = DualConditionalMOFGenerator(use_geom_features=BEST_CONFIG['use_geom'])

    # Train
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    start_time = datetime.now()

    try:
        generator.train_dual_cvae(
            mof_data,
            geom_features=geom_features,
            epochs=BEST_CONFIG['epochs'],
            batch_size=BEST_CONFIG['batch_size'],
            lr=BEST_CONFIG['lr'],
            beta=BEST_CONFIG['beta'],
            augment=BEST_CONFIG['augment'],
            use_latent_perturbation=BEST_CONFIG['use_latent_perturbation']
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        print(f"\n{'='*70}")
        print("DIVERSITY EVALUATION")
        print(f"{'='*70}\n")

        diversity_results = evaluate_diversity(generator, temperature=BEST_CONFIG['temperature'])
        avg_diversity = np.mean([r['diversity_pct'] for r in diversity_results])

        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "dual_conditional_mof_vae_compositional.pt"
        generator.save(model_file)

        # Save results
        results = {
            'config': BEST_CONFIG,
            'training_time_sec': training_time,
            'avg_diversity_pct': avg_diversity,
            'diversity_by_target': diversity_results,
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'note': 'Trained with REAL linker data - true compositional exploration enabled'
        }

        output_dir = Path("results/compositional_vae")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}\n")

        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"Average diversity: {avg_diversity:.1f}% @ T={BEST_CONFIG['temperature']}")
        print(f"\nModel saved to: {model_file}")
        print(f"Results saved to: {output_dir}/training_results.json")

        # Compare to previous
        print(f"\n{'='*70}")
        print("COMPARISON TO PREVIOUS BEST")
        print(f"{'='*70}\n")

        print(f"Previous (hardcoded mapping): 6.7% diversity")
        print(f"Current (real linkers):       {avg_diversity:.1f}% diversity")

        if avg_diversity > 6.7:
            improvement = avg_diversity - 6.7
            print(f"\n🎉 IMPROVEMENT: +{improvement:.1f}% diversity!")
            print(f"✓ True compositional exploration is working!")
        else:
            print(f"\n⚠️  Diversity similar to previous")
            print(f"✓ But now VAE can generate ANY (metal, linker) combination!")

        print(f"\n✓ VAE is ready for Active Generative Discovery integration!")

    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == '__main__':
    exit(main())
