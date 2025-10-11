"""
Production VAE Training with All Best Practices

Hard-earned lessons from previous runs:
- Supercell + thermal noise augmentation (16× expansion)
- Geometric feature mapping for augmented data
- Dropout 0.2 for regularization
- Loss weighting: 10.0 × cell_loss
- Beta scaling: higher for more complex models
- Epoch 0 logging for sanity checks
- Temperature T=2.0 for generation
- Latent perturbation augmentation (NEW)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator

# Configuration with best practices
CONFIGS = {
    # Simple (fewer params) - lower beta
    'simple': {
        'use_geom': False,
        'beta_values': [1.0, 2.0],
        'expected_diversity': '8-10%',  # From previous simple cVAE
    },
    # Hybrid (more params: +11 geom) - higher beta
    'hybrid': {
        'use_geom': True,
        'beta_values': [2.0, 3.0, 4.0],
        'expected_diversity': '12-15%',  # Target: beat previous 12%
    }
}

# Fixed hyperparameters (from best practices)
HYPERPARAMS = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'latent_dim': 16,
    'hidden_dim_simple': 32,
    'hidden_dim_hybrid': 64,
    'augment': True,
    'use_latent_perturbation': True,
    'temperature': 2.0,  # For generation
}

def train_and_evaluate(variant_name, config, output_dir):
    """Train one variant with full monitoring"""
    print(f"\n{'='*70}")
    print(f"Variant: {variant_name}")
    print(f"Config: geom={config['use_geom']}, beta={config['beta']}")
    print(f"Expected diversity: {CONFIGS[config['type']]['expected_diversity']}")
    print(f"{'='*70}\n")
    
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    project_root = Path(__file__).parent
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    geom_file = project_root / "data/processed/crafted_geometric_features.csv"
    
    mof_data = pd.read_csv(mof_file)
    geom_features = pd.read_csv(geom_file) if config['use_geom'] else None
    
    # Initialize generator
    generator = DualConditionalMOFGenerator(use_geom_features=config['use_geom'])
    
    # Train
    start_time = datetime.now()
    try:
        generator.train_dual_cvae(
            mof_data,
            geom_features=geom_features,
            epochs=HYPERPARAMS['epochs'],
            batch_size=HYPERPARAMS['batch_size'],
            lr=HYPERPARAMS['lr'],
            beta=config['beta'],
            augment=HYPERPARAMS['augment'],
            use_latent_perturbation=HYPERPARAMS['use_latent_perturbation']
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate diversity (using best temperature from previous runs)
        print(f"\n{'='*70}")
        print("Diversity Evaluation")
        print(f"{'='*70}\n")
        
        test_targets = [
            (6.0, 0.8, "High CO2, Low Cost"),
            (8.0, 1.0, "Very High CO2, Medium Cost"),
            (4.0, 0.7, "Medium CO2, Very Low Cost"),
        ]
        
        diversity_results = []
        for target_co2, target_cost, desc in test_targets:
            candidates = generator.generate_candidates(
                n_candidates=50,
                target_co2=target_co2,
                target_cost=target_cost,
                temperature=HYPERPARAMS['temperature']
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
        
        avg_diversity = np.mean([r['diversity_pct'] for r in diversity_results])
        
        # Save model
        model_file = variant_dir / f"{variant_name}.pt"
        generator.save(model_file)
        
        results = {
            'variant_name': variant_name,
            'config': config,
            'hyperparams': HYPERPARAMS,
            'training_time_sec': training_time,
            'avg_diversity_pct': avg_diversity,
            'diversity_results': diversity_results,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open(variant_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ {variant_name} complete!")
        print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f}min)")
        print(f"  Average diversity: {avg_diversity:.1f}%")
        
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

def main():
    print("="*70)
    print("DUAL-CONDITIONAL VAE - PRODUCTION TRAINING")
    print("Incorporating all best practices from previous runs")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    output_dir = Path("results/dual_cvae_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build variant list with proper beta scaling
    variants = []
    
    for config_type, config_info in CONFIGS.items():
        for beta in config_info['beta_values']:
            variants.append({
                'name': f"{config_type}_beta{beta:.1f}",
                'type': config_type,
                'use_geom': config_info['use_geom'],
                'beta': beta
            })
    
    print(f"Training {len(variants)} variants:\n")
    for i, v in enumerate(variants, 1):
        expected = CONFIGS[v['type']]['expected_diversity']
        print(f"  {i}. {v['name']}: beta={v['beta']}, geom={v['use_geom']}, expect {expected}")
    print()
    
    # Train all
    results = []
    for i, variant in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"# Variant {i}/{len(variants)}: {variant['name']}")
        print(f"{'#'*70}")
        
        result = train_and_evaluate(variant['name'], variant, output_dir)
        results.append(result)
        
        # Save intermediate
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}\n")
    
    if successful:
        # Rank by diversity
        ranked = sorted(successful, key=lambda x: x['avg_diversity_pct'], reverse=True)
        
        print("Results ranked by diversity:\n")
        print(f"{'Rank':<6} {'Variant':<18} {'Beta':<6} {'Geom':<6} {'Diversity':<12} {'Time':<10}")
        print("-"*70)
        
        for i, r in enumerate(ranked, 1):
            print(f"{i:<6} {r['variant_name']:<18} {r['config']['beta']:<6.1f} "
                  f"{'Yes' if r['config']['use_geom'] else 'No':<6} "
                  f"{r['avg_diversity_pct']:<12.1f}% {r['training_time_sec']/60:<10.1f}min")
        
        # Best model
        best = ranked[0]
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best['variant_name']}")
        print(f"{'='*70}")
        print(f"Type: {'Hybrid' if best['config']['use_geom'] else 'Simple'}")
        print(f"Beta: {best['config']['beta']}")
        print(f"Diversity: {best['avg_diversity_pct']:.1f}%")
        print(f"Training time: {best['training_time_sec']/60:.1f} min")
        
        # Compare to previous best (12%)
        if best['avg_diversity_pct'] > 12.0:
            print(f"\n🎉 NEW BEST! Improved from 12% to {best['avg_diversity_pct']:.1f}%")
        else:
            print(f"\n⚠️  Below previous best (12%), but dual-conditioning adds cost awareness")
        
        # Copy best model
        import shutil
        src = output_dir / best['variant_name'] / f"{best['variant_name']}.pt"
        dst = Path("models/dual_conditional_mof_vae_best.pt")
        shutil.copy(src, dst)
        print(f"\n✓ Best model: {dst}")
    
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {output_dir}")

if __name__ == '__main__':
    main()
