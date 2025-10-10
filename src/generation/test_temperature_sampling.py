"""
Quick test: Does temperature sampling fix mode collapse?

Load existing trained VAE and test diversity at different temperatures
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json


def temperature_sample_categorical(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from categorical distribution with temperature

    Args:
        logits: Raw logits [batch_size, n_classes]
        temperature: Sampling temperature
            - T=1.0: Normal sampling
            - T>1.0: More uniform (more diverse)
            - T<1.0: Sharper (less diverse)
            - T→0: Argmax (deterministic)
    """
    if temperature <= 0:
        return logits.argmax(dim=1)

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(1)
    return samples


def test_temperature_diversity(vae, n_samples=1000, temperatures=[0.5, 1.0, 1.5, 2.0]):
    """Test generation diversity at different temperatures"""
    vae.eval()

    results = {}

    print(f"\n{'='*60}")
    print(f"Testing Temperature Sampling (generating {n_samples} samples)")
    print(f"{'='*60}\n")

    for temp in temperatures:
        with torch.no_grad():
            # Sample from latent space
            samples = vae.sample(n_samples=n_samples)

            # Extract logits
            metal_logits = samples[:, :vae.n_metals]
            linker_logits = samples[:, vae.n_metals:vae.n_metals+vae.n_linkers]

            # Sample with temperature
            if temp == 0:
                # Argmax (original approach)
                metal_samples = metal_logits.argmax(dim=1)
                linker_samples = linker_logits.argmax(dim=1)
            else:
                metal_samples = temperature_sample_categorical(metal_logits, temp)
                linker_samples = temperature_sample_categorical(linker_logits, temp)

            # Count unique combinations
            combinations = list(zip(metal_samples.tolist(), linker_samples.tolist()))
            unique_combinations = set(combinations)
            diversity = len(unique_combinations) / n_samples

            # Count frequency of each combination
            from collections import Counter
            combo_counts = Counter(combinations)
            top_3 = combo_counts.most_common(3)

            results[temp] = {
                'unique_count': len(unique_combinations),
                'diversity': diversity,
                'top_combinations': top_3
            }

            print(f"Temperature: {temp if temp > 0 else 'argmax'}")
            print(f"  Unique combinations: {len(unique_combinations)} / {n_samples}")
            print(f"  Diversity: {diversity:.1%}")
            print(f"  Top 3 most common:")
            for (metal, linker), count in top_3:
                print(f"    Metal {metal}, Linker {linker}: {count} samples ({count/n_samples:.1%})")
            print()

    return results


if __name__ == '__main__':
    print("Temperature Sampling Test\n" + "="*60)

    # Load the best trained model
    project_root = Path(__file__).parents[2]
    model_dir = project_root / "results" / "vae_evaluation"

    # Find the best model file
    model_files = list(model_dir.glob("best_vae_simple_beta1.0_*.pt"))
    if not model_files:
        print("❌ No saved model found!")
        print(f"   Looking in: {model_dir}")
        exit(1)

    model_file = sorted(model_files)[-1]  # Most recent
    print(f"✓ Loading model: {model_file.name}\n")

    # Load model
    checkpoint = torch.load(model_file)

    # Reconstruct VAE
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from mof_vae import MOF_VAE

    # Get model dimensions from checkpoint (infer from state dict)
    state_dict = checkpoint['model_state']
    n_metals = state_dict['decoder.4.weight'].shape[0] - 4  # Total output - 4 cell params

    # We know from training: 6 metals, 4 linkers
    n_metals = 6
    n_linkers = 4

    vae = MOF_VAE(
        n_metals=n_metals,
        n_linkers=n_linkers,
        latent_dim=16,
        hidden_dim=32
    )
    vae.load_state_dict(state_dict)
    vae.eval()

    print(f"✓ Model loaded (6 metals, 4 linkers)\n")

    # Test different temperatures
    temperatures = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = test_temperature_diversity(vae, n_samples=1000, temperatures=temperatures)

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Temp':<10} {'Unique':<10} {'Diversity':<15} {'Top combo %':<15}")
    print("-" * 60)
    for temp, res in results.items():
        temp_str = 'argmax' if temp == 0 else f"{temp:.1f}"
        top_pct = res['top_combinations'][0][1] / 1000
        print(f"{temp_str:<10} {res['unique_count']:<10} {res['diversity']:<15.1%} {top_pct:<15.1%}")

    # Best temperature
    best_temp = max(results.items(), key=lambda x: x[1]['diversity'])
    print(f"\n🏆 Best temperature: {best_temp[0]}")
    print(f"   Diversity: {best_temp[1]['diversity']:.1%} ({best_temp[1]['unique_count']} unique combinations)")

    # Save results
    output_file = project_root / "results" / "vae_evaluation" / "temperature_test_results.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for temp, res in results.items():
        json_results[str(temp)] = {
            'unique_count': res['unique_count'],
            'diversity': res['diversity'],
            'top_combinations': [[int(m), int(l), int(c)] for (m, l), c in res['top_combinations']]
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
