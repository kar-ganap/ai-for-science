"""
Improved VAE Evaluation with:
1. Temperature sampling (fixes mode collapse)
2. Train/val split (held-out evaluation)
3. Epoch 1 loss tracking

Re-trains Simple VAE β=1.0 with these improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime


def temperature_sample_categorical(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from categorical distribution with temperature

    Args:
        logits: Raw logits [batch_size, n_classes]
        temperature: Sampling temperature (higher = more diverse)
            - T=1.0: Normal sampling
            - T>1.0: More uniform (more diversity)
            - T<1.0: Sharper (less diversity)
            - T→0: Argmax (deterministic)

    Returns:
        Sampled indices [batch_size]
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Sample from categorical distribution
    probs = F.softmax(scaled_logits, dim=1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(1)

    return samples


class ImprovedVAEEvaluator:
    """
    Improved VAE evaluation with train/val split and temperature sampling
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def load_and_split_data(self, val_fraction: float = 0.2):
        """Load data and split into train/val"""
        # Load augmented MOF data
        mof_file = self.project_root / "data" / "processed" / "crafted_mofs_augmented.csv"
        self.mof_data = pd.read_csv(mof_file)

        # Random split (stratified by original MOF to avoid data leakage from augmentation)
        np.random.seed(42)
        unique_mof_ids = self.mof_data['mof_id'].unique()
        n_val_mofs = int(len(unique_mof_ids) * val_fraction)

        val_mof_ids = np.random.choice(unique_mof_ids, size=n_val_mofs, replace=False)
        val_mask = self.mof_data['mof_id'].isin(val_mof_ids)

        self.train_data = self.mof_data[~val_mask].reset_index(drop=True)
        self.val_data = self.mof_data[val_mask].reset_index(drop=True)

        print(f"✓ Loaded {len(self.mof_data)} augmented MOF samples")
        print(f"  Train: {len(self.train_data)} samples ({len(self.train_data['mof_id'].unique())} unique MOFs)")
        print(f"  Val:   {len(self.val_data)} samples ({len(self.val_data['mof_id'].unique())} unique MOFs)")

    def prepare_vae_data(self, mof_data: pd.DataFrame):
        """Prepare data for VAE training"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from mof_vae import MOFGenerator

        generator = MOFGenerator()
        X, metal_encoder, linker_encoder = generator._encode_mofs(mof_data)

        return X, metal_encoder, linker_encoder, generator.cell_mean, generator.cell_std

    def train_simple_vae_improved(self, beta: float = 1.0, epochs: int = 100, batch_size: int = 32):
        """Train simple VAE with improvements"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from mof_vae import MOF_VAE, vae_loss

        print(f"\n{'='*60}")
        print(f"Training Improved Simple VAE (β={beta})")
        print(f"{'='*60}\n")

        # Prepare train and val data
        X_train, metal_encoder, linker_encoder, cell_mean, cell_std = self.prepare_vae_data(self.train_data)
        X_val, _, _, _, _ = self.prepare_vae_data(self.val_data)

        train_dataset = torch.FloatTensor(X_train)
        val_dataset = torch.FloatTensor(X_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        vae = MOF_VAE(
            n_metals=len(metal_encoder),
            n_linkers=len(linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training loop with epoch 1 logging
        train_losses = {'total': [], 'recon': [], 'kl': []}
        val_losses = {'total': [], 'recon': [], 'kl': []}

        for epoch in range(epochs):
            vae.train()
            epoch_train_loss = 0
            epoch_train_recon = 0
            epoch_train_kl = 0

            for batch in train_loader:
                optimizer.zero_grad()

                recon, mu, logvar = vae(batch)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, batch, mu, logvar,
                    vae.n_metals, vae.n_linkers,
                    beta=beta
                )

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_recon += recon_loss.item()
                epoch_train_kl += kl_loss.item()

            # Track train losses
            train_losses['total'].append(epoch_train_loss / len(train_dataset))
            train_losses['recon'].append(epoch_train_recon / len(train_dataset))
            train_losses['kl'].append(epoch_train_kl / len(train_dataset))

            # Validation
            vae.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = vae(val_dataset)
                val_loss, val_recon_loss, val_kl_loss = vae_loss(
                    val_recon, val_dataset, val_mu, val_logvar,
                    vae.n_metals, vae.n_linkers,
                    beta=beta
                )

                val_losses['total'].append(val_loss.item() / len(val_dataset))
                val_losses['recon'].append(val_recon_loss.item() / len(val_dataset))
                val_losses['kl'].append(val_kl_loss.item() / len(val_dataset))

            # Print: epoch 1, then every 20
            if epoch == 0 or (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss={train_losses['total'][-1]:.2f} "
                      f"(Recon={train_losses['recon'][-1]:.2f}, KL={train_losses['kl'][-1]:.2f})")
                print(f"  Val   Loss={val_losses['total'][-1]:.2f} "
                      f"(Recon={val_losses['recon'][-1]:.2f}, KL={val_losses['kl'][-1]:.2f})")

        # Evaluate on validation set with different temperatures
        print(f"\n{'='*40}")
        print("Evaluating with different temperatures")
        print(f"{'='*40}\n")

        temperatures = [0.5, 1.0, 1.5, 2.0]
        temp_results = {}

        for temp in temperatures:
            metrics = self.evaluate_vae_with_temperature(vae, val_dataset, temperature=temp)
            temp_results[temp] = metrics

            print(f"Temperature {temp}:")
            print(f"  Val Recon Acc: {metrics['recon_accuracy']:.1%}")
            print(f"  Val Cell RMSE: {metrics['cell_rmse']:.3f}")
            print(f"  Generation Diversity: {metrics['generation_diversity']:.1%}")
            print()

        result = {
            'beta': beta,
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch1_train_loss': train_losses['total'][0],
            'final_train_loss': train_losses['total'][-1],
            'final_val_loss': val_losses['total'][-1],
            'temperature_results': temp_results,
            'model': vae,
            'metal_encoder': metal_encoder,
            'linker_encoder': linker_encoder,
        }

        print(f"✓ Improved VAE training complete!")
        print(f"  Epoch 1 train loss: {train_losses['total'][0]:.2f}")
        print(f"  Final train loss:   {train_losses['total'][-1]:.2f}")
        print(f"  Final val loss:     {val_losses['total'][-1]:.2f}")
        print(f"  Train-val gap:      {train_losses['total'][-1] - val_losses['total'][-1]:.2f}")

        return result

    def evaluate_vae_with_temperature(self, vae, val_dataset, temperature: float = 1.0):
        """
        Evaluate VAE on validation set with temperature sampling

        Args:
            vae: Trained VAE model
            val_dataset: Validation data
            temperature: Sampling temperature for diversity metric
        """
        vae.eval()

        with torch.no_grad():
            # Reconstruction on validation set
            recon, mu, logvar = vae(val_dataset)

            # Metal accuracy (still use argmax for reconstruction eval)
            metal_pred = recon[:, :vae.n_metals].argmax(dim=1)
            metal_true = val_dataset[:, :vae.n_metals].argmax(dim=1)
            metal_acc = (metal_pred == metal_true).float().mean().item()

            # Linker accuracy
            linker_pred = recon[:, vae.n_metals:vae.n_metals+vae.n_linkers].argmax(dim=1)
            linker_true = val_dataset[:, vae.n_metals:vae.n_metals+vae.n_linkers].argmax(dim=1)
            linker_acc = (linker_pred == linker_true).float().mean().item()

            recon_acc = (metal_acc + linker_acc) / 2

            # Cell param RMSE
            cell_pred = recon[:, vae.n_metals+vae.n_linkers:]
            cell_true = val_dataset[:, vae.n_metals+vae.n_linkers:]
            cell_rmse = torch.sqrt(F.mse_loss(cell_pred, cell_true)).item()

            # Latent coverage
            latent_vars = logvar.exp().mean(dim=0)
            effective_dim = (latent_vars.sum() ** 2) / (latent_vars ** 2).sum()
            latent_coverage = effective_dim.item() / vae.latent_dim

            # Generation diversity WITH TEMPERATURE SAMPLING
            samples = vae.sample(n_samples=1000)

            # Use temperature sampling instead of argmax!
            metal_logits = samples[:, :vae.n_metals]
            linker_logits = samples[:, vae.n_metals:vae.n_metals+vae.n_linkers]

            metal_samples = temperature_sample_categorical(metal_logits, temperature)
            linker_samples = temperature_sample_categorical(linker_logits, temperature)

            # Count unique combinations
            combinations = set(zip(metal_samples.tolist(), linker_samples.tolist()))
            diversity = len(combinations) / 1000

        return {
            'recon_accuracy': recon_acc,
            'metal_accuracy': metal_acc,
            'linker_accuracy': linker_acc,
            'cell_rmse': cell_rmse,
            'latent_coverage': latent_coverage,
            'generation_diversity': diversity,
            'unique_combinations': len(combinations)
        }

    def save_results(self, result):
        """Save improved evaluation results"""
        results_dir = self.project_root / "results" / "vae_evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary (exclude model)
        summary = {
            'beta': result['beta'],
            'epochs': result['epochs'],
            'epoch1_train_loss': result['epoch1_train_loss'],
            'final_train_loss': result['final_train_loss'],
            'final_val_loss': result['final_val_loss'],
            'train_val_gap': result['final_train_loss'] - result['final_val_loss'],
            'temperature_results': result['temperature_results'],
        }

        summary_file = results_dir / f"improved_evaluation_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Results saved to: {summary_file}")

        # Save model
        model_file = results_dir / f"improved_vae_beta{result['beta']}_{timestamp}.pt"
        torch.save({
            'model_state': result['model'].state_dict(),
            'beta': result['beta'],
            'metal_encoder': result['metal_encoder'],
            'linker_encoder': result['linker_encoder'],
        }, model_file)

        print(f"✓ Model saved to: {model_file}")


if __name__ == '__main__':
    print("Improved VAE Evaluation\n" + "="*60)

    # Setup
    project_root = Path(__file__).parents[2]
    evaluator = ImprovedVAEEvaluator(project_root)

    # Load and split data
    evaluator.load_and_split_data(val_fraction=0.2)

    # Train with improvements
    result = evaluator.train_simple_vae_improved(beta=1.0, epochs=100)

    # Save
    evaluator.save_results(result)

    print(f"\n{'='*60}")
    print("✓ Improved evaluation complete!")
    print(f"{'='*60}\n")

    print("Key findings:")
    print(f"  Epoch 1 loss: {result['epoch1_train_loss']:.2f}")
    print(f"  Final train loss: {result['final_train_loss']:.2f}")
    print(f"  Final val loss: {result['final_val_loss']:.2f}")
    print(f"  Overfitting? {'Yes' if result['final_train_loss'] < result['final_val_loss'] - 0.5 else 'No'}")

    print(f"\nBest temperature for diversity:")
    best_temp = max(result['temperature_results'].items(),
                    key=lambda x: x[1]['generation_diversity'])
    print(f"  T={best_temp[0]}: {best_temp[1]['generation_diversity']:.1%} diversity "
          f"({best_temp[1]['unique_combinations']} unique out of 1000)")
