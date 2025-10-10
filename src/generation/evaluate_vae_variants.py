"""
VAE Variants Evaluation

Train and compare:
1. Simple VAE (composition + cell params only)
2. Hybrid VAE (composition + cell params + geometric features)

Multiple β values tested for each variant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class HybridMOF_VAE(nn.Module):
    """
    Hybrid VAE that includes geometric features

    Encodes: Metal + Linker + Cell params + Geometric features
    Latent space: Continuous (Gaussian)
    Decodes: Back to metal + linker + cell params (no need to decode geometric features)
    """

    def __init__(self,
                 n_metals: int,
                 n_linkers: int,
                 n_geom_features: int = 11,
                 latent_dim: int = 16,
                 hidden_dim: int = 64):
        """
        Initialize Hybrid VAE

        Args:
            n_metals: Number of unique metal types
            n_linkers: Number of unique linker types
            n_geom_features: Number of geometric features
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer size (larger for hybrid)
        """
        super(HybridMOF_VAE, self).__init__()

        self.n_metals = n_metals
        self.n_linkers = n_linkers
        self.n_geom_features = n_geom_features
        self.latent_dim = latent_dim

        # Input: one-hot metal + one-hot linker + 4 cell params + geometric features
        input_dim = n_metals + n_linkers + 4 + n_geom_features

        # Output: one-hot metal + one-hot linker + 4 cell params (no geometric features in output)
        output_dim = n_metals + n_linkers + 4

        # Encoder (larger for hybrid)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, x):
        """Encode input to latent distribution"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to output"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, n_samples: int = 1):
        """Sample from latent space"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            samples = self.decode(z)
        return samples


def hybrid_vae_loss(recon_x, x_output_only, mu, logvar, n_metals, n_linkers, beta=1.0):
    """
    Hybrid VAE loss (same as simple VAE, but input has extra features)

    Args:
        recon_x: Reconstructed output (metal + linker + cell)
        x_output_only: Target output (metal + linker + cell, no geometric features)
        mu, logvar: Latent distribution parameters
        n_metals, n_linkers: Categorical dimensions
        beta: KL weight
    """
    # Split reconstruction
    metal_recon = recon_x[:, :n_metals]
    linker_recon = recon_x[:, n_metals:n_metals+n_linkers]
    cell_recon = recon_x[:, n_metals+n_linkers:]

    # Split target
    metal_target = x_output_only[:, :n_metals]
    linker_target = x_output_only[:, n_metals:n_metals+n_linkers]
    cell_target = x_output_only[:, n_metals+n_linkers:]

    # Reconstruction loss
    metal_loss = F.cross_entropy(metal_recon, metal_target.argmax(dim=1), reduction='sum')
    linker_loss = F.cross_entropy(linker_recon, linker_target.argmax(dim=1), reduction='sum')
    cell_loss = F.mse_loss(cell_recon, cell_target, reduction='sum')

    recon_loss = metal_loss + linker_loss + 10.0 * cell_loss

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class VAEEvaluator:
    """
    Evaluate and compare VAE variants
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = []

    def load_data(self):
        """Load MOF data and geometric features"""
        # Load augmented MOF data
        mof_file = self.project_root / "data" / "processed" / "crafted_mofs_augmented.csv"
        geom_file = self.project_root / "data" / "processed" / "crafted_geometric_features.csv"

        self.mof_data = pd.read_csv(mof_file)
        self.geom_features = pd.read_csv(geom_file)

        print(f"✓ Loaded {len(self.mof_data)} augmented MOF samples")
        print(f"✓ Loaded {len(self.geom_features)} geometric feature sets")

    def prepare_simple_vae_data(self):
        """Prepare data for simple VAE (composition + cell params only)"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from mof_vae import MOFGenerator

        generator = MOFGenerator()
        X, metal_encoder, linker_encoder = generator._encode_mofs(self.mof_data)

        return X, metal_encoder, linker_encoder, generator.cell_mean, generator.cell_std

    def prepare_hybrid_vae_data(self):
        """
        Prepare data for hybrid VAE (composition + cell params + geometric features)

        Match augmented MOFs to their original geometric features (based on mof_id)
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from mof_vae import MOFGenerator

        # First get simple encoding
        generator = MOFGenerator()
        X_simple, metal_encoder, linker_encoder = generator._encode_mofs(self.mof_data)

        # Now add geometric features
        # Map mof_id to geometric features
        geom_dict = {}
        for _, row in self.geom_features.iterrows():
            geom_dict[row['mof_id']] = row.drop('mof_id').values

        # Geometric feature columns (excluding mof_id)
        geom_feature_cols = [c for c in self.geom_features.columns if c != 'mof_id']

        # Match augmented MOFs to original geometric features
        geom_features_matched = []
        for _, mof in self.mof_data.iterrows():
            mof_id = mof['mof_id']
            if mof_id in geom_dict:
                geom_features_matched.append(geom_dict[mof_id])
            else:
                # Fallback: use mean features
                geom_features_matched.append(self.geom_features[geom_feature_cols].mean().values)

        geom_array = np.array(geom_features_matched, dtype=np.float64)

        # Normalize geometric features
        geom_mean = np.mean(geom_array, axis=0)
        geom_std = np.std(geom_array, axis=0) + 1e-8
        geom_normalized = (geom_array - geom_mean) / geom_std

        # Concatenate: [metal_onehot, linker_onehot, cell_normalized, geom_normalized]
        X_hybrid = np.concatenate([X_simple, geom_normalized], axis=1)

        print(f"  ✓ Simple VAE input dim: {X_simple.shape[1]}")
        print(f"  ✓ Hybrid VAE input dim: {X_hybrid.shape[1]} (+{geom_normalized.shape[1]} geom features)")

        return X_hybrid, X_simple, metal_encoder, linker_encoder, generator.cell_mean, generator.cell_std, geom_mean, geom_std

    def train_simple_vae(self, beta: float, epochs: int = 100, batch_size: int = 32):
        """Train simple VAE with given beta"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from mof_vae import MOF_VAE, vae_loss

        print(f"\n{'='*60}")
        print(f"Training Simple VAE (β={beta})")
        print(f"{'='*60}\n")

        # Prepare data
        X, metal_encoder, linker_encoder, cell_mean, cell_std = self.prepare_simple_vae_data()

        dataset = torch.FloatTensor(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        vae = MOF_VAE(
            n_metals=len(metal_encoder),
            n_linkers=len(linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training loop
        losses = {'total': [], 'recon': [], 'kl': []}

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for batch in dataloader:
                optimizer.zero_grad()

                recon, mu, logvar = vae(batch)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, batch, mu, logvar,
                    vae.n_metals, vae.n_linkers,
                    beta=beta
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            # Track losses
            losses['total'].append(epoch_loss / len(dataset))
            losses['recon'].append(epoch_recon / len(dataset))
            losses['kl'].append(epoch_kl / len(dataset))

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={losses['total'][-1]:.2f} "
                      f"(Recon={losses['recon'][-1]:.2f}, KL={losses['kl'][-1]:.2f})")

        # Evaluate
        metrics = self.evaluate_vae(vae, dataset, is_hybrid=False)

        result = {
            'type': 'simple',
            'beta': beta,
            'epochs': epochs,
            'final_loss': losses['total'][-1],
            'final_recon': losses['recon'][-1],
            'final_kl': losses['kl'][-1],
            'metrics': metrics,
            'model': vae,
            'losses': losses
        }

        self.results.append(result)

        print(f"\n✓ Simple VAE (β={beta}) training complete!")
        print(f"  Final loss: {losses['total'][-1]:.2f}")
        print(f"  Reconstruction quality: {metrics['recon_accuracy']:.1%}")
        print(f"  Latent coverage: {metrics['latent_coverage']:.2f}")

        return result

    def train_hybrid_vae(self, beta: float, epochs: int = 100, batch_size: int = 32):
        """Train hybrid VAE with given beta"""
        print(f"\n{'='*60}")
        print(f"Training Hybrid VAE (β={beta})")
        print(f"{'='*60}\n")

        # Prepare data
        X_hybrid, X_simple, metal_encoder, linker_encoder, cell_mean, cell_std, geom_mean, geom_std = self.prepare_hybrid_vae_data()

        dataset_input = torch.FloatTensor(X_hybrid)
        dataset_output = torch.FloatTensor(X_simple)

        # Create dataloader with paired inputs/outputs
        combined_dataset = torch.utils.data.TensorDataset(dataset_input, dataset_output)
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # Initialize hybrid model
        vae = HybridMOF_VAE(
            n_metals=len(metal_encoder),
            n_linkers=len(linker_encoder),
            n_geom_features=11,
            latent_dim=16,
            hidden_dim=64
        )

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training loop
        losses = {'total': [], 'recon': [], 'kl': []}

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for batch_input, batch_output in dataloader:
                optimizer.zero_grad()

                recon, mu, logvar = vae(batch_input)
                loss, recon_loss, kl_loss = hybrid_vae_loss(
                    recon, batch_output, mu, logvar,
                    vae.n_metals, vae.n_linkers,
                    beta=beta
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            # Track losses
            losses['total'].append(epoch_loss / len(dataset_input))
            losses['recon'].append(epoch_recon / len(dataset_input))
            losses['kl'].append(epoch_kl / len(dataset_input))

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={losses['total'][-1]:.2f} "
                      f"(Recon={losses['recon'][-1]:.2f}, KL={losses['kl'][-1]:.2f})")

        # Evaluate
        metrics = self.evaluate_vae(vae, dataset_input, is_hybrid=True, output_data=dataset_output)

        result = {
            'type': 'hybrid',
            'beta': beta,
            'epochs': epochs,
            'final_loss': losses['total'][-1],
            'final_recon': losses['recon'][-1],
            'final_kl': losses['kl'][-1],
            'metrics': metrics,
            'model': vae,
            'losses': losses
        }

        self.results.append(result)

        print(f"\n✓ Hybrid VAE (β={beta}) training complete!")
        print(f"  Final loss: {losses['total'][-1]:.2f}")
        print(f"  Reconstruction quality: {metrics['recon_accuracy']:.1%}")
        print(f"  Latent coverage: {metrics['latent_coverage']:.2f}")

        return result

    def evaluate_vae(self, vae, dataset, is_hybrid=False, output_data=None):
        """
        Evaluate VAE quality

        Metrics:
        1. Reconstruction accuracy (metal + linker correct)
        2. Cell param reconstruction error (RMSE)
        3. Latent space coverage (effective dimensionality)
        4. Generation diversity (unique samples in 1000 generations)
        """
        vae.eval()

        with torch.no_grad():
            # 1. Reconstruction accuracy
            if is_hybrid:
                recon, mu, logvar = vae(dataset)
                target = output_data
            else:
                recon, mu, logvar = vae(dataset)
                target = dataset

            # Metal accuracy
            metal_pred = recon[:, :vae.n_metals].argmax(dim=1)
            metal_true = target[:, :vae.n_metals].argmax(dim=1)
            metal_acc = (metal_pred == metal_true).float().mean().item()

            # Linker accuracy
            linker_pred = recon[:, vae.n_metals:vae.n_metals+vae.n_linkers].argmax(dim=1)
            linker_true = target[:, vae.n_metals:vae.n_metals+vae.n_linkers].argmax(dim=1)
            linker_acc = (linker_pred == linker_true).float().mean().item()

            # Combined accuracy
            recon_acc = (metal_acc + linker_acc) / 2

            # 2. Cell param RMSE
            cell_pred = recon[:, vae.n_metals+vae.n_linkers:]
            cell_true = target[:, vae.n_metals+vae.n_linkers:]
            cell_rmse = torch.sqrt(F.mse_loss(cell_pred, cell_true)).item()

            # 3. Latent space coverage (variance explained)
            latent_vars = logvar.exp().mean(dim=0)
            effective_dim = (latent_vars.sum() ** 2) / (latent_vars ** 2).sum()
            latent_coverage = effective_dim.item() / vae.latent_dim

            # 4. Generation diversity
            samples = vae.sample(n_samples=1000)
            metal_samples = samples[:, :vae.n_metals].argmax(dim=1)
            linker_samples = samples[:, vae.n_metals:vae.n_metals+vae.n_linkers].argmax(dim=1)

            # Count unique (metal, linker) combinations
            combinations = set(zip(metal_samples.tolist(), linker_samples.tolist()))
            diversity = len(combinations) / 1000

        return {
            'recon_accuracy': recon_acc,
            'metal_accuracy': metal_acc,
            'linker_accuracy': linker_acc,
            'cell_rmse': cell_rmse,
            'latent_coverage': latent_coverage,
            'generation_diversity': diversity
        }

    def run_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation of all VAE variants

        Simple VAE: β = [1.0, 3.0, 5.0]
        Hybrid VAE: β = [1.0, 3.0, 5.0]
        """
        print(f"\n{'='*60}")
        print("Comprehensive VAE Evaluation")
        print(f"{'='*60}\n")

        # Load data
        self.load_data()

        # Test β values
        betas = [1.0, 3.0, 5.0]

        # Train simple VAEs
        print("\n" + "="*60)
        print("SIMPLE VAE VARIANTS")
        print("="*60)

        for beta in betas:
            self.train_simple_vae(beta=beta, epochs=100)

        # Train hybrid VAEs
        print("\n" + "="*60)
        print("HYBRID VAE VARIANTS")
        print("="*60)

        for beta in betas:
            self.train_hybrid_vae(beta=beta, epochs=100)

        # Summary
        self.print_summary()

        # Save results
        self.save_results()

        return self.results

    def print_summary(self):
        """Print summary comparison"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}\n")

        # Create summary table
        print(f"{'Type':<8} {'β':<6} {'Loss':<8} {'Recon Acc':<12} {'Cell RMSE':<12} {'Diversity':<10}")
        print("-" * 60)

        for result in self.results:
            print(f"{result['type']:<8} "
                  f"{result['beta']:<6.1f} "
                  f"{result['final_loss']:<8.2f} "
                  f"{result['metrics']['recon_accuracy']:<12.1%} "
                  f"{result['metrics']['cell_rmse']:<12.3f} "
                  f"{result['metrics']['generation_diversity']:<10.1%}")

        # Best model
        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60 + "\n")

        # Score: High recon accuracy + Low cell RMSE + High diversity
        for result in self.results:
            result['score'] = (
                result['metrics']['recon_accuracy'] * 100 +
                (1 - result['metrics']['cell_rmse']) * 10 +
                result['metrics']['generation_diversity'] * 50 -
                result['final_loss'] * 0.1
            )

        best = max(self.results, key=lambda x: x['score'])

        print(f"🏆 Best: {best['type'].upper()} VAE with β={best['beta']}")
        print(f"   Reconstruction accuracy: {best['metrics']['recon_accuracy']:.1%}")
        print(f"   Cell RMSE: {best['metrics']['cell_rmse']:.3f}")
        print(f"   Generation diversity: {best['metrics']['generation_diversity']:.1%}")
        print(f"   Latent coverage: {best['metrics']['latent_coverage']:.2f}")
        print(f"   Overall score: {best['score']:.2f}")

    def save_results(self):
        """Save evaluation results"""
        results_dir = self.project_root / "results" / "vae_evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary JSON
        summary = []
        for result in self.results:
            summary.append({
                'type': result['type'],
                'beta': result['beta'],
                'epochs': result['epochs'],
                'final_loss': result['final_loss'],
                'final_recon': result['final_recon'],
                'final_kl': result['final_kl'],
                'metrics': result['metrics'],
                'score': result['score']
            })

        summary_file = results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Results saved to: {summary_file}")

        # Save best model
        best = max(self.results, key=lambda x: x['score'])
        model_file = results_dir / f"best_vae_{best['type']}_beta{best['beta']}_{timestamp}.pt"

        torch.save({
            'model_state': best['model'].state_dict(),
            'type': best['type'],
            'beta': best['beta'],
            'metrics': best['metrics'],
            'score': best['score']
        }, model_file)

        print(f"✓ Best model saved to: {model_file}")


if __name__ == '__main__':
    print("VAE Variants Evaluation\n" + "="*60)

    # Setup
    project_root = Path(__file__).parents[2]
    evaluator = VAEEvaluator(project_root)

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()

    print(f"\n{'='*60}")
    print("✓ Comprehensive evaluation complete!")
    print(f"{'='*60}\n")
