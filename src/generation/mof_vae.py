"""
Tiny VAE for MOF Generation

Learns distribution of MOF compositions (metal + linker + structure)
Generates new candidates by sampling latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MOF_VAE(nn.Module):
    """
    Variational Autoencoder for MOF generation

    Encodes: Metal (categorical) + Linker (categorical) + Cell params (continuous)
    Latent space: Continuous (Gaussian)
    Decodes: Back to metal + linker + cell params
    """

    def __init__(self,
                 n_metals: int,
                 n_linkers: int,
                 latent_dim: int = 16,
                 hidden_dim: int = 32):
        """
        Initialize VAE

        Args:
            n_metals: Number of unique metal types
            n_linkers: Number of unique linker types
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer size
        """
        super(MOF_VAE, self).__init__()

        self.n_metals = n_metals
        self.n_linkers = n_linkers
        self.latent_dim = latent_dim

        # Input: one-hot metal + one-hot linker + 4 cell params
        input_dim = n_metals + n_linkers + 4

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
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


def vae_loss(recon_x, x, mu, logvar,
             n_metals, n_linkers,
             beta=1.0):
    """
    VAE loss = Reconstruction + KL divergence

    Reconstruction:
    - Cross-entropy for metal (categorical)
    - Cross-entropy for linker (categorical)
    - MSE for cell params (continuous)

    KL divergence: Standard Gaussian prior
    """
    # Split reconstruction into components
    metal_recon = recon_x[:, :n_metals]
    linker_recon = recon_x[:, n_metals:n_metals+n_linkers]
    cell_recon = recon_x[:, n_metals+n_linkers:]

    # Split target
    metal_target = x[:, :n_metals]
    linker_target = x[:, n_metals:n_metals+n_linkers]
    cell_target = x[:, n_metals+n_linkers:]

    # Reconstruction loss
    metal_loss = F.cross_entropy(metal_recon, metal_target.argmax(dim=1), reduction='sum')
    linker_loss = F.cross_entropy(linker_recon, linker_target.argmax(dim=1), reduction='sum')
    cell_loss = F.mse_loss(cell_recon, cell_target, reduction='sum')

    recon_loss = metal_loss + linker_loss + 10.0 * cell_loss  # Weight cell params more

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class MOFGenerator:
    """
    MOF Generator using VAE
    """

    def __init__(self, cost_estimator=None):
        self.cost_estimator = cost_estimator
        self.vae = None
        self.metal_encoder = None
        self.linker_encoder = None
        self.cell_mean = None
        self.cell_std = None

        # Common metals and linkers (from CRAFTED)
        self.metals = ['Zn', 'Cu', 'Fe', 'Al', 'Ca', 'Zr', 'Cr', 'Unknown']
        self.linkers = [
            'terephthalic acid',       # BDC
            'trimesic acid',            # BTC
            '2,6-naphthalenedicarboxylic acid',  # NDC
            'biphenyl-4,4-dicarboxylic acid',     # BPDC
        ]

    def train_vae(self, mof_data: pd.DataFrame,
                  epochs: int = 100,
                  batch_size: int = 32,
                  lr: float = 1e-3,
                  beta: float = 1.0,
                  augment: bool = True):
        """
        Train VAE on MOF dataset

        Args:
            mof_data: DataFrame with columns: metal, cell_a, cell_b, cell_c, volume
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            beta: KL divergence weight (beta-VAE)
            augment: Apply data augmentation (supercells + thermal noise)
        """
        print(f"\n{'='*60}")
        print("Training MOF VAE")
        print(f"{'='*60}\n")

        # Augment data if requested
        if augment:
            from .mof_augmentation import MOFAugmenter
            augmenter = MOFAugmenter()
            mof_data = augmenter.augment_dataset(
                mof_data,
                use_supercells=True,
                use_thermal_noise=True,
                noise_level=0.02
            )

        # Encode data
        X, self.metal_encoder, self.linker_encoder = self._encode_mofs(mof_data)

        # Create dataset
        dataset = torch.FloatTensor(X)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize VAE
        self.vae = MOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        # Training loop
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for batch in dataloader:
                optimizer.zero_grad()

                recon, mu, logvar = self.vae(batch)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, batch, mu, logvar,
                    self.vae.n_metals, self.vae.n_linkers,
                    beta=beta
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataset)
                avg_recon = total_recon / len(dataset)
                avg_kl = total_kl / len(dataset)
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.2f} (Recon={avg_recon:.2f}, KL={avg_kl:.2f})")

        print(f"\n✓ VAE training complete!")
        self.vae.eval()

    def _encode_mofs(self, mof_data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Encode MOF data for VAE

        Returns:
            X: Encoded features [n_samples, n_features]
            metal_encoder: {metal: index}
            linker_encoder: {linker: index}
        """
        # Create encoders
        unique_metals = mof_data['metal'].unique().tolist()
        metal_encoder = {m: i for i, m in enumerate(unique_metals)}

        # Use standard linkers (may not be in dataset)
        linker_encoder = {l: i for i, l in enumerate(self.linkers)}

        # Assign linkers based on metal (approximation)
        metal_linker_map = {
            'Zn': 'terephthalic acid',
            'Cu': 'trimesic acid',
            'Fe': 'terephthalic acid',
            'Ca': 'terephthalic acid',
            'Al': 'terephthalic acid',
            'Zr': 'terephthalic acid',
            'Cr': 'terephthalic acid',
            'Unknown': 'terephthalic acid'
        }

        n_samples = len(mof_data)
        n_metals = len(metal_encoder)
        n_linkers = len(linker_encoder)

        # One-hot encode metals
        metal_onehot = np.zeros((n_samples, n_metals))
        for i, metal in enumerate(mof_data['metal']):
            metal_onehot[i, metal_encoder[metal]] = 1

        # One-hot encode linkers (approximated)
        linker_onehot = np.zeros((n_samples, n_linkers))
        for i, metal in enumerate(mof_data['metal']):
            linker = metal_linker_map.get(metal, 'terephthalic acid')
            linker_onehot[i, linker_encoder[linker]] = 1

        # Normalize cell parameters
        cell_params = mof_data[['cell_a', 'cell_b', 'cell_c', 'volume']].values
        self.cell_mean = cell_params.mean(axis=0)
        self.cell_std = cell_params.std(axis=0)
        cell_normalized = (cell_params - self.cell_mean) / (self.cell_std + 1e-8)

        # Concatenate
        X = np.concatenate([metal_onehot, linker_onehot, cell_normalized], axis=1)

        print(f"  ✓ Encoded {n_samples} MOFs")
        print(f"  ✓ Metals: {list(metal_encoder.keys())}")
        print(f"  ✓ Linkers: {list(linker_encoder.keys())}")
        print(f"  ✓ Feature dim: {X.shape[1]}")

        return X, metal_encoder, linker_encoder

    def generate_candidates(self,
                          n_candidates: int = 100,
                          max_cost: float = 2.0,
                          temperature: float = 1.0) -> List[Dict]:
        """
        Generate MOF candidates using VAE

        Args:
            n_candidates: Number to generate
            max_cost: Maximum synthesis cost ($/g)
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            List of candidate MOF compositions
        """
        if self.vae is None:
            raise ValueError("VAE not trained! Call train_vae() first.")

        candidates = []
        attempts = 0
        max_attempts = n_candidates * 10  # Try up to 10x

        print(f"\nGenerating {n_candidates} MOF candidates...")

        with torch.no_grad():
            while len(candidates) < n_candidates and attempts < max_attempts:
                attempts += 1

                # Sample latent space
                z = torch.randn(1, self.vae.latent_dim) * temperature
                output = self.vae.decode(z).numpy()[0]

                # Decode
                candidate = self._decode_mof(output)

                # Check cost if estimator available
                if self.cost_estimator and candidate:
                    cost_data = self.cost_estimator.estimate_synthesis_cost(candidate)
                    if cost_data['total_cost_per_gram'] > max_cost:
                        continue  # Too expensive
                    candidate['estimated_cost'] = cost_data['total_cost_per_gram']

                if candidate:
                    candidates.append(candidate)

        print(f"  ✓ Generated {len(candidates)} candidates (tried {attempts})")
        return candidates

    def _decode_mof(self, output: np.ndarray) -> Optional[Dict]:
        """
        Decode VAE output to MOF composition

        Args:
            output: VAE decoder output

        Returns:
            MOF composition dict or None if invalid
        """
        n_metals = len(self.metal_encoder)
        n_linkers = len(self.linker_encoder)

        # Decode metal (argmax of logits)
        metal_logits = output[:n_metals]
        metal_idx = np.argmax(metal_logits)
        metal = list(self.metal_encoder.keys())[metal_idx]

        # Decode linker (argmax of logits)
        linker_logits = output[n_metals:n_metals+n_linkers]
        linker_idx = np.argmax(linker_logits)
        linker = list(self.linker_encoder.keys())[linker_idx]

        # Decode cell params (denormalize)
        cell_normalized = output[n_metals+n_linkers:]
        cell_params = cell_normalized * self.cell_std + self.cell_mean

        # Validate
        if metal == 'Unknown':
            return None  # Skip unknown metals

        if any(cell_params < 0):
            return None  # Invalid cell params

        return {
            'metal': metal,
            'linker': linker,
            'cell_a': float(cell_params[0]),
            'cell_b': float(cell_params[1]),
            'cell_c': float(cell_params[2]),
            'volume': float(cell_params[3])
        }

    def save(self, path: Path):
        """Save trained VAE"""
        if self.vae is None:
            raise ValueError("No VAE to save")

        torch.save({
            'vae_state': self.vae.state_dict(),
            'metal_encoder': self.metal_encoder,
            'linker_encoder': self.linker_encoder,
            'cell_mean': self.cell_mean,
            'cell_std': self.cell_std,
        }, path)
        print(f"✓ VAE saved to {path}")

    def load(self, path: Path):
        """Load trained VAE"""
        checkpoint = torch.load(path)

        self.metal_encoder = checkpoint['metal_encoder']
        self.linker_encoder = checkpoint['linker_encoder']
        self.cell_mean = checkpoint['cell_mean']
        self.cell_std = checkpoint['cell_std']

        self.vae = MOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )
        self.vae.load_state_dict(checkpoint['vae_state'])
        self.vae.eval()

        print(f"✓ VAE loaded from {path}")


if __name__ == '__main__':
    # Test VAE training
    print("Testing MOF VAE\n" + "="*60)

    # Load CRAFTED MOF data
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        exit(1)

    mof_data = pd.read_csv(mof_file)
    print(f"✓ Loaded {len(mof_data)} MOFs\n")

    # Initialize generator
    generator = MOFGenerator()

    # Train VAE
    generator.train_vae(mof_data, epochs=50, batch_size=32)

    # Generate candidates
    candidates = generator.generate_candidates(n_candidates=20)

    print(f"\n{'='*60}")
    print("Sample Generated MOFs:")
    print(f"{'='*60}\n")

    for i, mof in enumerate(candidates[:5], 1):
        print(f"{i}. {mof['metal']}-based MOF")
        print(f"   Linker: {mof['linker']}")
        print(f"   Cell: a={mof['cell_a']:.2f}, b={mof['cell_b']:.2f}, c={mof['cell_c']:.2f}")
        print(f"   Volume: {mof['volume']:.2f} Ų")
        if 'estimated_cost' in mof:
            print(f"   Est. cost: ${mof['estimated_cost']:.2f}/g")
        print()

    # Save model
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    generator.save(model_dir / "mof_vae.pt")

    print("✓ MOF VAE test complete!")
