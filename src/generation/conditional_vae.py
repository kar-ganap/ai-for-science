"""
Conditional VAE for MOF Generation

Key difference from standard VAE:
- Encoder: Takes (metal, linker, cell, CO2_uptake) → latent
- Decoder: Takes (latent, CO2_uptake) → (metal, linker, cell)
- Generation: Sample with target CO2 uptake → get MOFs optimized for that performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ConditionalMOF_VAE(nn.Module):
    """
    Conditional VAE for targeted MOF generation

    Conditions on CO2 uptake to generate MOFs with desired performance
    """

    def __init__(self,
                 n_metals: int,
                 n_linkers: int,
                 latent_dim: int = 16,
                 hidden_dim: int = 32):
        super(ConditionalMOF_VAE, self).__init__()

        self.n_metals = n_metals
        self.n_linkers = n_linkers
        self.latent_dim = latent_dim

        # Input: one-hot metal + one-hot linker + 4 cell params + 1 CO2 uptake
        input_dim = n_metals + n_linkers + 4 + 1  # +1 for CO2

        # Output: one-hot metal + one-hot linker + 4 cell params (no CO2)
        output_dim = n_metals + n_linkers + 4

        # Encoder: (structure + CO2) → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (latent + CO2) → structure
        # Concatenate latent + CO2 before decoding
        decoder_input_dim = latent_dim + 1  # latent + CO2

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, x):
        """Encode (structure + CO2) to latent distribution"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        """
        Decode latent + condition to structure

        Args:
            z: Latent code [batch_size, latent_dim]
            condition: CO2 uptake [batch_size, 1]
        """
        # Concatenate latent + condition
        decoder_input = torch.cat([z, condition], dim=1)
        return self.decoder(decoder_input)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [metal_onehot, linker_onehot, cell_params, CO2_uptake]

        Returns:
            recon: Reconstructed structure (no CO2)
            mu, logvar: Latent distribution parameters
        """
        # Encode full input (including CO2)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Extract CO2 condition from input
        co2_condition = x[:, -1:]  # Last column is CO2

        # Decode with condition
        recon = self.decode(z, co2_condition)

        return recon, mu, logvar

    def conditional_sample(self, n_samples: int = 1, target_co2: float = 5.0):
        """
        Sample MOFs conditioned on target CO2 uptake

        Args:
            n_samples: Number of samples to generate
            target_co2: Target CO2 uptake (mmol/g)

        Returns:
            Generated samples (structure only, no CO2)
        """
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim)

            # Create condition tensor (same CO2 for all samples)
            condition = torch.full((n_samples, 1), target_co2)

            # Decode
            samples = self.decode(z, condition)

        return samples


def conditional_vae_loss(recon_x, x_structure_only, mu, logvar, n_metals, n_linkers, beta=1.0):
    """
    Conditional VAE loss

    Same as standard VAE loss, but:
    - Input has CO2, output doesn't
    - Reconstruction only on structure components

    Args:
        recon_x: Reconstructed structure [batch, n_metals+n_linkers+4]
        x_structure_only: Target structure (no CO2) [batch, n_metals+n_linkers+4]
        mu, logvar: Latent distribution
        n_metals, n_linkers: Categorical dimensions
        beta: KL weight
    """
    # Split reconstruction
    metal_recon = recon_x[:, :n_metals]
    linker_recon = recon_x[:, n_metals:n_metals+n_linkers]
    cell_recon = recon_x[:, n_metals+n_linkers:]

    # Split target
    metal_target = x_structure_only[:, :n_metals]
    linker_target = x_structure_only[:, n_metals:n_metals+n_linkers]
    cell_target = x_structure_only[:, n_metals+n_linkers:]

    # Reconstruction loss
    metal_loss = F.cross_entropy(metal_recon, metal_target.argmax(dim=1), reduction='sum')
    linker_loss = F.cross_entropy(linker_recon, linker_target.argmax(dim=1), reduction='sum')
    cell_loss = F.mse_loss(cell_recon, cell_target, reduction='sum')

    recon_loss = metal_loss + linker_loss + 10.0 * cell_loss

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class ConditionalMOFGenerator:
    """
    MOF Generator using Conditional VAE
    """

    def __init__(self):
        self.cvae = None
        self.metal_encoder = None
        self.linker_encoder = None
        self.cell_mean = None
        self.cell_std = None
        self.co2_mean = None
        self.co2_std = None

        # Standard metals and linkers (from CRAFTED)
        self.metals = ['Zn', 'Cu', 'Fe', 'Al', 'Ca', 'Zr', 'Cr', 'Unknown']
        self.linkers = [
            'terephthalic acid',
            'trimesic acid',
            '2,6-naphthalenedicarboxylic acid',
            'biphenyl-4,4-dicarboxylic acid',
        ]

    def train_cvae(self, mof_data: pd.DataFrame,
                   epochs: int = 100,
                   batch_size: int = 32,
                   lr: float = 1e-3,
                   beta: float = 1.0,
                   augment: bool = True):
        """
        Train Conditional VAE on MOF dataset

        Args:
            mof_data: DataFrame with columns: metal, cell_a, cell_b, cell_c, volume, co2_uptake_mean
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            beta: KL divergence weight
            augment: Apply data augmentation
        """
        print(f"\n{'='*60}")
        print("Training Conditional MOF VAE")
        print(f"{'='*60}\n")

        # Augment data if requested
        if augment:
            from mof_augmentation import MOFAugmenter
            augmenter = MOFAugmenter()
            mof_data = augmenter.augment_dataset(
                mof_data,
                use_supercells=True,
                use_thermal_noise=True,
                noise_level=0.02
            )

        # Encode data (now includes CO2)
        X_full, X_structure, self.metal_encoder, self.linker_encoder = self._encode_mofs_with_co2(mof_data)

        # Create datasets
        dataset_full = torch.FloatTensor(X_full)  # Input (has CO2)
        dataset_structure = torch.FloatTensor(X_structure)  # Target (no CO2)

        combined_dataset = torch.utils.data.TensorDataset(dataset_full, dataset_structure)
        dataloader = torch.utils.data.DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize cVAE
        self.cvae = ConditionalMOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )

        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr)

        # Training loop
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for batch_full, batch_structure in dataloader:
                optimizer.zero_grad()

                recon, mu, logvar = self.cvae(batch_full)
                loss, recon_loss, kl_loss = conditional_vae_loss(
                    recon, batch_structure, mu, logvar,
                    self.cvae.n_metals, self.cvae.n_linkers,
                    beta=beta
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            if epoch == 0 or (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataset_full)
                avg_recon = total_recon / len(dataset_full)
                avg_kl = total_kl / len(dataset_full)
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.2f} (Recon={avg_recon:.2f}, KL={avg_kl:.2f})")

        print(f"\n✓ Conditional VAE training complete!")
        self.cvae.eval()

    def _encode_mofs_with_co2(self, mof_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Encode MOF data INCLUDING CO2 uptake

        Returns:
            X_full: [metal, linker, cell, CO2] - for input
            X_structure: [metal, linker, cell] - for target
            metal_encoder: {metal: index}
            linker_encoder: {linker: index}
        """
        # Create encoders
        unique_metals = mof_data['metal'].unique().tolist()
        metal_encoder = {m: i for i, m in enumerate(unique_metals)}
        linker_encoder = {l: i for i, l in enumerate(self.linkers)}

        # Metal-linker mapping (approximation)
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

        # One-hot encode linkers
        linker_onehot = np.zeros((n_samples, n_linkers))
        for i, metal in enumerate(mof_data['metal']):
            linker = metal_linker_map.get(metal, 'terephthalic acid')
            linker_onehot[i, linker_encoder[linker]] = 1

        # Normalize cell parameters
        cell_params = mof_data[['cell_a', 'cell_b', 'cell_c', 'volume']].values
        self.cell_mean = cell_params.mean(axis=0)
        self.cell_std = cell_params.std(axis=0)
        cell_normalized = (cell_params - self.cell_mean) / (self.cell_std + 1e-8)

        # Normalize CO2 uptake
        co2_values = mof_data['co2_uptake_mean'].values.reshape(-1, 1)
        self.co2_mean = co2_values.mean()
        self.co2_std = co2_values.std()
        co2_normalized = (co2_values - self.co2_mean) / (self.co2_std + 1e-8)

        # Concatenate
        X_structure = np.concatenate([metal_onehot, linker_onehot, cell_normalized], axis=1)
        X_full = np.concatenate([X_structure, co2_normalized], axis=1)

        print(f"  ✓ Encoded {n_samples} MOFs with CO2 uptake")
        print(f"  ✓ Metals: {list(metal_encoder.keys())}")
        print(f"  ✓ Linkers: {list(linker_encoder.keys())}")
        print(f"  ✓ CO2 range: {co2_values.min():.2f} - {co2_values.max():.2f} mmol/g")
        print(f"  ✓ Input dim: {X_full.shape[1]} (includes CO2)")
        print(f"  ✓ Output dim: {X_structure.shape[1]} (structure only)")

        return X_full, X_structure, metal_encoder, linker_encoder

    def generate_candidates(self,
                          n_candidates: int = 100,
                          target_co2: float = 5.0,
                          temperature: float = 1.0) -> List[Dict]:
        """
        Generate MOF candidates conditioned on target CO2 uptake

        Args:
            n_candidates: Number to generate
            target_co2: Target CO2 uptake (mmol/g)
            temperature: Sampling temperature for diversity

        Returns:
            List of candidate MOF compositions
        """
        if self.cvae is None:
            raise ValueError("cVAE not trained! Call train_cvae() first.")

        # Denormalize target CO2
        target_co2_normalized = (target_co2 - self.co2_mean) / (self.co2_std + 1e-8)

        candidates = []
        print(f"\nGenerating {n_candidates} MOF candidates for CO2 = {target_co2:.2f} mmol/g...")

        with torch.no_grad():
            # Generate in one batch
            samples = self.cvae.conditional_sample(
                n_samples=n_candidates,
                target_co2=float(target_co2_normalized)
            )

            for i in range(n_candidates):
                candidate = self._decode_mof(samples[i].numpy(), temperature)
                if candidate:
                    candidate['target_co2'] = target_co2
                    candidates.append(candidate)

        print(f"  ✓ Generated {len(candidates)} valid candidates")
        return candidates

    def _decode_mof(self, output: np.ndarray, temperature: float = 1.0) -> Optional[Dict]:
        """Decode cVAE output to MOF composition with temperature sampling"""
        n_metals = len(self.metal_encoder)
        n_linkers = len(self.linker_encoder)

        # Temperature sampling for categorical variables
        def sample_categorical(logits, temp):
            if temp <= 0:
                return np.argmax(logits)
            scaled = logits / temp
            probs = np.exp(scaled - np.max(scaled))
            probs = probs / probs.sum()
            return np.random.choice(len(probs), p=probs)

        # Decode metal
        metal_logits = output[:n_metals]
        metal_idx = sample_categorical(metal_logits, temperature)
        metal = list(self.metal_encoder.keys())[metal_idx]

        # Decode linker
        linker_logits = output[n_metals:n_metals+n_linkers]
        linker_idx = sample_categorical(linker_logits, temperature)
        linker = list(self.linker_encoder.keys())[linker_idx]

        # Decode cell params
        cell_normalized = output[n_metals+n_linkers:]
        cell_params = cell_normalized * self.cell_std + self.cell_mean

        # Validate
        if metal == 'Unknown':
            return None
        if any(cell_params < 0):
            return None

        return {
            'metal': metal,
            'linker': linker,
            'cell_a': float(cell_params[0]),
            'cell_b': float(cell_params[1]),
            'cell_c': float(cell_params[2]),
            'volume': float(cell_params[3])
        }

    def save(self, path: Path):
        """Save trained cVAE"""
        if self.cvae is None:
            raise ValueError("No cVAE to save")

        torch.save({
            'cvae_state': self.cvae.state_dict(),
            'metal_encoder': self.metal_encoder,
            'linker_encoder': self.linker_encoder,
            'cell_mean': self.cell_mean,
            'cell_std': self.cell_std,
            'co2_mean': self.co2_mean,
            'co2_std': self.co2_std,
        }, path)
        print(f"✓ Conditional VAE saved to {path}")

    def load(self, path: Path):
        """Load trained cVAE"""
        checkpoint = torch.load(path)

        self.metal_encoder = checkpoint['metal_encoder']
        self.linker_encoder = checkpoint['linker_encoder']
        self.cell_mean = checkpoint['cell_mean']
        self.cell_std = checkpoint['cell_std']
        self.co2_mean = checkpoint['co2_mean']
        self.co2_std = checkpoint['co2_std']

        self.cvae = ConditionalMOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=32
        )
        self.cvae.load_state_dict(checkpoint['cvae_state'])
        self.cvae.eval()

        print(f"✓ Conditional VAE loaded from {path}")


if __name__ == '__main__':
    print("Testing Conditional MOF VAE\n" + "="*60)

    # Load CRAFTED MOF data
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        exit(1)

    mof_data = pd.read_csv(mof_file)
    print(f"✓ Loaded {len(mof_data)} MOFs with CO2 uptake data\n")

    # Check CO2 range
    print(f"CO2 uptake range: {mof_data['co2_uptake_mean'].min():.2f} - {mof_data['co2_uptake_mean'].max():.2f} mmol/g")
    print(f"CO2 uptake mean: {mof_data['co2_uptake_mean'].mean():.2f} mmol/g\n")

    # Initialize generator
    generator = ConditionalMOFGenerator()

    # Train cVAE
    generator.train_cvae(mof_data, epochs=100, batch_size=32, beta=1.0)

    # Test conditional generation at different CO2 targets
    print(f"\n{'='*60}")
    print("Testing Conditional Generation")
    print(f"{'='*60}\n")

    co2_targets = [2.0, 4.0, 6.0, 8.0]  # Low to high performance

    for target in co2_targets:
        candidates = generator.generate_candidates(
            n_candidates=50,
            target_co2=target,
            temperature=2.0
        )

        print(f"\nTarget CO2 = {target:.1f} mmol/g:")
        print(f"  Generated {len(candidates)} candidates")

        # Check diversity
        unique_combos = set((c['metal'], c['linker']) for c in candidates)
        print(f"  Unique (metal, linker): {len(unique_combos)}")

        # Show top combinations
        from collections import Counter
        combo_counts = Counter((c['metal'], c['linker']) for c in candidates)
        print(f"  Top 3:")
        for (metal, linker), count in combo_counts.most_common(3):
            print(f"    {metal} + {linker}: {count} samples")

    # Save model
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    generator.save(model_dir / "conditional_mof_vae.pt")

    print(f"\n✓ Conditional MOF VAE test complete!")
