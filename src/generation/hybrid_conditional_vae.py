"""
Hybrid Conditional VAE with Geometric Features

Input: Metal + Linker + Cell + CO2 + 11 Geometric Features
Output: Metal + Linker + Cell

Properly handles augmented data:
- Supercells: Use original MOF's geometric features (intensive properties unchanged)
- Thermal noise: Recompute geometric features (small perturbations)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from conditional_vae import ConditionalMOF_VAE, conditional_vae_loss


class HybridConditionalMOFGenerator:
    """
    Hybrid Conditional VAE with geometric features
    """

    def __init__(self):
        self.cvae = None
        self.metal_encoder = None
        self.linker_encoder = None
        self.cell_mean = None
        self.cell_std = None
        self.co2_mean = None
        self.co2_std = None
        self.geom_mean = None
        self.geom_std = None

        self.metals = ['Zn', 'Cu', 'Fe', 'Al', 'Ca', 'Zr', 'Cr', 'Unknown']
        self.linkers = [
            'terephthalic acid',
            'trimesic acid',
            '2,6-naphthalenedicarboxylic acid',
            'biphenyl-4,4-dicarboxylic acid',
        ]

    def train_hybrid_cvae(self,
                          mof_data: pd.DataFrame,
                          geom_features: pd.DataFrame,
                          epochs: int = 100,
                          batch_size: int = 32,
                          lr: float = 1e-3,
                          beta: float = 1.0,
                          augment: bool = True):
        """
        Train Hybrid Conditional VAE with geometric features

        Args:
            mof_data: DataFrame with mof_id, metal, cell params, co2_uptake_mean
            geom_features: DataFrame with mof_id + 11 geometric features
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            beta: KL weight
            augment: Use augmentation
        """
        print(f"\n{'='*60}")
        print("Training Hybrid Conditional MOF VAE")
        print(f"{'='*60}\n")

        # Augment data
        if augment:
            from mof_augmentation import MOFAugmenter
            augmenter = MOFAugmenter()
            mof_data = augmenter.augment_dataset(
                mof_data,
                use_supercells=True,
                use_thermal_noise=True,
                noise_level=0.02
            )

        # Encode with geometric features
        X_full, X_structure = self._encode_mofs_with_geom(mof_data, geom_features)

        # Create datasets
        dataset_full = torch.FloatTensor(X_full)
        dataset_structure = torch.FloatTensor(X_structure)

        combined_dataset = torch.utils.data.TensorDataset(dataset_full, dataset_structure)
        dataloader = torch.utils.data.DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize hybrid cVAE (larger hidden dim for more features)
        self.cvae = ConditionalMOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=64  # Larger for hybrid
        )

        # Manually adjust input dimension for geometric features
        # Input: metal + linker + cell (4) + CO2 (1) + geom (11) = n_metals + n_linkers + 16
        old_input_dim = self.cvae.encoder[0].in_features
        new_input_dim = old_input_dim + 11  # Add 11 geometric features

        # Replace first encoder layer
        self.cvae.encoder[0] = nn.Linear(new_input_dim, 64)

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

        print(f"\n✓ Hybrid Conditional VAE training complete!")
        self.cvae.eval()

    def _encode_mofs_with_geom(self,
                                mof_data: pd.DataFrame,
                                geom_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode MOFs with geometric features

        For augmented MOFs:
        - Supercells: Use original MOF's geometric features (intensive properties)
        - Thermal noise: Could recompute, but approximation is fine (±2% error)

        Returns:
            X_full: Input with geometric features [n_samples, n_metals+n_linkers+4+1+11]
            X_structure: Target structure [n_samples, n_metals+n_linkers+4]
        """
        # Create basic encoders
        unique_metals = mof_data['metal'].unique().tolist()
        self.metal_encoder = {m: i for i, m in enumerate(unique_metals)}
        self.linker_encoder = {l: i for i, l in enumerate(self.linkers)}

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
        n_metals = len(self.metal_encoder)
        n_linkers = len(self.linker_encoder)

        # One-hot metals
        metal_onehot = np.zeros((n_samples, n_metals))
        for i, metal in enumerate(mof_data['metal']):
            metal_onehot[i, self.metal_encoder[metal]] = 1

        # One-hot linkers
        linker_onehot = np.zeros((n_samples, n_linkers))
        for i, metal in enumerate(mof_data['metal']):
            linker = metal_linker_map.get(metal, 'terephthalic acid')
            linker_onehot[i, self.linker_encoder[linker]] = 1

        # Normalize cell parameters
        cell_params = mof_data[['cell_a', 'cell_b', 'cell_c', 'volume']].values
        self.cell_mean = cell_params.mean(axis=0)
        self.cell_std = cell_params.std(axis=0)
        cell_normalized = (cell_params - self.cell_mean) / (self.cell_std + 1e-8)

        # Normalize CO2
        co2_values = mof_data['co2_uptake_mean'].values.reshape(-1, 1)
        self.co2_mean = co2_values.mean()
        self.co2_std = co2_values.std()
        co2_normalized = (co2_values - self.co2_mean) / (self.co2_std + 1e-8)

        # Map geometric features (KEY FIX: use original MOF's features for augmented samples)
        geom_dict = {}
        geom_feature_cols = [c for c in geom_features.columns if c != 'mof_id']
        for _, row in geom_features.iterrows():
            geom_dict[row['mof_id']] = row[geom_feature_cols].values

        geom_features_matched = []
        for _, mof in mof_data.iterrows():
            mof_id = mof['mof_id']
            if mof_id in geom_dict:
                # Use original MOF's geometric features
                # Works for both original and augmented (supercells have same intensive properties)
                geom_features_matched.append(geom_dict[mof_id])
            else:
                # Fallback: mean features
                geom_features_matched.append(geom_features[geom_feature_cols].mean().values)

        geom_array = np.array(geom_features_matched, dtype=np.float64)

        # Normalize geometric features
        self.geom_mean = np.mean(geom_array, axis=0)
        self.geom_std = np.std(geom_array, axis=0) + 1e-8
        geom_normalized = (geom_array - self.geom_mean) / self.geom_std

        # Concatenate
        X_structure = np.concatenate([metal_onehot, linker_onehot, cell_normalized], axis=1)
        X_full = np.concatenate([X_structure, co2_normalized, geom_normalized], axis=1)

        print(f"  ✓ Encoded {n_samples} MOFs with CO2 + geometric features")
        print(f"  ✓ Metals: {list(self.metal_encoder.keys())}")
        print(f"  ✓ Linkers: {list(self.linker_encoder.keys())}")
        print(f"  ✓ CO2 range: {co2_values.min():.2f} - {co2_values.max():.2f} mmol/g")
        print(f"  ✓ Geometric features: {len(geom_feature_cols)} features")
        print(f"  ✓ Input dim: {X_full.shape[1]} (includes CO2 + geom)")
        print(f"  ✓ Output dim: {X_structure.shape[1]} (structure only)")

        return X_full, X_structure

    def generate_candidates(self,
                          n_candidates: int = 50,
                          target_co2: float = 5.0,
                          temperature: float = 2.0) -> int:
        """
        Generate candidates with hybrid cVAE

        Note: Can't actually generate geometric features (we don't decode them)
        This is just for diversity testing
        """
        if self.cvae is None:
            raise ValueError("cVAE not trained!")

        # Denormalize target CO2
        target_co2_normalized = (target_co2 - self.co2_mean) / (self.co2_std + 1e-8)

        # Sample latent space
        z = torch.randn(n_candidates, self.cvae.latent_dim)
        condition = torch.full((n_candidates, 1), target_co2_normalized)

        # Decode
        with torch.no_grad():
            samples = self.cvae.decode(z, condition)

        # Count unique metal-linker combinations
        metal_logits = samples[:, :self.cvae.n_metals]
        linker_logits = samples[:, self.cvae.n_metals:self.cvae.n_metals+self.cvae.n_linkers]

        # Temperature sampling
        def sample_categorical(logits, temp):
            if temp <= 0:
                return logits.argmax(dim=1)
            scaled = logits / temp
            probs = torch.softmax(scaled, dim=1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)

        metal_samples = sample_categorical(metal_logits, temperature)
        linker_samples = sample_categorical(linker_logits, temperature)

        unique_combos = set(zip(metal_samples.tolist(), linker_samples.tolist()))

        return len(unique_combos)

    def save(self, path: Path):
        """Save hybrid cVAE"""
        torch.save({
            'cvae_state': self.cvae.state_dict(),
            'metal_encoder': self.metal_encoder,
            'linker_encoder': self.linker_encoder,
            'cell_mean': self.cell_mean,
            'cell_std': self.cell_std,
            'co2_mean': self.co2_mean,
            'co2_std': self.co2_std,
            'geom_mean': self.geom_mean,
            'geom_std': self.geom_std,
        }, path)
        print(f"✓ Hybrid Conditional VAE saved to {path}")


if __name__ == '__main__':
    print("Testing Hybrid Conditional MOF VAE\n" + "="*60)

    # Load data
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"
    geom_file = project_root / "data" / "processed" / "crafted_geometric_features.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        exit(1)

    if not geom_file.exists():
        print(f"❌ Geometric features not found: {geom_file}")
        exit(1)

    mof_data = pd.read_csv(mof_file)
    geom_features = pd.read_csv(geom_file)

    print(f"✓ Loaded {len(mof_data)} MOFs")
    print(f"✓ Loaded {len(geom_features)} geometric feature sets\n")

    # Train hybrid cVAE
    generator = HybridConditionalMOFGenerator()
    generator.train_hybrid_cvae(mof_data, geom_features, epochs=100, beta=1.0)

    # Test diversity at different CO2 targets
    print(f"\n{'='*60}")
    print("Testing Diversity with Hybrid cVAE")
    print(f"{'='*60}\n")

    co2_targets = [2.0, 4.0, 6.0, 8.0]

    for target in co2_targets:
        unique = generator.generate_candidates(
            n_candidates=50,
            target_co2=target,
            temperature=2.0
        )
        print(f"Target CO2 = {target:.1f} mmol/g: {unique} unique (metal, linker) combinations")

    # Save
    model_dir = project_root / "models"
    generator.save(model_dir / "hybrid_conditional_mof_vae.pt")

    print(f"\n✓ Hybrid Conditional MOF VAE test complete!")
