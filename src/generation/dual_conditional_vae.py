"""
Dual-Conditional VAE for Economic Generative Discovery

Key Innovation: Conditions on BOTH CO2 uptake AND synthesis cost
- Encoder: Takes (metal, linker, cell, CO2, COST, geom_features) → latent
- Decoder: Takes (latent, CO2, COST) → (metal, linker, cell)
- Generation: Sample with (target_co2=7.0, target_cost=0.8) → economically optimized MOFs

This enables true Economic Generative Discovery:
- Generate high-performance MOFs (high CO2)
- That are economically viable (low cost)
- Perfect alignment with Economic Active Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DualConditionalMOF_VAE(nn.Module):
    """
    Dual-Conditional VAE for targeted MOF generation

    Conditions on BOTH CO2 uptake and synthesis cost to generate
    economically viable, high-performance MOFs
    """

    def __init__(self,
                 n_metals: int,
                 n_linkers: int,
                 latent_dim: int = 16,
                 hidden_dim: int = 64,
                 use_geom_features: bool = True):
        super(DualConditionalMOF_VAE, self).__init__()

        self.n_metals = n_metals
        self.n_linkers = n_linkers
        self.latent_dim = latent_dim
        self.use_geom_features = use_geom_features

        # Input: metal + linker + cell(4) + CO2(1) + Cost(1) + geom(11 if used)
        geom_dim = 11 if use_geom_features else 0
        input_dim = n_metals + n_linkers + 4 + 2 + geom_dim  # +2 for CO2 and Cost

        # Output: metal + linker + cell(4) (NO CO2, NO Cost, NO geom)
        output_dim = n_metals + n_linkers + 4

        # Encoder: (structure + CO2 + Cost + geom) → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (latent + CO2 + Cost) → structure
        decoder_input_dim = latent_dim + 2  # latent + CO2 + Cost

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, x):
        """Encode (structure + CO2 + Cost + geom) to latent distribution"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, conditions):
        """
        Decode latent + conditions to structure

        Args:
            z: Latent code [batch_size, latent_dim]
            conditions: [CO2, Cost] [batch_size, 2]
        """
        decoder_input = torch.cat([z, conditions], dim=1)
        return self.decoder(decoder_input)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [metal, linker, cell, CO2, Cost, geom_features]

        Returns:
            recon: Reconstructed structure (no CO2/Cost/geom)
            mu, logvar: Latent distribution parameters
        """
        # Encode full input (including CO2, Cost, geom)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Extract dual conditions from input
        # Structure: [metal, linker, cell(4), CO2, Cost, geom(11 if used)]
        # CO2 is at position: n_metals + n_linkers + 4
        # Cost is at position: n_metals + n_linkers + 4 + 1
        structure_dim = self.n_metals + self.n_linkers + 4
        co2_idx = structure_dim
        cost_idx = structure_dim + 1

        co2_condition = x[:, co2_idx:co2_idx+1]
        cost_condition = x[:, cost_idx:cost_idx+1]
        dual_condition = torch.cat([co2_condition, cost_condition], dim=1)

        # Decode with dual conditions
        recon = self.decode(z, dual_condition)

        return recon, mu, logvar

    def conditional_sample(self,
                          n_samples: int = 1,
                          target_co2: float = 7.0,
                          target_cost: float = 0.8):
        """
        Sample MOFs conditioned on BOTH target CO2 and target cost

        Args:
            n_samples: Number of samples to generate
            target_co2: Target CO2 uptake (mol/kg)
            target_cost: Target synthesis cost ($/g)

        Returns:
            Generated samples (structure only, no CO2/cost)
        """
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim)

            # Create dual condition tensor
            co2_cond = torch.full((n_samples, 1), target_co2)
            cost_cond = torch.full((n_samples, 1), target_cost)
            dual_condition = torch.cat([co2_cond, cost_cond], dim=1)

            # Decode
            samples = self.decode(z, dual_condition)

        return samples


def dual_conditional_vae_loss(recon_x, x_structure_only, mu, logvar,
                               n_metals, n_linkers, beta=1.0):
    """
    Dual-Conditional VAE loss (same as conditional, but clearer naming)
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


class DualConditionalMOFGenerator:
    """
    MOF Generator using Dual-Conditional VAE

    Generates MOFs optimized for BOTH performance and cost
    """

    def __init__(self, use_geom_features: bool = True):
        self.cvae = None
        self.metal_encoder = None
        self.linker_encoder = None
        self.cell_mean = None
        self.cell_std = None
        self.co2_mean = None
        self.co2_std = None
        self.cost_mean = None
        self.cost_std = None
        self.geom_mean = None
        self.geom_std = None
        self.use_geom_features = use_geom_features

        # Standard metals and linkers (from CRAFTED)
        self.metals = ['Zn', 'Cu', 'Fe', 'Al', 'Ca', 'Zr', 'Cr', 'Unknown']
        self.linkers = [
            'terephthalic acid',
            'trimesic acid',
            '2,6-naphthalenedicarboxylic acid',
            'biphenyl-4,4-dicarboxylic acid',
        ]

    def train_dual_cvae(self,
                       mof_data: pd.DataFrame,
                       geom_features: Optional[pd.DataFrame] = None,
                       epochs: int = 100,
                       batch_size: int = 32,
                       lr: float = 1e-3,
                       beta: float = 1.0,
                       augment: bool = True,
                       use_latent_perturbation: bool = True):
        """
        Train Dual-Conditional VAE on MOF dataset

        Args:
            mof_data: DataFrame with metal, cell, co2_uptake_mean, synthesis_cost
            geom_features: Optional geometric features
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            beta: KL weight
            augment: Apply data augmentation
            use_latent_perturbation: Add latent space perturbation augmentation
        """
        print(f"\n{'='*60}")
        print("Training Dual-Conditional MOF VAE (CO2 + Cost)")
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

        # Encode data (now includes CO2 AND Cost)
        if self.use_geom_features and geom_features is not None:
            X_full, X_structure = self._encode_mofs_dual_with_geom(mof_data, geom_features)
        else:
            X_full, X_structure = self._encode_mofs_dual(mof_data)

        # Create datasets
        dataset_full = torch.FloatTensor(X_full)  # Input (has CO2 + Cost + geom)
        dataset_structure = torch.FloatTensor(X_structure)  # Target (no CO2/cost/geom)

        combined_dataset = torch.utils.data.TensorDataset(dataset_full, dataset_structure)
        dataloader = torch.utils.data.DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize Dual cVAE
        self.cvae = DualConditionalMOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=64,
            use_geom_features=self.use_geom_features
        )

        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr)

        # Training loop
        print(f"Training for {epochs} epochs...")
        print(f"Latent perturbation: {'Enabled' if use_latent_perturbation else 'Disabled'}")

        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for batch_full, batch_structure in dataloader:
                optimizer.zero_grad()

                # Regular forward pass
                recon, mu, logvar = self.cvae(batch_full)
                loss, recon_loss, kl_loss = dual_conditional_vae_loss(
                    recon, batch_structure, mu, logvar,
                    self.cvae.n_metals, self.cvae.n_linkers,
                    beta=beta
                )

                # Optional: Latent space perturbation augmentation
                if use_latent_perturbation and np.random.rand() < 0.3:  # 30% of batches
                    # Perturb latent codes
                    z_perturbed = mu + torch.randn_like(mu) * 0.1

                    # Extract conditions (same indices as forward)
                    structure_dim = self.cvae.n_metals + self.cvae.n_linkers + 4
                    co2_idx = structure_dim
                    cost_idx = structure_dim + 1
                    co2_cond = batch_full[:, co2_idx:co2_idx+1]
                    cost_cond = batch_full[:, cost_idx:cost_idx+1]
                    dual_cond = torch.cat([co2_cond, cost_cond], dim=1)

                    # Decode perturbed latents
                    recon_perturbed = self.cvae.decode(z_perturbed, dual_cond)
                    loss_perturbed, _, _ = dual_conditional_vae_loss(
                        recon_perturbed, batch_structure, mu, logvar,
                        self.cvae.n_metals, self.cvae.n_linkers,
                        beta=beta
                    )
                    loss = loss + 0.2 * loss_perturbed  # Weighted augmentation loss

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

        print(f"\n✓ Dual-Conditional VAE training complete!")
        self.cvae.eval()

    def _encode_mofs_dual(self, mof_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode MOF data INCLUDING CO2 uptake AND synthesis cost (no geom features)

        Returns:
            X_full: [metal, linker, cell, CO2, Cost] - for input
            X_structure: [metal, linker, cell] - for target
        """
        # Load linker assignments
        project_root = Path(__file__).parents[2]
        linker_file = project_root / "data/processed/crafted_mofs_linkers.csv"

        if not linker_file.exists():
            raise FileNotFoundError(
                f"Linker assignments not found: {linker_file}\n"
                f"Run create_linker_assignments.py first!"
            )

        linker_data = pd.read_csv(linker_file)

        # Merge linker information with MOF data
        mof_data_with_linkers = mof_data.merge(
            linker_data[['mof_id', 'linker']],
            on='mof_id',
            how='left'
        )

        # Check for missing linkers
        missing_linkers = mof_data_with_linkers['linker'].isna().sum()
        if missing_linkers > 0:
            print(f"  ⚠️  {missing_linkers} MOFs missing linker assignments, using default")
            mof_data_with_linkers['linker'].fillna('terephthalic acid', inplace=True)

        # Create encoders
        unique_metals = mof_data_with_linkers['metal'].unique().tolist()
        self.metal_encoder = {m: i for i, m in enumerate(unique_metals)}
        self.linker_encoder = {l: i for i, l in enumerate(self.linkers)}

        n_samples = len(mof_data_with_linkers)
        n_metals = len(self.metal_encoder)
        n_linkers = len(self.linker_encoder)

        # One-hot encode metals
        metal_onehot = np.zeros((n_samples, n_metals))
        for i, metal in enumerate(mof_data_with_linkers['metal']):
            metal_onehot[i, self.metal_encoder[metal]] = 1

        # One-hot encode linkers (NOW USING REAL DATA!)
        linker_onehot = np.zeros((n_samples, n_linkers))
        for i, linker in enumerate(mof_data_with_linkers['linker']):
            if linker in self.linker_encoder:
                linker_onehot[i, self.linker_encoder[linker]] = 1
            else:
                # Fallback to terephthalic acid if unknown linker
                linker_onehot[i, self.linker_encoder['terephthalic acid']] = 1

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

        # Normalize synthesis cost (NEW!)
        cost_values = mof_data['synthesis_cost'].values.reshape(-1, 1)
        self.cost_mean = cost_values.mean()
        self.cost_std = cost_values.std()
        cost_normalized = (cost_values - self.cost_mean) / (self.cost_std + 1e-8)

        # Concatenate
        X_structure = np.concatenate([metal_onehot, linker_onehot, cell_normalized], axis=1)
        X_full = np.concatenate([X_structure, co2_normalized, cost_normalized], axis=1)

        # Count unique combinations
        unique_combos = mof_data_with_linkers.groupby(['metal', 'linker']).size()
        n_combos = len(unique_combos)

        print(f"  ✓ Encoded {n_samples} MOFs with CO2 + Cost")
        print(f"  ✓ Metals: {list(self.metal_encoder.keys())}")
        print(f"  ✓ Linkers: {list(self.linker_encoder.keys())}")
        print(f"  ✓ Unique (metal, linker) combinations: {n_combos}")
        print(f"  ✓ CO2 range: {co2_values.min():.2f} - {co2_values.max():.2f} mol/kg")
        print(f"  ✓ Cost range: ${cost_values.min():.2f} - ${cost_values.max():.2f}/g")
        print(f"  ✓ Input dim: {X_full.shape[1]} (includes CO2 + Cost)")
        print(f"  ✓ Output dim: {X_structure.shape[1]} (structure only)")

        return X_full, X_structure

    def _encode_mofs_dual_with_geom(self,
                                     mof_data: pd.DataFrame,
                                     geom_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode MOFs with CO2, Cost, AND geometric features

        Returns:
            X_full: [metal, linker, cell, CO2, Cost, geom] - for input
            X_structure: [metal, linker, cell] - for target
        """
        # First get basic encoding (metal, linker, cell, CO2, Cost)
        X_basic, X_structure = self._encode_mofs_dual(mof_data)

        # Map geometric features
        geom_dict = {}
        geom_feature_cols = [c for c in geom_features.columns if c != 'mof_id']
        for _, row in geom_features.iterrows():
            geom_dict[row['mof_id']] = row[geom_feature_cols].values

        geom_features_matched = []
        for _, mof in mof_data.iterrows():
            mof_id = mof['mof_id']
            if mof_id in geom_dict:
                geom_features_matched.append(geom_dict[mof_id])
            else:
                geom_features_matched.append(geom_features[geom_feature_cols].mean().values)

        geom_array = np.array(geom_features_matched, dtype=np.float64)

        # Normalize geometric features
        self.geom_mean = np.mean(geom_array, axis=0)
        self.geom_std = np.std(geom_array, axis=0) + 1e-8
        geom_normalized = (geom_array - self.geom_mean) / self.geom_std

        # Concatenate: [metal, linker, cell, CO2, Cost, geom]
        X_full = np.concatenate([X_basic, geom_normalized], axis=1)

        print(f"  ✓ Added {len(geom_feature_cols)} geometric features")
        print(f"  ✓ Total input dim: {X_full.shape[1]} (structure + CO2 + Cost + geom)")

        return X_full, X_structure

    def generate_candidates(self,
                          n_candidates: int = 100,
                          target_co2: float = 7.0,
                          target_cost: float = 0.8,
                          temperature: float = 2.0,
                          min_compositional_diversity: float = 0.3) -> List[Dict]:
        """
        Generate MOF candidates conditioned on BOTH CO2 and cost
        WITH COMPOSITIONAL DIVERSITY ENFORCEMENT

        Args:
            n_candidates: Number to generate
            target_co2: Target CO2 uptake (mol/kg)
            target_cost: Target synthesis cost ($/g)
            temperature: Sampling temperature for diversity
            min_compositional_diversity: Minimum fraction of unique (metal, linker) pairs (default: 0.3 = 30%)

        Returns:
            List of candidate MOF compositions with predicted properties
        """
        if self.cvae is None:
            raise ValueError("Dual cVAE not trained! Call train_dual_cvae() first.")

        # Normalize targets
        target_co2_normalized = (target_co2 - self.co2_mean) / (self.co2_std + 1e-8)
        target_cost_normalized = (target_cost - self.cost_mean) / (self.cost_std + 1e-8)

        # Calculate minimum unique compositions needed
        min_unique_compositions = max(int(n_candidates * min_compositional_diversity), 1)

        candidates = []
        composition_counts = {}  # Track (metal, linker) -> count

        print(f"\nGenerating {n_candidates} MOF candidates with diversity enforcement:")
        print(f"  Target CO2: {target_co2:.2f} mol/kg")
        print(f"  Target Cost: ${target_cost:.2f}/g")
        print(f"  Minimum unique compositions: {min_unique_compositions} ({min_compositional_diversity*100:.0f}%)")

        # Generate in batches with diversity enforcement
        max_attempts = n_candidates * 5  # Safety limit
        attempts = 0
        batch_size = 50

        with torch.no_grad():
            while len(candidates) < n_candidates and attempts < max_attempts:
                # Generate batch
                samples = self.cvae.conditional_sample(
                    n_samples=batch_size,
                    target_co2=float(target_co2_normalized),
                    target_cost=float(target_cost_normalized)
                )

                for i in range(batch_size):
                    if len(candidates) >= n_candidates:
                        break

                    candidate = self._decode_mof(samples[i].numpy(), temperature)
                    if not candidate:
                        continue

                    candidate['target_co2'] = target_co2
                    candidate['target_cost'] = target_cost

                    composition = (candidate['metal'], candidate['linker'])
                    current_count = composition_counts.get(composition, 0)
                    n_unique_so_far = len(composition_counts)

                    # Diversity enforcement logic:
                    # Accept if: (1) New composition, OR
                    #           (2) We have enough unique compositions AND this one isn't overrepresented
                    if composition not in composition_counts:
                        # New composition - always accept
                        candidates.append(candidate)
                        composition_counts[composition] = 1
                    elif n_unique_so_far >= min_unique_compositions:
                        # We have enough diversity, allow duplicates but limit per composition
                        max_per_composition = n_candidates // n_unique_so_far + 2
                        if current_count < max_per_composition:
                            candidates.append(candidate)
                            composition_counts[composition] = current_count + 1
                    # else: reject (need more unique compositions first)

                attempts += batch_size

        print(f"  ✓ Generated {len(candidates)} valid candidates (attempts: {attempts})")

        # Count diversity
        unique_combos = set((c['metal'], c['linker']) for c in candidates)
        diversity_pct = 100 * len(unique_combos) / len(candidates) if candidates else 0
        print(f"  ✓ Unique (metal, linker): {len(unique_combos)} ({diversity_pct:.1f}% compositional diversity)")

        # Show composition breakdown
        if len(unique_combos) <= 15:  # Only if reasonable number
            print(f"  ✓ Composition distribution:")
            for comp, count in sorted(composition_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"      {comp[0]:4s} + {comp[1][:25]:25s}: {count:2d} candidates")

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
        """Save trained dual cVAE"""
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
            'cost_mean': self.cost_mean,
            'cost_std': self.cost_std,
            'geom_mean': self.geom_mean,
            'geom_std': self.geom_std,
            'use_geom_features': self.use_geom_features,
        }, path)
        print(f"✓ Dual-Conditional VAE saved to {path}")

    def load(self, path: Path):
        """Load trained dual cVAE"""
        checkpoint = torch.load(path)

        self.metal_encoder = checkpoint['metal_encoder']
        self.linker_encoder = checkpoint['linker_encoder']
        self.cell_mean = checkpoint['cell_mean']
        self.cell_std = checkpoint['cell_std']
        self.co2_mean = checkpoint['co2_mean']
        self.co2_std = checkpoint['co2_std']
        self.cost_mean = checkpoint['cost_mean']
        self.cost_std = checkpoint['cost_std']
        self.geom_mean = checkpoint.get('geom_mean', None)
        self.geom_std = checkpoint.get('geom_std', None)
        self.use_geom_features = checkpoint.get('use_geom_features', True)

        self.cvae = DualConditionalMOF_VAE(
            n_metals=len(self.metal_encoder),
            n_linkers=len(self.linker_encoder),
            latent_dim=16,
            hidden_dim=64,
            use_geom_features=self.use_geom_features
        )
        self.cvae.load_state_dict(checkpoint['cvae_state'])
        self.cvae.eval()

        print(f"✓ Dual-Conditional VAE loaded from {path}")


if __name__ == '__main__':
    print("Training Dual-Conditional MOF VAE\n" + "="*60)

    # Load CRAFTED MOF data with costs
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"
    geom_file = project_root / "data" / "processed" / "crafted_geometric_features.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        exit(1)

    mof_data = pd.read_csv(mof_file)
    print(f"✓ Loaded {len(mof_data)} MOFs with CO2 and cost data\n")

    # Check data
    print(f"CO2 uptake range: {mof_data['co2_uptake_mean'].min():.2f} - {mof_data['co2_uptake_mean'].max():.2f} mol/kg")
    print(f"Synthesis cost range: ${mof_data['synthesis_cost'].min():.2f} - ${mof_data['synthesis_cost'].max():.2f}/g\n")

    # Load geometric features if available
    geom_features = None
    use_geom = False
    if geom_file.exists():
        geom_features = pd.read_csv(geom_file)
        use_geom = True
        print(f"✓ Loaded {len(geom_features)} geometric feature sets\n")

    # Initialize generator
    generator = DualConditionalMOFGenerator(use_geom_features=use_geom)

    # Train dual cVAE
    generator.train_dual_cvae(
        mof_data,
        geom_features=geom_features,
        epochs=100,
        batch_size=32,
        beta=1.0,
        augment=True,
        use_latent_perturbation=True  # NEW augmentation!
    )

    # Test dual-conditional generation
    print(f"\n{'='*60}")
    print("Testing Dual-Conditional Generation")
    print(f"{'='*60}\n")

    # Test different (CO2, Cost) targets
    test_targets = [
        (6.0, 0.8),  # High performance, low cost
        (8.0, 1.0),  # Very high performance, medium cost
        (4.0, 0.7),  # Medium performance, very low cost
    ]

    for target_co2, target_cost in test_targets:
        candidates = generator.generate_candidates(
            n_candidates=50,
            target_co2=target_co2,
            target_cost=target_cost,
            temperature=2.0
        )
        print()

    # Save model
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    generator.save(model_dir / "dual_conditional_mof_vae.pt")

    print(f"\n✓ Dual-Conditional MOF VAE training complete!")
    print(f"✓ Model ready for Economic Generative Discovery integration!")
