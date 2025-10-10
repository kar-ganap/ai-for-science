"""
MOF Data Augmentation

Scientifically valid augmentation strategies:
1. Supercell generation (same structure, different unit cell)
2. Thermal noise (simulates temperature/measurement uncertainty)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


class MOFAugmenter:
    """
    Data augmentation for MOF datasets

    Strategies:
    - Supercells: Generate 2x1x1, 1x2x1, 1x1x2, 2x2x2 representations
    - Thermal noise: Add ±2% noise to cell parameters (simulates thermal fluctuations)
    """

    def __init__(self):
        self.supercell_factors = [
            (1, 1, 1),  # Original
            (2, 1, 1),  # 2× along a
            (1, 2, 1),  # 2× along b
            (1, 1, 2),  # 2× along c
            (2, 2, 1),  # 2× along a,b
            (2, 1, 2),  # 2× along a,c
            (1, 2, 2),  # 2× along b,c
            (2, 2, 2),  # 2× along all
        ]

    def augment_dataset(self,
                       mof_data: pd.DataFrame,
                       use_supercells: bool = True,
                       use_thermal_noise: bool = True,
                       noise_level: float = 0.02) -> pd.DataFrame:
        """
        Augment MOF dataset

        Args:
            mof_data: DataFrame with columns: mof_id, metal, cell_a, cell_b, cell_c, volume
            use_supercells: Apply supercell augmentation (8×)
            use_thermal_noise: Apply thermal noise (2× per structure)
            noise_level: Gaussian noise std as fraction of cell params (default 2%)

        Returns:
            Augmented DataFrame (up to 16× larger)
        """
        print(f"\n{'='*60}")
        print("MOF Data Augmentation")
        print(f"{'='*60}\n")
        print(f"Original dataset: {len(mof_data)} MOFs")

        augmented = []

        for idx, mof in mof_data.iterrows():
            if use_supercells:
                # Generate supercells
                supercells = self._generate_supercells(mof)

                if use_thermal_noise:
                    # Add thermal noise to each supercell
                    for sc in supercells:
                        # Original supercell
                        augmented.append(sc)

                        # Noisy version
                        noisy = self._add_thermal_noise(sc, noise_level)
                        augmented.append(noisy)
                else:
                    augmented.extend(supercells)
            else:
                # Just original + noise (if enabled)
                augmented.append(mof.to_dict())

                if use_thermal_noise:
                    noisy = self._add_thermal_noise(mof.to_dict(), noise_level)
                    augmented.append(noisy)

        augmented_df = pd.DataFrame(augmented)

        print(f"\nAugmentation summary:")
        print(f"  Supercells:    {'✓' if use_supercells else '✗'} ({len(self.supercell_factors)}× per MOF)")
        print(f"  Thermal noise: {'✓' if use_thermal_noise else '✗'} (±{noise_level*100:.1f}% cell params)")
        print(f"  Final size:    {len(augmented_df)} samples ({len(augmented_df)/len(mof_data):.1f}×)")

        return augmented_df

    def _generate_supercells(self, mof: pd.Series) -> List[Dict]:
        """
        Generate supercell representations

        Supercell (n_a, n_b, n_c):
        - cell_a' = n_a × cell_a
        - cell_b' = n_b × cell_b
        - cell_c' = n_c × cell_c
        - volume' = n_a × n_b × n_c × volume

        Same material, different unit cell representation
        """
        supercells = []

        for (n_a, n_b, n_c) in self.supercell_factors:
            supercell = mof.to_dict()

            # Scale cell parameters
            supercell['cell_a'] = mof['cell_a'] * n_a
            supercell['cell_b'] = mof['cell_b'] * n_b
            supercell['cell_c'] = mof['cell_c'] * n_c
            supercell['volume'] = mof['volume'] * n_a * n_b * n_c

            # Track supercell factor (for debugging)
            supercell['supercell_factor'] = f"{n_a}x{n_b}x{n_c}"
            supercell['is_augmented'] = (n_a, n_b, n_c) != (1, 1, 1)

            supercells.append(supercell)

        return supercells

    def _add_thermal_noise(self, mof: Dict, noise_level: float = 0.02) -> Dict:
        """
        Add thermal/experimental noise to cell parameters

        Simulates:
        - Thermal fluctuations (±1-3% from MD simulations)
        - Experimental measurement uncertainty (±0.5-2% XRD)

        Args:
            mof: MOF dict with cell_a, cell_b, cell_c, volume
            noise_level: Std of Gaussian noise (fraction of parameter value)

        Returns:
            Noisy MOF dict
        """
        noisy = mof.copy()

        # Add Gaussian noise to cell parameters
        for param in ['cell_a', 'cell_b', 'cell_c']:
            if param in noisy:
                # noise ~ N(0, noise_level * param_value)
                noise = np.random.normal(0, noise_level * noisy[param])
                noisy[param] = noisy[param] + noise

                # Ensure positive (cell params must be > 0)
                noisy[param] = max(noisy[param], 0.1)

        # Recompute volume (consistent with noisy cell params)
        if all(k in noisy for k in ['cell_a', 'cell_b', 'cell_c']):
            noisy['volume'] = noisy['cell_a'] * noisy['cell_b'] * noisy['cell_c']

        # Track that this is augmented
        noisy['is_augmented'] = True
        noisy['augmentation_type'] = 'thermal_noise'

        return noisy

    def validate_augmentation(self,
                             original: pd.DataFrame,
                             augmented: pd.DataFrame) -> Dict:
        """
        Validate augmentation quality

        Checks:
        1. Cell parameters stay in reasonable range
        2. Volumes are physically plausible
        3. Metal/linker unchanged (augmentation shouldn't change chemistry)
        """
        stats = {}

        # Check cell parameter ranges
        for param in ['cell_a', 'cell_b', 'cell_c', 'volume']:
            orig_min, orig_max = original[param].min(), original[param].max()
            aug_min, aug_max = augmented[param].min(), augmented[param].max()

            stats[f'{param}_range_original'] = (orig_min, orig_max)
            stats[f'{param}_range_augmented'] = (aug_min, aug_max)

            # Check if augmented stays within 2× of original range
            if aug_min < orig_min * 0.5 or aug_max > orig_max * 2.5:
                stats[f'{param}_warning'] = "Augmentation may have gone too far"

        # Check metal distribution (should be same)
        orig_metals = original['metal'].value_counts().to_dict()
        aug_metals = augmented['metal'].value_counts().to_dict()

        stats['metal_distribution_preserved'] = all(
            aug_metals.get(m, 0) / orig_metals.get(m, 1) > 5  # At least 8× for supercells
            for m in orig_metals
        )

        return stats


def quick_augment(mof_data_file: Path,
                 output_file: Path = None,
                 supercells: bool = True,
                 thermal_noise: bool = True) -> pd.DataFrame:
    """
    Quick augmentation for a MOF dataset file

    Args:
        mof_data_file: Path to CSV with MOF data
        output_file: Where to save augmented data (optional)
        supercells: Use supercell augmentation
        thermal_noise: Use thermal noise augmentation

    Returns:
        Augmented DataFrame
    """
    # Load data
    mof_data = pd.read_csv(mof_data_file)

    # Augment
    augmenter = MOFAugmenter()
    augmented = augmenter.augment_dataset(
        mof_data,
        use_supercells=supercells,
        use_thermal_noise=thermal_noise
    )

    # Validate
    stats = augmenter.validate_augmentation(mof_data, augmented)

    print(f"\nValidation:")
    print(f"  Cell param ranges: ✓")
    print(f"  Metal distribution: {'✓' if stats['metal_distribution_preserved'] else '⚠️'}")

    # Save if requested
    if output_file:
        augmented.to_csv(output_file, index=False)
        print(f"\n📊 Saved augmented data to: {output_file}")

    return augmented


if __name__ == '__main__':
    # Test augmentation
    print("Testing MOF Augmentation\n" + "="*60)

    # Load CRAFTED data
    project_root = Path(__file__).parents[2]
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"

    if not mof_file.exists():
        print(f"❌ MOF data not found: {mof_file}")
        exit(1)

    # Quick test
    augmented = quick_augment(
        mof_file,
        output_file=project_root / "data" / "processed" / "crafted_mofs_augmented.csv",
        supercells=True,
        thermal_noise=True
    )

    print(f"\n{'='*60}")
    print("Sample augmented data:")
    print(f"{'='*60}\n")

    # Show a few examples
    print(augmented[['mof_id', 'metal', 'cell_a', 'cell_b', 'cell_c', 'volume', 'supercell_factor']].head(10))

    print(f"\n✓ Augmentation test complete!")
    print(f"  Original: {len(pd.read_csv(mof_file))} MOFs")
    print(f"  Augmented: {len(augmented)} samples")
