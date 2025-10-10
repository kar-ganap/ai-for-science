"""
Geometric Feature Extraction for MOFs

Extract structural and geometric features from CIF files
Uses pymatgen only (no external tools like Zeo++)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from typing import Dict, List
from tqdm import tqdm


class GeometricFeatureExtractor:
    """
    Extract geometric features from MOF CIF files

    Features (without external tools):
    - Density
    - Volume per atom
    - Number of sites
    - Species diversity
    - Metal fraction
    - Average coordination
    - Packing efficiency proxy
    """

    def __init__(self):
        self.cnn = CrystalNN()

    def extract_features(self, cif_file: Path) -> Dict:
        """
        Extract geometric features from CIF file

        Args:
            cif_file: Path to CIF file

        Returns:
            Dictionary of geometric features
        """
        try:
            structure = Structure.from_file(str(cif_file))

            features = {
                # Basic structural properties
                'density': structure.density,
                'n_sites': len(structure),
                'n_species': len(structure.composition.elements),
                'volume_per_atom': structure.volume / len(structure),

                # Composition-based
                'metal_fraction': self._get_metal_fraction(structure),
                'organic_fraction': self._get_organic_fraction(structure),

                # Coordination (sample-based for speed)
                'avg_coordination': self._get_avg_coordination(structure, sample_size=20),

                # Packing estimates
                'packing_fraction': self._estimate_packing_fraction(structure),
                'void_fraction_proxy': self._estimate_void_fraction(structure),

                # Diversity measures
                'composition_complexity': len(structure.composition.elements),
                'site_diversity': len(set([s.species_string for s in structure])),
            }

            return features

        except Exception as e:
            print(f"  ⚠️  Failed to extract features from {cif_file.name}: {e}")
            return None

    def _get_metal_fraction(self, structure: Structure) -> float:
        """Fraction of sites that are metals"""
        metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn']

        metal_sites = sum(1 for site in structure if any(
            el.symbol in metals for el in site.species
        ))

        return metal_sites / len(structure) if len(structure) > 0 else 0

    def _get_organic_fraction(self, structure: Structure) -> float:
        """Fraction of sites that are organic (C, H, N, O)"""
        organic_elements = ['C', 'H', 'N', 'O']

        organic_sites = sum(1 for site in structure if any(
            el.symbol in organic_elements for el in site.species
        ))

        return organic_sites / len(structure) if len(structure) > 0 else 0

    def _get_avg_coordination(self, structure: Structure, sample_size: int = 20) -> float:
        """
        Average coordination number (sample-based for speed)

        Full calculation is slow, so sample sites
        """
        if len(structure) == 0:
            return 0

        # Sample sites (or use all if structure is small)
        sample_indices = np.random.choice(
            len(structure),
            min(sample_size, len(structure)),
            replace=False
        )

        coords = []
        for idx in sample_indices:
            try:
                nn_info = self.cnn.get_nn_info(structure, int(idx))
                coords.append(len(nn_info))
            except:
                coords.append(0)  # Failed to get neighbors

        return np.mean(coords) if coords else 0

    def _estimate_packing_fraction(self, structure: Structure) -> float:
        """
        Estimate packing fraction using atomic radii

        Packing fraction ≈ (sum of atomic volumes) / unit cell volume
        """
        from pymatgen.core.periodic_table import Element

        atomic_volumes = []
        for site in structure:
            for el, occ in site.species.items():
                # Use ionic or atomic radius
                radius = el.atomic_radius if el.atomic_radius else 1.5  # Fallback
                volume = (4/3) * np.pi * (radius ** 3)
                atomic_volumes.append(volume * occ)

        total_atomic_volume = sum(atomic_volumes)
        packing_fraction = total_atomic_volume / structure.volume

        # Clamp to [0, 1] (shouldn't exceed 1, but sometimes estimates are off)
        return min(packing_fraction, 1.0)

    def _estimate_void_fraction(self, structure: Structure) -> float:
        """
        Estimate void fraction (complement of packing fraction)

        For MOFs, this correlates with pore volume
        """
        packing = self._estimate_packing_fraction(structure)
        return 1.0 - packing


def extract_features_for_dataset(
    cif_directory: Path,
    mof_ids: List[str],
    output_file: Path = None
) -> pd.DataFrame:
    """
    Extract geometric features for a dataset of MOFs

    Args:
        cif_directory: Directory containing CIF files
        mof_ids: List of MOF IDs to process
        output_file: Where to save features (optional)

    Returns:
        DataFrame with mof_id + geometric features
    """
    print(f"\n{'='*60}")
    print("Extracting Geometric Features from CIF Files")
    print(f"{'='*60}\n")
    print(f"Processing {len(mof_ids)} MOFs...")

    extractor = GeometricFeatureExtractor()

    results = []
    failed = []

    for mof_id in tqdm(mof_ids, desc="Extracting features"):
        # Try to find CIF file (could be in different subdirectories)
        cif_file = None
        for subdir in cif_directory.iterdir():
            if subdir.is_dir():
                potential_file = subdir / f"{mof_id}.cif"
                if potential_file.exists():
                    cif_file = potential_file
                    break

        if cif_file is None:
            failed.append(mof_id)
            continue

        features = extractor.extract_features(cif_file)

        if features:
            features['mof_id'] = mof_id
            results.append(features)
        else:
            failed.append(mof_id)

    features_df = pd.DataFrame(results)

    print(f"\n✓ Successfully extracted features for {len(results)} MOFs")
    if failed:
        print(f"⚠️  Failed for {len(failed)} MOFs")

    # Save if requested
    if output_file:
        features_df.to_csv(output_file, index=False)
        print(f"\n📊 Saved features to: {output_file}")

    return features_df


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Geometric Feature Extraction\n" + "="*60)

    # Load CRAFTED MOF IDs
    project_root = Path(__file__).parents[2]
    mof_data_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"
    cif_directory = project_root / "data" / "CRAFTED-2.0.0" / "CIF_FILES"

    if not mof_data_file.exists():
        print(f"❌ MOF data not found: {mof_data_file}")
        exit(1)

    if not cif_directory.exists():
        print(f"❌ CIF directory not found: {cif_directory}")
        exit(1)

    # Get MOF IDs
    mof_data = pd.read_csv(mof_data_file)
    mof_ids = mof_data['mof_id'].unique().tolist()

    print(f"Found {len(mof_ids)} unique MOF IDs\n")

    # Extract features
    features_df = extract_features_for_dataset(
        cif_directory,
        mof_ids,
        output_file=project_root / "data" / "processed" / "crafted_geometric_features.csv"
    )

    print(f"\n{'='*60}")
    print("Sample extracted features:")
    print(f"{'='*60}\n")

    print(features_df.head())

    print(f"\nFeature statistics:")
    print(features_df.describe())

    print(f"\n✓ Geometric feature extraction test complete!")
