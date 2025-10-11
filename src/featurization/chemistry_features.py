"""
Chemistry-Informed Features for MOF Surrogate Model

Adds chemical properties beyond simple one-hot encoding to help
the surrogate model understand the chemistry of MOFs.

These features capture:
- Electronic properties (electronegativity, oxidation states)
- Geometric properties (ionic radii, coordination)
- Linker characteristics (length, rigidity, aromaticity)
- Metal-linker compatibility
"""

import numpy as np
from typing import Dict, List


class ChemistryFeaturizer:
    """
    Generate chemistry-informed features for MOFs
    """

    # Metal properties
    ELECTRONEGATIVITY = {
        'Zn': 1.65, 'Fe': 1.83, 'Ca': 1.00, 'Al': 1.61,
        'Ti': 1.54, 'Cu': 1.90, 'Zr': 1.33, 'Cr': 1.66,
        'Unknown': 1.50  # Average
    }

    IONIC_RADII = {  # 6-coordinate, Angstroms
        'Zn': 0.74, 'Fe': 0.78, 'Ca': 1.00, 'Al': 0.54,
        'Ti': 0.61, 'Cu': 0.73, 'Zr': 0.72, 'Cr': 0.62,
        'Unknown': 0.70
    }

    OXIDATION_STATES = {
        'Zn': 2, 'Fe': 2, 'Ca': 2, 'Al': 3,
        'Ti': 4, 'Cu': 2, 'Zr': 4, 'Cr': 3,
        'Unknown': 2
    }

    # Typical coordination numbers
    COORDINATION_PREFERENCE = {
        'Zn': 5.0,  # Average of 4 and 6
        'Fe': 6.0,
        'Ca': 7.0,  # Average of 6,7,8
        'Al': 5.0,  # Average of 4 and 6
        'Ti': 6.0,
        'Cu': 5.0,  # Average of 4,5,6
        'Zr': 7.0,  # Average of 6,7,8
        'Cr': 6.0,
        'Unknown': 6.0
    }

    # Lewis acidity (qualitative scale: 0=soft, 1=borderline, 2=hard)
    LEWIS_ACIDITY = {
        'Zn': 1.0, 'Fe': 1.0, 'Ca': 2.0, 'Al': 2.0,
        'Ti': 2.0, 'Cu': 1.0, 'Zr': 2.0, 'Cr': 2.0,
        'Unknown': 1.0
    }

    # Linker properties
    LINKER_LENGTH = {  # Carboxylate separation, Angstroms
        'terephthalic acid': 11.0,
        'trimesic acid': 9.5,
        '2,6-naphthalenedicarboxylic acid': 13.0,
        'biphenyl-4,4-dicarboxylic acid': 15.0,
        'Unknown': 11.5  # Average
    }

    LINKER_DENTICITY = {  # Number of coordination sites
        'terephthalic acid': 2,
        'trimesic acid': 3,
        '2,6-naphthalenedicarboxylic acid': 2,
        'biphenyl-4,4-dicarboxylic acid': 2,
        'Unknown': 2
    }

    LINKER_RIGIDITY = {  # 0=flexible, 1=rigid
        'terephthalic acid': 0.9,
        'trimesic acid': 1.0,
        '2,6-naphthalenedicarboxylic acid': 0.8,
        'biphenyl-4,4-dicarboxylic acid': 0.6,
        'Unknown': 0.8
    }

    LINKER_AROMATICITY = {  # Number of aromatic rings
        'terephthalic acid': 1,
        'trimesic acid': 1,
        '2,6-naphthalenedicarboxylic acid': 2,
        'biphenyl-4,4-dicarboxylic acid': 2,
        'Unknown': 1
    }

    LINKER_MOLECULAR_WEIGHT = {
        'terephthalic acid': 166,
        'trimesic acid': 210,
        '2,6-naphthalenedicarboxylic acid': 216,
        'biphenyl-4,4-dicarboxylic acid': 242,
        'Unknown': 208
    }

    def __init__(self, include_derived: bool = True):
        """
        Args:
            include_derived: Include derived chemistry features (compatibility, etc.)
        """
        self.include_derived = include_derived

    def featurize(self, mof: Dict) -> np.ndarray:
        """
        Generate chemistry-informed features for a MOF

        Args:
            mof: MOF dictionary with keys: metal, linker, cell_a, cell_b, cell_c, volume

        Returns:
            features: Chemistry feature vector
        """
        metal = mof.get('metal', 'Unknown')
        linker = mof.get('linker', 'Unknown')

        features = []

        # === METAL PROPERTIES ===
        features.append(self.ELECTRONEGATIVITY[metal])
        features.append(self.IONIC_RADII[metal])
        features.append(self.OXIDATION_STATES[metal])
        features.append(self.COORDINATION_PREFERENCE[metal])
        features.append(self.LEWIS_ACIDITY[metal])

        # === LINKER PROPERTIES ===
        features.append(self.LINKER_LENGTH[linker])
        features.append(self.LINKER_DENTICITY[linker])
        features.append(self.LINKER_RIGIDITY[linker])
        features.append(self.LINKER_AROMATICITY[linker])
        features.append(self.LINKER_MOLECULAR_WEIGHT[linker])

        if self.include_derived:
            # === DERIVED FEATURES ===

            # 1. Metal-linker compatibility score
            # Metals with high coordination prefer multidentate linkers
            coord_match = self.COORDINATION_PREFERENCE[metal] / self.LINKER_DENTICITY[linker]
            features.append(coord_match)

            # 2. Estimated pore size (linker length - 2 * metal radius)
            pore_size = self.LINKER_LENGTH[linker] - 2 * self.IONIC_RADII[metal]
            features.append(pore_size)

            # 3. Framework density estimate (molecular weight / volume)
            cell_volume = mof.get('volume', 1000.0)
            metal_mw = {
                'Zn': 65, 'Fe': 56, 'Ca': 40, 'Al': 27,
                'Ti': 48, 'Cu': 64, 'Zr': 91, 'Cr': 52,
                'Unknown': 60
            }
            total_mw = metal_mw.get(metal, 60) + self.LINKER_MOLECULAR_WEIGHT[linker]
            framework_density = total_mw / (cell_volume * 1.66)  # g/cm³
            features.append(framework_density)

            # 4. Linker/cell ratio (indicator of framework topology)
            avg_cell = (mof.get('cell_a', 10.0) +
                       mof.get('cell_b', 10.0) +
                       mof.get('cell_c', 10.0)) / 3
            linker_cell_ratio = self.LINKER_LENGTH[linker] / avg_cell
            features.append(linker_cell_ratio)

            # 5. Volumetric efficiency (volume per molecular weight)
            vol_per_mw = cell_volume / total_mw
            features.append(vol_per_mw)

            # 6. Metal-linker electronic compatibility
            # Hard metals (high Lewis acidity) prefer rigid linkers
            electronic_match = self.LEWIS_ACIDITY[metal] * self.LINKER_RIGIDITY[linker]
            features.append(electronic_match)

            # 7. CO2 affinity proxy
            # CO2 is a linear molecule, prefers:
            # - Open metal sites (high Lewis acidity)
            # - Aromatic rings (pi-pi interactions)
            # - Appropriate pore size (5-15 Angstroms)
            co2_affinity = (
                self.LEWIS_ACIDITY[metal] * 0.3 +
                self.LINKER_AROMATICITY[linker] * 0.2 +
                (1.0 if 5.0 < pore_size < 15.0 else 0.3) * 0.5
            )
            features.append(co2_affinity)

            # 8. Framework flexibility score
            # Rigid linkers + small metals = rigid framework
            flexibility = (1.0 - self.LINKER_RIGIDITY[linker]) * self.IONIC_RADII[metal]
            features.append(flexibility)

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        names = [
            'electronegativity',
            'ionic_radius',
            'oxidation_state',
            'coordination_preference',
            'lewis_acidity',
            'linker_length',
            'linker_denticity',
            'linker_rigidity',
            'linker_aromaticity',
            'linker_molecular_weight',
        ]

        if self.include_derived:
            names.extend([
                'coord_linker_match',
                'estimated_pore_size',
                'framework_density',
                'linker_cell_ratio',
                'volumetric_efficiency',
                'electronic_compatibility',
                'co2_affinity_proxy',
                'framework_flexibility',
            ])

        return names

    def featurize_batch(self, mofs: List[Dict]) -> np.ndarray:
        """Featurize a batch of MOFs"""
        return np.array([self.featurize(mof) for mof in mofs])


if __name__ == '__main__':
    """Test chemistry featurizer"""

    print("="*70)
    print("CHEMISTRY-INFORMED FEATURIZER TEST")
    print("="*70 + "\n")

    featurizer = ChemistryFeaturizer(include_derived=True)

    # Test MOFs
    test_mofs = [
        {
            'metal': 'Zn',
            'linker': 'terephthalic acid',
            'cell_a': 15.0,
            'cell_b': 15.0,
            'cell_c': 15.0,
            'volume': 3375.0
        },
        {
            'metal': 'Cu',
            'linker': '2,6-naphthalenedicarboxylic acid',
            'cell_a': 18.0,
            'cell_b': 18.0,
            'cell_c': 18.0,
            'volume': 5832.0
        },
        {
            'metal': 'Fe',
            'linker': 'trimesic acid',
            'cell_a': 12.0,
            'cell_b': 12.0,
            'cell_c': 12.0,
            'volume': 1728.0
        },
    ]

    feature_names = featurizer.get_feature_names()
    print(f"Features ({len(feature_names)} dimensions):\n")

    for i, mof in enumerate(test_mofs, 1):
        print(f"MOF {i}: {mof['metal']} + {mof['linker']}")

        features = featurizer.featurize(mof)

        print(f"  Shape: {features.shape}")
        print(f"\n  Key features:")
        for j, (name, val) in enumerate(zip(feature_names, features)):
            if j < 10 or name in ['estimated_pore_size', 'co2_affinity_proxy', 'framework_density']:
                print(f"    {name:30s}: {val:.3f}")
        print()

    print("="*70)
    print("Batch featurization:")
    batch_features = featurizer.featurize_batch(test_mofs)
    print(f"  Shape: {batch_features.shape}")
    print(f"  Mean: {batch_features.mean(axis=0)[:5]}")
    print(f"  Std:  {batch_features.std(axis=0)[:5]}")

    print("\n✓ Chemistry featurizer working!")
    print("\nThese features capture chemical intuition that simple")
    print("one-hot encoding misses, helping the surrogate generalize")
    print("to novel MOF structures.")
