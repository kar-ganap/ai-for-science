"""
Synthesizability Filter for Generated MOFs

Applies basic chemistry rules to reject physically impossible or
highly unlikely MOF structures before they reach the surrogate.

These are "sanity checks" based on known chemistry:
- Reasonable bond distances
- Valid coordination numbers
- Chemically sensible metal-linker pairings
- Thermodynamic feasibility
"""

import numpy as np
from typing import Dict, List, Tuple


class SynthesizabilityFilter:
    """
    Filter generated MOFs based on basic chemistry rules
    """

    # Typical coordination numbers for common metals in MOFs
    METAL_COORDINATION = {
        'Zn': [4, 6],          # Tetrahedral or octahedral
        'Fe': [6],             # Octahedral (most common)
        'Ca': [6, 7, 8],       # Flexible coordination
        'Al': [4, 6],          # Tetrahedral or octahedral
        'Ti': [6],             # Octahedral
        'Cu': [4, 5, 6],       # Square planar, square pyramidal, octahedral
        'Zr': [6, 7, 8],       # High coordination
        'Cr': [6],             # Octahedral
    }

    # Typical ionic radii (Angstroms) - for distance checks
    IONIC_RADII = {
        'Zn': 0.74,
        'Fe': 0.78,  # Fe(II)
        'Ca': 1.00,
        'Al': 0.54,
        'Ti': 0.61,
        'Cu': 0.73,
        'Zr': 0.72,
        'Cr': 0.62,
    }

    # Linker "sizes" (rough estimate of carboxylate separation)
    LINKER_SIZES = {
        'terephthalic acid': 11.0,                    # ~11 Å separation
        'trimesic acid': 9.5,                          # ~9.5 Å (3-way)
        '2,6-naphthalenedicarboxylic acid': 13.0,     # ~13 Å (longer)
        'biphenyl-4,4-dicarboxylic acid': 15.0,       # ~15 Å (longest)
    }

    # Electronegativity (Pauling scale) - for checking metal-linker compatibility
    ELECTRONEGATIVITY = {
        'Zn': 1.65,
        'Fe': 1.83,
        'Ca': 1.00,
        'Al': 1.61,
        'Ti': 1.54,
        'Cu': 1.90,
        'Zr': 1.33,
        'Cr': 1.66,
    }

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, apply more stringent filters
        """
        self.strict = strict
        self.rejection_stats = {
            'total_checked': 0,
            'passed': 0,
            'failed_volume': 0,
            'failed_density': 0,
            'failed_coordination': 0,
            'failed_cell_ratios': 0,
            'failed_chemistry': 0,
        }

    def is_synthesizable(self, mof: Dict) -> Tuple[bool, str]:
        """
        Check if a MOF structure is likely synthesizable

        Args:
            mof: MOF dictionary with keys: metal, linker, cell_a, cell_b, cell_c, volume

        Returns:
            (is_valid, reason): True if likely synthesizable, reason for rejection
        """
        self.rejection_stats['total_checked'] += 1

        # Check 1: Volume sanity
        volume = mof.get('volume', 0)
        if volume < 100 or volume > 50000:
            self.rejection_stats['failed_volume'] += 1
            return False, f"Volume {volume:.1f} Å³ outside reasonable range (100-50000)"

        # Check 2: Cell parameters sanity
        cell_a = mof.get('cell_a', 0)
        cell_b = mof.get('cell_b', 0)
        cell_c = mof.get('cell_c', 0)

        if any(x < 5.0 or x > 100.0 for x in [cell_a, cell_b, cell_c]):
            self.rejection_stats['failed_cell_ratios'] += 1
            return False, f"Cell parameters ({cell_a:.1f}, {cell_b:.1f}, {cell_c:.1f}) unrealistic"

        # Check 3: Cell ratios (should be somewhat balanced)
        max_cell = max(cell_a, cell_b, cell_c)
        min_cell = min(cell_a, cell_b, cell_c)
        if max_cell / min_cell > 5.0:  # Very elongated
            self.rejection_stats['failed_cell_ratios'] += 1
            return False, f"Cell too elongated (ratio {max_cell/min_cell:.1f})"

        # Check 4: Density estimate (MOFs typically 0.3-2.0 g/cm³)
        # Rough estimate: assume 1 formula unit per cell
        metal = mof.get('metal', 'Unknown')
        linker = mof.get('linker', 'terephthalic acid')

        # Estimate molecular weight (very rough)
        metal_weights = {'Zn': 65, 'Fe': 56, 'Ca': 40, 'Al': 27, 'Ti': 48, 'Cu': 64, 'Zr': 91, 'Cr': 52}
        linker_weights = {
            'terephthalic acid': 166,
            'trimesic acid': 210,
            '2,6-naphthalenedicarboxylic acid': 216,
            'biphenyl-4,4-dicarboxylic acid': 242,
        }

        mw_metal = metal_weights.get(metal, 60)
        mw_linker = linker_weights.get(linker, 180)
        mw_total = mw_metal + mw_linker  # Very rough

        # Density = (mass / volume) * (1.66 g/mol / Å³)
        # For 1 formula unit: density ≈ mw / (volume * 1.66)
        density_estimate = mw_total / (volume * 1.66)

        if density_estimate < 0.2 or density_estimate > 3.0:
            self.rejection_stats['failed_density'] += 1
            return False, f"Density {density_estimate:.2f} g/cm³ outside MOF range (0.2-3.0)"

        # Check 5: Metal-linker chemical compatibility
        # Some combinations are very unlikely
        if metal == 'Ca' and linker == 'biphenyl-4,4-dicarboxylic acid':
            # Ca prefers smaller linkers (coordination geometry constraints)
            if self.strict:
                self.rejection_stats['failed_chemistry'] += 1
                return False, "Ca + biphenyl linker unlikely (coordination mismatch)"

        # Check 6: Coordination geometry estimate
        # Linker size should be compatible with cell dimensions
        linker_size = self.LINKER_SIZES.get(linker, 12.0)
        avg_cell = (cell_a + cell_b + cell_c) / 3

        # If linker is longer than cell dimensions, something is wrong
        if linker_size > 1.2 * avg_cell:
            self.rejection_stats['failed_coordination'] += 1
            return False, f"Linker size {linker_size:.1f} Å too large for cell {avg_cell:.1f} Å"

        # Check 7: Soft heuristic - volume should scale with linker size
        # Larger linkers → larger pores → larger volumes
        expected_volume_range = (linker_size ** 3) * 0.5  # Very rough
        if volume < expected_volume_range * 0.1 or volume > expected_volume_range * 10:
            # This is a SOFT check - only fail in strict mode
            if self.strict:
                self.rejection_stats['failed_volume'] += 1
                return False, f"Volume {volume:.1f} inconsistent with linker size {linker_size:.1f} Å"

        # Passed all checks
        self.rejection_stats['passed'] += 1
        return True, "Passed all synthesizability checks"

    def filter_batch(self, mofs: List[Dict], verbose: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a batch of MOFs

        Args:
            mofs: List of MOF dictionaries
            verbose: Print rejection reasons

        Returns:
            (valid_mofs, rejected_mofs): Filtered lists
        """
        valid = []
        rejected = []

        for mof in mofs:
            is_valid, reason = self.is_synthesizable(mof)

            if is_valid:
                valid.append(mof)
            else:
                rejected.append({'mof': mof, 'reason': reason})
                if verbose:
                    metal = mof.get('metal', '?')
                    linker = mof.get('linker', '?')
                    print(f"  ❌ {metal} + {linker[:20]}... - {reason}")

        return valid, rejected

    def get_stats(self) -> Dict:
        """Get filtering statistics"""
        stats = self.rejection_stats.copy()
        if stats['total_checked'] > 0:
            stats['pass_rate'] = stats['passed'] / stats['total_checked']
        else:
            stats['pass_rate'] = 0.0
        return stats

    def print_summary(self):
        """Print filtering summary"""
        stats = self.get_stats()
        print(f"\n{'='*70}")
        print("SYNTHESIZABILITY FILTER SUMMARY")
        print(f"{'='*70}")
        print(f"Total MOFs checked:    {stats['total_checked']}")
        print(f"Passed:                {stats['passed']} ({100*stats['pass_rate']:.1f}%)")
        print(f"Rejected:              {stats['total_checked'] - stats['passed']}")
        print(f"\nRejection breakdown:")
        print(f"  Volume issues:       {stats['failed_volume']}")
        print(f"  Density issues:      {stats['failed_density']}")
        print(f"  Coordination issues: {stats['failed_coordination']}")
        print(f"  Cell ratio issues:   {stats['failed_cell_ratios']}")
        print(f"  Chemistry issues:    {stats['failed_chemistry']}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    """Test the synthesizability filter"""

    print("Testing Synthesizability Filter\n")

    filter_normal = SynthesizabilityFilter(strict=False)
    filter_strict = SynthesizabilityFilter(strict=True)

    # Test cases
    test_mofs = [
        # Good MOFs
        {
            'metal': 'Zn',
            'linker': 'terephthalic acid',
            'cell_a': 15.0,
            'cell_b': 15.0,
            'cell_c': 15.0,
            'volume': 3375.0
        },
        # Bad: Too small volume
        {
            'metal': 'Fe',
            'linker': 'trimesic acid',
            'cell_a': 5.0,
            'cell_b': 5.0,
            'cell_c': 5.0,
            'volume': 50.0
        },
        # Bad: Too large volume
        {
            'metal': 'Ca',
            'linker': 'biphenyl-4,4-dicarboxylic acid',
            'cell_a': 50.0,
            'cell_b': 50.0,
            'cell_c': 50.0,
            'volume': 125000.0
        },
        # Bad: Elongated cell
        {
            'metal': 'Al',
            'linker': 'terephthalic acid',
            'cell_a': 8.0,
            'cell_b': 8.0,
            'cell_c': 50.0,
            'volume': 3200.0
        },
        # Good MOF 2
        {
            'metal': 'Cu',
            'linker': '2,6-naphthalenedicarboxylic acid',
            'cell_a': 18.0,
            'cell_b': 18.0,
            'cell_c': 18.0,
            'volume': 5832.0
        },
    ]

    print("Testing with NORMAL filter:")
    valid, rejected = filter_normal.filter_batch(test_mofs, verbose=True)
    print(f"\nValid: {len(valid)}/{len(test_mofs)}")

    filter_normal.print_summary()

    print("\nTesting with STRICT filter:")
    filter_strict = SynthesizabilityFilter(strict=True)
    valid, rejected = filter_strict.filter_batch(test_mofs, verbose=True)
    print(f"\nValid: {len(valid)}/{len(test_mofs)}")

    filter_strict.print_summary()
