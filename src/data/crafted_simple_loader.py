"""
Simple CRAFTED Loader - Direct file ID approach

Works directly with CRAFTED file IDs without needing framework name mapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings


class SimpleCRAFTEDLoader:
    """Simple loader working with file IDs directly"""

    def __init__(self, crafted_dir: Path):
        self.crafted_dir = Path(crafted_dir)
        self.isotherm_dir = self.crafted_dir / "ISOTHERM_FILES"
        self.cif_dir = self.crafted_dir / "CIF_FILES"

        # Get all MOF file IDs from CIF directory
        cif_files = list((self.cif_dir / 'Qeq').glob('*.cif'))
        self.all_mof_ids = [f.stem for f in cif_files]

        # Filter to just MOFs (exclude COFs)
        # MOFs have alphabetic refcode prefixes (e.g., ABUWOJ, AFITIT)
        # COFs have numeric IDs (e.g., 05000N2, 19001N2)
        self.mof_ids = [mid for mid in self.all_mof_ids
                       if mid[0].isalpha()]

        print(f"Found {len(self.all_mof_ids)} total structures")
        print(f"Filtered to {len(self.mof_ids)} MOFs")

    def load_isotherm(self, mof_id: str, charge_scheme: str, forcefield: str,
                     gas: str = 'CO2', temperature: int = 298) -> Optional[pd.DataFrame]:
        """Load isotherm file"""
        filename = f"{charge_scheme}_{mof_id}_{forcefield}_{gas}_{temperature}.csv"
        filepath = self.isotherm_dir / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath, comment='#',
                           names=['pressure', 'uptake', 'error'])
            return df
        except Exception as e:
            warnings.warn(f"Error loading {filename}: {e}")
            return None

    def get_uptake_at_pressure(self, mof_id: str, target_pressure: float = 1e5,
                               charge_scheme: str = 'Qeq', forcefield: str = 'UFF',
                               gas: str = 'CO2', temperature: int = 298) -> Optional[float]:
        """Get uptake at specific pressure"""
        isotherm = self.load_isotherm(mof_id, charge_scheme, forcefield, gas, temperature)
        if isotherm is None:
            return None

        # Find closest pressure
        idx = (isotherm['pressure'] - target_pressure).abs().idxmin()
        return isotherm.loc[idx, 'uptake']

    def compute_uncertainty(self, mof_id: str, target_pressure: float = 1e5,
                           temperature: int = 298) -> tuple:
        """Compute mean and std across all force fields and charge schemes"""
        force_fields = ['UFF', 'DREIDING']
        charge_schemes = ['DDEC', 'EQeq', 'MPNN', 'NEUTRAL', 'PACMOF', 'Qeq']

        uptakes = []
        for ff in force_fields:
            for cs in charge_schemes:
                uptake = self.get_uptake_at_pressure(
                    mof_id, target_pressure, cs, ff, 'CO2', temperature
                )
                if uptake is not None and uptake > 0:  # Filter out zeros
                    uptakes.append(uptake)

        if len(uptakes) == 0:
            return None, None, 0

        return np.mean(uptakes), np.std(uptakes), len(uptakes)

    def extract_metal_from_cif(self, mof_id: str, charge_scheme: str = 'Qeq') -> Optional[str]:
        """Extract primary metal from CIF"""
        cif_path = self.cif_dir / charge_scheme / f"{mof_id}.cif"

        if not cif_path.exists():
            return None

        metals = set()
        common_metals = {'Zn', 'Cu', 'Zr', 'Co', 'Ni', 'Fe', 'Cr', 'Mn', 'Mg',
                        'Al', 'Cd', 'Ca', 'V', 'Ti', 'Sc', 'Mo', 'W', 'Ag'}

        with open(cif_path, 'r') as f:
            in_atom_section = False
            for line in f:
                if '_atom_site_type_symbol' in line:
                    in_atom_section = True
                    continue

                if in_atom_section:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        atom_type = parts[1]  # Second column is atom type
                        if atom_type in common_metals:
                            metals.add(atom_type)

        if len(metals) == 0:
            return 'Unknown'

        # Priority order for common MOF metals
        priority = ['Zn', 'Cu', 'Zr', 'Co', 'Fe', 'Cr', 'Al', 'Ni', 'Mn']
        for metal in priority:
            if metal in metals:
                return metal

        return list(metals)[0]

    def get_geometric_properties(self, mof_id: str, charge_scheme: str = 'Qeq') -> Optional[dict]:
        """
        Extract geometric properties from CIF file

        Simplified version - just basic cell parameters
        For full geometric properties, would need proper CIF parser
        """
        cif_path = self.cif_dir / charge_scheme / f"{mof_id}.cif"

        if not cif_path.exists():
            return None

        props = {}

        with open(cif_path, 'r') as f:
            for line in f:
                if '_cell_length_a' in line:
                    props['cell_a'] = float(line.split()[1])
                elif '_cell_length_b' in line:
                    props['cell_b'] = float(line.split()[1])
                elif '_cell_length_c' in line:
                    props['cell_c'] = float(line.split()[1])
                elif '_cell_volume' in line:
                    props['volume'] = float(line.split()[1])

        return props if props else None

    def build_dataset(self, target_pressure: float = 1e5,
                     temperature: int = 298,
                     min_methods: int = 6,
                     max_mofs: Optional[int] = None) -> pd.DataFrame:
        """
        Build dataset for Economic AL

        Args:
            target_pressure: Pressure for CO2 uptake (Pa)
            temperature: Temperature (K)
            min_methods: Minimum number of FF/charge combinations required
            max_mofs: Maximum MOFs to process (for testing)

        Returns:
            DataFrame with features and CO2 uptake
        """
        print(f"\nBuilding dataset: {target_pressure/1e5:.1f} bar, {temperature}K")
        print("=" * 70)

        mofs_to_process = self.mof_ids[:max_mofs] if max_mofs else self.mof_ids

        rows = []
        for i, mof_id in enumerate(mofs_to_process):
            # Compute uncertainty
            mean_uptake, std_uptake, n_methods = self.compute_uncertainty(
                mof_id, target_pressure, temperature
            )

            if mean_uptake is None or n_methods < min_methods:
                continue

            # Get metal
            metal = self.extract_metal_from_cif(mof_id)

            # Get geometric properties
            geom = self.get_geometric_properties(mof_id)

            row = {
                'mof_id': mof_id,
                'co2_uptake_mean': mean_uptake,
                'co2_uptake_std': std_uptake,
                'n_methods': n_methods,
                'metal': metal,
            }

            if geom:
                row.update(geom)

            rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(mofs_to_process)} MOFs, found {len(rows)} valid...")

        df = pd.DataFrame(rows)

        print("\n" + "=" * 70)
        print(f"✅ Dataset complete: {len(df)} MOFs")

        if len(df) > 0:
            print(f"\nCO2 Uptake Statistics (@ {target_pressure/1e5:.1f} bar, {temperature}K):")
            print(f"  Mean:   {df['co2_uptake_mean'].mean():.2f} ± {df['co2_uptake_mean'].std():.2f} mol/kg")
            print(f"  Range:  {df['co2_uptake_mean'].min():.2f} - {df['co2_uptake_mean'].max():.2f} mol/kg")
            print(f"\nUncertainty (across FF/charges):")
            print(f"  Mean:   {df['co2_uptake_std'].mean():.3f} mol/kg")
            print(f"  Max:    {df['co2_uptake_std'].max():.3f} mol/kg")
            print(f"\nMethods per MOF: {df['n_methods'].mean():.1f} (min: {df['n_methods'].min()}, max: {df['n_methods'].max()})")
            print(f"\nMetals found:")
            metal_counts = df['metal'].value_counts()
            for metal, count in metal_counts.head(10).items():
                print(f"  {metal}: {count}")

        return df


if __name__ == '__main__':
    from pathlib import Path

    project_root = Path(__file__).parents[2]
    crafted_dir = project_root / "data" / "CRAFTED-2.0.0"

    print("CRAFTED Simple Loader Test")
    print("=" * 70)

    loader = SimpleCRAFTEDLoader(crafted_dir)

    # Build dataset (test with first 100 MOFs)
    print("\nTesting with first 100 MOFs...")
    df_test = loader.build_dataset(target_pressure=1e5, temperature=298,
                                   min_methods=6, max_mofs=100)

    if len(df_test) > 0:
        print(f"\n✅ Test successful! Building full dataset...")

        # Build full dataset
        df_full = loader.build_dataset(target_pressure=1e5, temperature=298,
                                       min_methods=6)

        # Save
        output_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        df_full.to_csv(output_file, index=False)

        print(f"\n📁 Saved to: {output_file}")
        print(f"\nSample data:")
        print(df_full.head())
    else:
        print("\n❌ No valid MOFs found in test")
