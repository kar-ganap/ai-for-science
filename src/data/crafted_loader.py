"""
CRAFTED Database Loader

Loads MOF geometric properties and CO2 adsorption isotherms from CRAFTED database.
Computes uncertainty across force fields and charge schemes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import re


class CRAFTEDLoader:
    """Load and process CRAFTED MOF database"""

    def __init__(self, crafted_dir: Path):
        """
        Initialize loader

        Args:
            crafted_dir: Path to CRAFTED-2.0.0 directory
        """
        self.crafted_dir = Path(crafted_dir)
        self.geometric_file = self.crafted_dir / "RAC_DBSCAN" / "CRAFTED_MOF_geometric.csv"
        self.isotherm_dir = self.crafted_dir / "ISOTHERM_FILES"
        self.cif_dir = self.crafted_dir / "CIF_FILES"

        # Load geometric properties
        self.geometric_df = pd.read_csv(self.geometric_file)
        print(f"Loaded {len(self.geometric_df)} MOFs with geometric properties")

    def load_isotherm(self, mof_id: str, charge_scheme: str,
                     forcefield: str, gas: str = 'CO2',
                     temperature: int = 298) -> pd.DataFrame:
        """
        Load single isotherm file

        Args:
            mof_id: MOF identifier (e.g., '05000N2', 'ABUWOJ')
            charge_scheme: Charge scheme (DDEC, EQeq, MPNN, NEUTRAL, PACMOF, Qeq)
            forcefield: Force field (UFF, DREIDING)
            gas: Gas type (CO2, N2)
            temperature: Temperature in K (273, 298, 323)

        Returns:
            DataFrame with columns: pressure, uptake, error
        """
        # Convert framework name to file ID if needed
        if not mof_id[0].isdigit():
            # Framework name like 'ABUWOJ' - need to find corresponding file ID
            # This mapping would need to be built from CIF filenames
            # For now, return None if we can't find it
            return None

        filename = f"{charge_scheme}_{mof_id}_{forcefield}_{gas}_{temperature}.csv"
        filepath = self.isotherm_dir / filename

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath, comment='#',
                        names=['pressure', 'uptake', 'error'])
        return df

    def get_co2_uptake_at_pressure(self, mof_id: str, target_pressure: float = 1e5,
                                   charge_scheme: str = 'Qeq', forcefield: str = 'UFF',
                                   temperature: int = 298) -> Optional[float]:
        """
        Get CO2 uptake at specific pressure

        Args:
            mof_id: MOF identifier
            target_pressure: Target pressure in Pa (default: 1e5 = 1 bar)
            charge_scheme: Charge scheme
            forcefield: Force field
            temperature: Temperature in K

        Returns:
            CO2 uptake in mol/kg (or None if not available)
        """
        isotherm = self.load_isotherm(mof_id, charge_scheme, forcefield,
                                      'CO2', temperature)
        if isotherm is None:
            return None

        # Find closest pressure point or interpolate
        idx = (isotherm['pressure'] - target_pressure).abs().idxmin()
        return isotherm.loc[idx, 'uptake']

    def compute_uncertainty(self, mof_id: str, target_pressure: float = 1e5,
                           temperature: int = 298) -> Tuple[float, float]:
        """
        Compute CO2 uptake mean and uncertainty across force fields and charge schemes

        Args:
            mof_id: MOF identifier
            target_pressure: Target pressure in Pa
            temperature: Temperature in K

        Returns:
            (mean_uptake, std_uptake) in mol/kg
        """
        force_fields = ['UFF', 'DREIDING']
        charge_schemes = ['DDEC', 'EQeq', 'MPNN', 'NEUTRAL', 'PACMOF', 'Qeq']

        uptakes = []
        for ff in force_fields:
            for cs in charge_schemes:
                uptake = self.get_co2_uptake_at_pressure(
                    mof_id, target_pressure, cs, ff, temperature
                )
                if uptake is not None:
                    uptakes.append(uptake)

        if len(uptakes) == 0:
            return None, None

        return np.mean(uptakes), np.std(uptakes)

    def extract_metal_from_cif(self, mof_id: str, charge_scheme: str = 'Qeq') -> Optional[str]:
        """
        Extract metal type from CIF file

        Args:
            mof_id: MOF identifier
            charge_scheme: Which charge scheme CIF to use

        Returns:
            Primary metal symbol (e.g., 'Zn', 'Cu', 'Zr')
        """
        cif_path = self.cif_dir / charge_scheme / f"{mof_id}.cif"

        if not cif_path.exists():
            return None

        # Read CIF and extract metal atoms
        metals = set()
        common_metals = {'Zn', 'Cu', 'Zr', 'Co', 'Ni', 'Fe', 'Cr', 'Mn', 'Mg',
                        'Al', 'Cd', 'Ca', 'V', 'Ti', 'Sc'}

        with open(cif_path, 'r') as f:
            for line in f:
                # Look for atom site labels like _atom_site_type_symbol
                if line.strip().startswith('_atom_site_type_symbol'):
                    continue

                # Check for metal symbols in the line
                for metal in common_metals:
                    if metal in line:
                        metals.add(metal)

        if len(metals) == 0:
            return None

        # Return most common MOF metal if multiple found
        # Priority: Zn > Cu > Zr > others
        priority = ['Zn', 'Cu', 'Zr', 'Co', 'Fe', 'Cr', 'Al']
        for metal in priority:
            if metal in metals:
                return metal

        return list(metals)[0]

    def build_dataset(self, target_pressure: float = 1e5,
                     temperature: int = 298,
                     min_methods: int = 6) -> pd.DataFrame:
        """
        Build complete dataset for Economic Active Learning

        Args:
            target_pressure: Pressure for CO2 uptake (default: 1 bar)
            temperature: Temperature for isotherms (default: 298K)
            min_methods: Minimum number of FF/charge combinations required

        Returns:
            DataFrame with features:
                - mof_name: Framework name
                - mof_id: File identifier
                - LCD: Largest cavity diameter (D_fs)
                - PLD: Pore limiting diameter (D_is)
                - ASA: Accessible surface area (m^2/g)
                - density: Framework density (g/cm^3)
                - void_fraction: Accessible volume fraction
                - co2_uptake_mean: Mean CO2 uptake (mol/kg)
                - co2_uptake_std: Std CO2 uptake (mol/kg)
                - n_methods: Number of methods used
                - metal: Primary metal type
        """
        print(f"\nBuilding dataset at {target_pressure/1e5:.1f} bar, {temperature}K")
        print("=" * 70)

        # Map framework names to file IDs by checking CIF directory
        print("Mapping framework names to file IDs...")
        cif_files = list((self.cif_dir / 'Qeq').glob('*.cif'))
        file_id_to_name = {}

        for cif_file in cif_files:
            file_id = cif_file.stem  # e.g., '05000N2'

            # Read first few lines to find framework name
            with open(cif_file, 'r') as f:
                for line in f:
                    if '_chemical_name_common' in line or '_cell_formula_units_Z' in line:
                        # Extract framework name (this is approximate)
                        continue
                    if line.startswith('data_'):
                        name = line.replace('data_', '').strip()
                        file_id_to_name[file_id] = name
                        break

        print(f"Found {len(file_id_to_name)} file ID mappings")

        # Create reverse mapping
        name_to_file_id = {v: k for k, v in file_id_to_name.items()}

        # Build dataset
        rows = []
        for idx, row in self.geometric_df.iterrows():
            mof_name = row['FrameworkName']

            # Find file ID
            file_id = name_to_file_id.get(mof_name)
            if file_id is None:
                # Try direct match
                potential_files = list((self.cif_dir / 'Qeq').glob(f'*{mof_name}*.cif'))
                if len(potential_files) > 0:
                    file_id = potential_files[0].stem
                else:
                    continue

            # Compute uncertainty across methods
            mean_uptake, std_uptake = self.compute_uncertainty(
                file_id, target_pressure, temperature
            )

            if mean_uptake is None:
                continue

            # Count how many methods were available
            force_fields = ['UFF', 'DREIDING']
            charge_schemes = ['DDEC', 'EQeq', 'MPNN', 'NEUTRAL', 'PACMOF', 'Qeq']
            n_methods = 0
            for ff in force_fields:
                for cs in charge_schemes:
                    uptake = self.get_co2_uptake_at_pressure(
                        file_id, target_pressure, cs, ff, temperature
                    )
                    if uptake is not None:
                        n_methods += 1

            if n_methods < min_methods:
                continue

            # Extract metal
            metal = self.extract_metal_from_cif(file_id)

            # Build row
            rows.append({
                'mof_name': mof_name,
                'mof_id': file_id,
                'LCD': row['D_fs'],  # Largest free sphere = LCD
                'PLD': row['D_is'],  # Largest included sphere = PLD
                'ASA': row['ASA_m^2/g'],
                'density': row['Density'],
                'void_fraction': row['AV_Volume_fraction'],
                'pore_volume': row['AV_cm^3/g'],
                'co2_uptake_mean': mean_uptake,
                'co2_uptake_std': std_uptake,
                'n_methods': n_methods,
                'metal': metal if metal else 'Unknown'
            })

            if len(rows) % 50 == 0:
                print(f"Processed {len(rows)} MOFs...")

        df = pd.DataFrame(rows)

        print("\n" + "=" * 70)
        print(f"✅ Dataset built: {len(df)} MOFs")
        print(f"\nStatistics:")
        print(f"  CO2 uptake:  {df['co2_uptake_mean'].mean():.2f} ± {df['co2_uptake_mean'].std():.2f} mol/kg")
        print(f"  Uncertainty: {df['co2_uptake_std'].mean():.3f} ± {df['co2_uptake_std'].std():.3f} mol/kg")
        print(f"  Metals found: {df['metal'].value_counts().to_dict()}")
        print(f"  Methods per MOF: {df['n_methods'].mean():.1f} (min: {df['n_methods'].min()}, max: {df['n_methods'].max()})")

        return df


if __name__ == '__main__':
    # Test the loader
    import sys
    from pathlib import Path

    project_root = Path(__file__).parents[2]
    crafted_dir = project_root / "data" / "CRAFTED-2.0.0"

    print("Testing CRAFTED Loader")
    print("=" * 70)

    loader = CRAFTEDLoader(crafted_dir)

    # Build dataset
    df = loader.build_dataset(target_pressure=1e5, temperature=298, min_methods=6)

    # Save
    output_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_file, index=False)

    print(f"\n📁 Saved to: {output_file}")
    print(f"\n✅ Data loader working!")
