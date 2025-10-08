"""
MOF Cost Estimator

Estimates synthesis cost based on reagent prices and typical stoichiometry.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class MOFCostEstimator:
    """Estimate synthesis cost for MOFs based on composition"""

    def __init__(self, reagent_db_path: Optional[str] = None):
        """
        Initialize cost estimator with reagent database

        Args:
            reagent_db_path: Path to reagent_prices.csv. If None, uses default location.
        """
        if reagent_db_path is None:
            # Default to data/reagent_prices.csv relative to project root
            project_root = Path(__file__).parents[2]
            reagent_db_path = project_root / "data" / "reagent_prices.csv"

        self.reagents = pd.read_csv(reagent_db_path)
        self.price_dict = {}
        self.reagent_info = {}

        # Build lookup dictionaries
        for _, row in self.reagents.iterrows():
            key = row['reagent'].lower()
            self.price_dict[key] = row['price_usd_per_g']
            self.reagent_info[key] = {
                'price': row['price_usd_per_g'],
                'cas': row['cas_number'],
                'supplier': row['supplier'],
                'category': row['category']
            }

    def estimate_synthesis_cost(self, mof_composition: Dict) -> Dict:
        """
        Estimate cost to synthesize 1g of MOF

        Args:
            mof_composition: Dict with keys:
                - metal: str (e.g., 'Zn', 'Cu', 'Zr')
                - linker: str (e.g., 'terephthalic acid', 'trimesic acid')
                - topology: str (optional, for future use)

        Returns:
            Dict with cost breakdown:
                - total_cost_per_gram: float
                - metal_contribution: float
                - linker_contribution: float
                - solvent_contribution: float
                - breakdown: dict with detailed info
        """
        metal = mof_composition.get('metal', 'unknown')
        linker = mof_composition.get('linker', 'unknown')

        # Lookup prices with fuzzy matching
        metal_salt = self._get_metal_salt_name(metal)
        metal_price = self._fuzzy_lookup(metal_salt, default=1.0)
        linker_price = self._fuzzy_lookup(linker, default=5.0)

        # Typical stoichiometry for 1g MOF yield
        # Based on literature (MOF-5, HKUST-1, UiO-66, etc.)
        # These are rough estimates - actual amounts vary by MOF
        metal_mass_g = self._estimate_metal_mass(metal)
        linker_mass_g = self._estimate_linker_mass(linker)

        # Solvent cost (typical: 10-20 mL DMF per gram of MOF)
        dmf_price = self._fuzzy_lookup('DMF', default=0.05)
        solvent_volume_ml = 15  # Average 15 mL per gram MOF
        solvent_cost = dmf_price * solvent_volume_ml  # DMF ~0.944 g/mL, so $/mL ≈ $/g

        # Calculate costs
        metal_cost = metal_price * metal_mass_g
        linker_cost = linker_price * linker_mass_g
        total = metal_cost + linker_cost + solvent_cost

        return {
            'total_cost_per_gram': total,
            'metal_contribution': metal_cost,
            'linker_contribution': linker_cost,
            'solvent_contribution': solvent_cost,
            'breakdown': {
                'metal': metal,
                'metal_salt': metal_salt,
                'metal_price_per_g': metal_price,
                'metal_mass_needed_g': metal_mass_g,
                'linker': linker,
                'linker_price_per_g': linker_price,
                'linker_mass_needed_g': linker_mass_g,
                'solvent_volume_ml': solvent_volume_ml,
                'solvent_price_per_ml': dmf_price
            }
        }

    def _get_metal_salt_name(self, metal: str) -> str:
        """Convert metal symbol to common salt name"""
        # Most MOFs use nitrate salts
        metal_lower = metal.lower()

        salt_map = {
            'zn': 'zinc nitrate hexahydrate',
            'cu': 'copper(ii) nitrate trihydrate',
            'zr': 'zirconium(iv) chloride',  # Zr typically uses chloride
            'co': 'cobalt(ii) nitrate hexahydrate',
            'ni': 'nickel(ii) nitrate hexahydrate',
            'mg': 'magnesium nitrate hexahydrate',
            'fe': 'iron(iii) nitrate nonahydrate',
            'cr': 'chromium(iii) nitrate nonahydrate',
            'mn': 'manganese(ii) nitrate tetrahydrate',
            'cd': 'cadmium nitrate tetrahydrate',
        }

        return salt_map.get(metal_lower, f'{metal} nitrate')

    def _estimate_metal_mass(self, metal: str) -> float:
        """
        Estimate metal salt mass needed for 1g MOF

        Based on typical MOF compositions:
        - MOF-5: Zn4O(BDC)3, ~60-70% by mass organic
        - HKUST-1: Cu3(BTC)2, ~50-60% by mass organic
        - UiO-66: Zr6O4(OH)4(BDC)6, ~40-50% by mass organic
        """
        metal_lower = metal.lower()

        # Typical mass ratios (conservative estimates accounting for yield)
        mass_map = {
            'zn': 0.15,   # Light metal, high organic content
            'cu': 0.20,   # Medium weight metal
            'zr': 0.25,   # Heavy metal but stable structure
            'co': 0.18,
            'ni': 0.18,
            'mg': 0.12,   # Very light metal
            'fe': 0.18,
            'cr': 0.18,
            'mn': 0.18,
            'cd': 0.22,   # Heavy metal
        }

        return mass_map.get(metal_lower, 0.15)  # Default to Zn-like

    def _estimate_linker_mass(self, linker: str) -> float:
        """
        Estimate linker mass needed for 1g MOF

        Typically 2-4x stoichiometric amount to ensure complete reaction
        """
        linker_lower = linker.lower()

        # Adjust based on linker size/reactivity
        if 'terephthalic' in linker_lower or 'bdc' in linker_lower:
            return 0.25  # Small linker, high loading
        elif 'trimesic' in linker_lower or 'btc' in linker_lower:
            return 0.30  # Tritopic, more needed
        elif 'biphenyl' in linker_lower or 'naphthalene' in linker_lower:
            return 0.35  # Extended linker, more mass
        else:
            return 0.25  # Default

    def _fuzzy_lookup(self, query: str, default: float = 5.0) -> float:
        """
        Find reagent price with fuzzy string matching

        Args:
            query: Reagent name to search for
            default: Default price if not found

        Returns:
            Price per gram in USD
        """
        query_lower = query.lower()

        # Exact match
        if query_lower in self.price_dict:
            return self.price_dict[query_lower]

        # Partial match (check if query is substring or vice versa)
        for key, price in self.price_dict.items():
            if query_lower in key or key in query_lower:
                return price

        # Default (conservative estimate)
        return default

    def estimate_batch_cost(self, mof_composition: Dict,
                           yield_percent: float = 70) -> Dict:
        """
        Estimate cost accounting for typical synthesis yields

        Args:
            mof_composition: MOF composition dict
            yield_percent: Expected yield (default 70%)

        Returns:
            Dict with adjusted costs
        """
        base_cost = self.estimate_synthesis_cost(mof_composition)

        # Adjust for yield
        actual_cost_per_gram = base_cost['total_cost_per_gram'] * (100 / yield_percent)

        return {
            **base_cost,
            'actual_cost_per_gram': actual_cost_per_gram,
            'yield_percent': yield_percent,
            'waste_cost': actual_cost_per_gram - base_cost['total_cost_per_gram']
        }

    def get_reagent_info(self, reagent_name: str) -> Optional[Dict]:
        """Get detailed information about a reagent"""
        query_lower = reagent_name.lower()

        # Try exact match first
        if query_lower in self.reagent_info:
            return self.reagent_info[query_lower]

        # Try partial match
        for key, info in self.reagent_info.items():
            if query_lower in key or key in query_lower:
                return info

        return None


if __name__ == '__main__':
    # Test the estimator
    estimator = MOFCostEstimator()

    print("Testing MOF Cost Estimator\n" + "="*50)

    # Test on known MOFs
    test_mofs = [
        {
            'name': 'MOF-5',
            'metal': 'Zn',
            'linker': 'terephthalic acid',
            'description': 'Classic Zn-BDC MOF'
        },
        {
            'name': 'HKUST-1',
            'metal': 'Cu',
            'linker': 'trimesic acid',
            'description': 'Cu-BTC MOF'
        },
        {
            'name': 'UiO-66',
            'metal': 'Zr',
            'linker': 'terephthalic acid',
            'description': 'Zr-BDC MOF (very stable)'
        },
    ]

    for mof in test_mofs:
        print(f"\n{mof['name']}: {mof['description']}")
        print("-" * 50)

        cost = estimator.estimate_synthesis_cost(mof)
        cost_adj = estimator.estimate_batch_cost(mof, yield_percent=70)

        print(f"  Base cost (100% yield): ${cost['total_cost_per_gram']:.2f}/g")
        print(f"    - Metal ({mof['metal']}): ${cost['metal_contribution']:.2f}")
        print(f"    - Linker: ${cost['linker_contribution']:.2f}")
        print(f"    - Solvent: ${cost['solvent_contribution']:.2f}")
        print(f"  Actual cost (70% yield): ${cost_adj['actual_cost_per_gram']:.2f}/g")
        print(f"  Cost-efficiency (if 10 mmol/g CO2): {10/cost_adj['actual_cost_per_gram']:.2f} mmol/$")

    print("\n" + "="*50)
    print("✅ Cost estimator ready for Economic AL integration!")
