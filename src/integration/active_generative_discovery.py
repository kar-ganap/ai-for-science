"""
Active Generative Discovery: Tight Coupling of VAE + Economic AL

This module implements the TIGHT COUPLING integration where:
1. Economic AL identifies promising regions
2. VAE generates candidates in those regions
3. Generated MOFs compete with real MOFs in AL selection pool
4. Loop continues: AL learns → VAE generates → AL selects → Validate → Repeat

Key Innovation: Generation happens INSIDE the AL loop, guided by what AL learns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add paths
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root / "src" / "generation"))
sys.path.insert(0, str(project_root / "src" / "cost"))

from dual_conditional_vae import DualConditionalMOFGenerator
from estimator import MOFCostEstimator


class ActiveGenerativeDiscovery:
    """
    Active Generative Discovery Engine

    Tightly integrates VAE generation with Economic AL for
    dynamic search space expansion.
    """

    def __init__(self,
                 vae_model_path: Path,
                 cost_estimator: Optional[MOFCostEstimator] = None,
                 n_generate_per_iteration: int = 100,
                 temperature: float = 4.0):
        """
        Initialize Active Generative Discovery

        Args:
            vae_model_path: Path to trained dual-conditional VAE
            cost_estimator: MOF cost estimator (creates default if None)
            n_generate_per_iteration: Number of candidates to generate per AL iteration
            temperature: Sampling temperature for diversity
        """
        self.vae_model_path = vae_model_path
        self.n_generate = n_generate_per_iteration
        self.temperature = temperature

        # Load VAE
        print(f"Loading VAE from {vae_model_path}...")
        self.vae = DualConditionalMOFGenerator(use_geom_features=False)
        self.vae.load(vae_model_path)
        print("✓ VAE loaded\n")

        # Cost estimator
        self.cost_estimator = cost_estimator or MOFCostEstimator()

        # Statistics tracking
        self.stats = {
            'iterations': 0,
            'total_generated': 0,
            'total_unique': 0,
            'total_novel': 0,
            'generation_history': []
        }

    def extract_promising_targets(self,
                                   validated_mofs: pd.DataFrame,
                                   target_percentile: float = 90) -> Tuple[float, float]:
        """
        Extract promising (CO2, Cost) targets from validated MOFs

        Strategy: Target the high-performance, low-cost region

        Args:
            validated_mofs: DataFrame of MOFs validated so far
            target_percentile: Percentile of performance to target

        Returns:
            (target_co2, target_cost): Target values for generation
        """
        # Target high CO2 (e.g., 90th percentile of validated performance)
        target_co2 = np.percentile(validated_mofs['co2_uptake_mean'], target_percentile)

        # Target low cost (e.g., 30th percentile of validated costs)
        # Inverted percentile since we want LOW cost
        target_cost = np.percentile(validated_mofs['synthesis_cost'], 100 - target_percentile)

        # Add small boost to encourage exploration beyond current best
        co2_range = validated_mofs['co2_uptake_mean'].max() - validated_mofs['co2_uptake_mean'].min()
        target_co2 += 0.15 * co2_range  # Aim 15% beyond current high performers

        # Clip to reasonable bounds
        target_co2 = np.clip(target_co2, 4.0, 12.0)
        target_cost = np.clip(target_cost, 0.7, 1.0)

        return target_co2, target_cost

    def generate_candidates(self,
                           target_co2: float,
                           target_cost: float,
                           min_compositional_diversity: float = 0.3) -> List[Dict]:
        """
        Generate MOF candidates using dual-conditional VAE

        Args:
            target_co2: Target CO2 uptake (mol/kg)
            target_cost: Target synthesis cost ($/g)
            min_compositional_diversity: Minimum fraction of unique (metal, linker) pairs

        Returns:
            List of generated MOF candidates
        """
        print(f"\n{'='*70}")
        print("GENERATIVE DISCOVERY")
        print(f"{'='*70}")
        print(f"Target region: CO2={target_co2:.2f} mol/kg, Cost=${target_cost:.2f}/g\n")

        candidates = self.vae.generate_candidates(
            n_candidates=self.n_generate,
            target_co2=target_co2,
            target_cost=target_cost,
            temperature=self.temperature,
            min_compositional_diversity=min_compositional_diversity
        )

        return candidates

    def deduplicate_candidates(self,
                               candidates: List[Dict],
                               tolerance: float = 0.1) -> List[Dict]:
        """
        Deduplicate generated candidates by structure

        Two MOFs are duplicates if:
        - Same metal + linker
        - Cell parameters within tolerance

        Args:
            candidates: List of generated MOFs
            tolerance: Relative tolerance for cell parameter matching

        Returns:
            Deduplicated list
        """
        if not candidates:
            return []

        unique = []

        for candidate in candidates:
            is_duplicate = False

            for existing in unique:
                # Check composition
                if (candidate['metal'] == existing['metal'] and
                    candidate['linker'] == existing['linker']):

                    # Check cell parameters (relative difference)
                    cell_keys = ['cell_a', 'cell_b', 'cell_c', 'volume']

                    matches = []
                    for key in cell_keys:
                        c_val = candidate[key]
                        e_val = existing[key]
                        rel_diff = abs(c_val - e_val) / (e_val + 1e-8)
                        matches.append(rel_diff < tolerance)

                    if all(matches):
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(candidate)

        return unique

    def filter_novel_mofs(self,
                         candidates: List[Dict],
                         existing_database: pd.DataFrame,
                         tolerance: float = 0.1) -> List[Dict]:
        """
        Filter candidates to keep only novel MOFs not in database

        Args:
            candidates: Generated MOF candidates
            existing_database: DataFrame of existing MOFs
            tolerance: Tolerance for structure matching

        Returns:
            Filtered list of novel MOFs
        """
        if not candidates:
            return []

        # Load linker data
        linker_file = project_root / "data/processed/crafted_mofs_linkers.csv"
        linker_data = pd.read_csv(linker_file)

        # Merge to get metal-linker pairs for existing MOFs
        db_with_linkers = existing_database.merge(
            linker_data[['mof_id', 'linker']],
            on='mof_id',
            how='left'
        )

        novel = []

        for candidate in candidates:
            is_novel = True

            # Check against all existing MOFs
            for _, existing in db_with_linkers.iterrows():
                # Check composition
                if (candidate['metal'] == existing['metal'] and
                    candidate['linker'] == existing.get('linker', None)):

                    # Check cell parameters
                    cell_keys = ['cell_a', 'cell_b', 'cell_c', 'volume']

                    matches = []
                    for key in cell_keys:
                        c_val = candidate[key]
                        e_val = existing[key]
                        rel_diff = abs(c_val - e_val) / (e_val + 1e-8)
                        matches.append(rel_diff < tolerance)

                    if all(matches):
                        is_novel = False
                        break

            if is_novel:
                novel.append(candidate)

        return novel

    def estimate_costs(self, candidates: List[Dict]) -> List[Dict]:
        """
        Estimate validation and synthesis costs for generated MOFs

        Args:
            candidates: List of generated MOFs

        Returns:
            Candidates with cost estimates added
        """
        for candidate in candidates:
            # Synthesis cost
            synth_cost = self.cost_estimator.estimate_synthesis_cost({
                'metal': candidate['metal'],
                'linker': candidate['linker']
            })
            candidate['synthesis_cost'] = synth_cost['total_cost_per_gram']

            # Validation cost (simple estimate based on metal cost)
            # Base cost: $30-70 depending on metal complexity
            metal_validation_costs = {
                'Zn': 35,  # Common, easy
                'Fe': 38,  # Common, easy
                'Ca': 42,  # Common, moderate
                'Al': 40,  # Common, moderate
                'Ti': 58,  # Less common, harder
                'Cu': 45,  # Common, moderate
                'Zr': 65,  # Uncommon, expensive
                'Cr': 55,  # Moderate
                'Unknown': 50  # Default
            }

            base_validation = metal_validation_costs.get(candidate['metal'], 50)
            candidate['validation_cost'] = base_validation

            # Add source tag
            candidate['source'] = 'generated'

        return candidates

    def augment_al_pool(self,
                       validated_mofs: pd.DataFrame,
                       unvalidated_real_mofs: pd.DataFrame,
                       iteration: int = 0) -> Tuple[List[Dict], Dict]:
        """
        MAIN METHOD: Augment AL pool with generated candidates

        This is the tight coupling in action:
        1. Extract promising targets from AL's validated data
        2. Generate candidates in that region
        3. Deduplicate and filter for novelty
        4. Estimate costs
        5. Return candidates ready to merge with AL pool

        Args:
            validated_mofs: MOFs validated so far by AL
            unvalidated_real_mofs: Remaining real MOFs not yet validated
            iteration: Current AL iteration number

        Returns:
            (novel_candidates, stats): Generated MOFs and iteration statistics
        """
        print(f"\n{'='*70}")
        print(f"ACTIVE GENERATIVE DISCOVERY - Iteration {iteration}")
        print(f"{'='*70}\n")

        print(f"Validated MOFs: {len(validated_mofs)}")
        print(f"Unvalidated real MOFs: {len(unvalidated_real_mofs)}")

        # Step 1: Extract targets from validated data
        target_co2, target_cost = self.extract_promising_targets(validated_mofs)
        print(f"\n✓ Identified promising region:")
        print(f"  Target CO2: {target_co2:.2f} mol/kg")
        print(f"  Target Cost: ${target_cost:.2f}/g")

        # Step 2: Generate candidates with diversity enforcement
        raw_candidates = self.generate_candidates(
            target_co2,
            target_cost,
            min_compositional_diversity=0.3  # Enforce 30% unique compositions
        )
        print(f"\n✓ Generated {len(raw_candidates)} raw candidates")

        # Step 3: Deduplicate within generated set
        unique_candidates = self.deduplicate_candidates(raw_candidates)
        print(f"✓ After deduplication: {len(unique_candidates)} unique")

        # Step 4: Filter against existing database (both validated and unvalidated)
        all_existing = pd.concat([validated_mofs, unvalidated_real_mofs], ignore_index=True)
        novel_candidates = self.filter_novel_mofs(unique_candidates, all_existing)
        print(f"✓ After novelty filter: {len(novel_candidates)} novel MOFs")

        # Step 5: Estimate costs
        novel_candidates = self.estimate_costs(novel_candidates)
        print(f"✓ Cost estimates added")

        # Update statistics
        self.stats['iterations'] += 1
        self.stats['total_generated'] += len(raw_candidates)
        self.stats['total_unique'] += len(unique_candidates)
        self.stats['total_novel'] += len(novel_candidates)

        iter_stats = {
            'iteration': iteration,
            'target_co2': target_co2,
            'target_cost': target_cost,
            'n_generated': len(raw_candidates),
            'n_unique': len(unique_candidates),
            'n_novel': len(novel_candidates),
            'diversity_pct': 100 * len(unique_candidates) / len(raw_candidates) if raw_candidates else 0,
            'novelty_pct': 100 * len(novel_candidates) / len(unique_candidates) if unique_candidates else 0
        }

        self.stats['generation_history'].append(iter_stats)

        # Summary
        print(f"\n{'='*70}")
        print("GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Raw generated: {len(raw_candidates)}")
        print(f"Unique:        {len(unique_candidates)} ({iter_stats['diversity_pct']:.1f}% diversity)")
        print(f"Novel:         {len(novel_candidates)} ({iter_stats['novelty_pct']:.1f}% novelty)")
        print(f"\n✓ Ready to merge into AL selection pool!")

        return novel_candidates, iter_stats

    def get_statistics(self) -> Dict:
        """Get cumulative statistics"""
        return self.stats.copy()


def create_al_candidate_pool(real_mofs: List[Dict],
                             generated_mofs: List[Dict]) -> pd.DataFrame:
    """
    Merge real and generated MOFs into unified AL candidate pool

    Args:
        real_mofs: List of real MOF dictionaries
        generated_mofs: List of generated MOF dictionaries

    Returns:
        DataFrame ready for AL acquisition function evaluation
    """
    # Ensure all MOFs have 'source' tag
    for mof in real_mofs:
        if 'source' not in mof:
            mof['source'] = 'real'

    # Combine
    all_mofs = real_mofs + generated_mofs

    # Convert to DataFrame
    pool_df = pd.DataFrame(all_mofs)

    return pool_df


if __name__ == '__main__':
    """
    Test Active Generative Discovery
    """
    print("="*70)
    print("TESTING ACTIVE GENERATIVE DISCOVERY")
    print("="*70 + "\n")

    # Load data
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    mof_data = pd.read_csv(mof_file)

    # Simulate: first 50 MOFs are "validated", rest are unvalidated
    validated = mof_data.iloc[:50].copy()
    unvalidated = mof_data.iloc[50:].copy()

    print(f"Simulated state:")
    print(f"  Validated: {len(validated)} MOFs")
    print(f"  Unvalidated: {len(unvalidated)} MOFs\n")

    # Initialize Active Generative Discovery
    vae_model = project_root / "models/dual_conditional_mof_vae_compositional.pt"

    agd = ActiveGenerativeDiscovery(
        vae_model_path=vae_model,
        n_generate_per_iteration=100,
        temperature=4.0
    )

    # Run one iteration
    novel_mofs, stats = agd.augment_al_pool(
        validated_mofs=validated,
        unvalidated_real_mofs=unvalidated,
        iteration=1
    )

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}\n")

    print(f"Generated {len(novel_mofs)} novel MOFs for AL consideration")

    if novel_mofs:
        print(f"\nExample generated MOFs:")
        for i, mof in enumerate(novel_mofs[:3], 1):
            print(f"\n{i}. {mof['metal']} + {mof['linker']}")
            print(f"   Cell: ({mof['cell_a']:.2f}, {mof['cell_b']:.2f}, {mof['cell_c']:.2f}) Å")
            print(f"   Volume: {mof['volume']:.2f} Å³")
            print(f"   Synthesis cost: ${mof['synthesis_cost']:.2f}/g")
            print(f"   Validation cost: ${mof['validation_cost']:.2f}")

    print(f"\n✓ Active Generative Discovery is working!")
