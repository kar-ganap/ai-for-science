"""
Analyze budget tradeoffs for Figure 2 baseline comparison
"""
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent
df = pd.read_csv(project_root / 'data/processed/crafted_mofs_co2_with_costs.csv')

print('='*70)
print('DATASET SUMMARY')
print('='*70)
print(f'Total MOFs: {len(df)}')
print(f'Training set: 100 MOFs')
print(f'Pool (unlabeled): {len(df) - 100} MOFs')

print('\n' + '='*70)
print('COST DISTRIBUTION (synthesis_cost per gram)')
print('='*70)
print(df['synthesis_cost'].describe())

print('\n' + '='*70)
print('CO2 UPTAKE DISTRIBUTION')
print('='*70)
print(df['co2_uptake_mean'].describe())

print('\n' + '='*70)
print('BUDGET ANALYSIS')
print('='*70)

# Median cost validation sampling
median_cost = df['synthesis_cost'].median()
print(f'\nMedian MOF cost: ${median_cost:.2f}/gram')
print(f'  - $50 budget:  ~{int(50/median_cost)} MOFs')
print(f'  - $500 budget: ~{int(500/median_cost)} MOFs')

# For 3 iterations
print(f'\nOver 3 iterations:')
print(f'  - $50 × 3 = $150:   ~{int(150/median_cost)} MOFs total ({100*150/median_cost/(len(df)-100):.1f}% of pool)')
print(f'  - $500 × 3 = $1500: ~{int(1500/median_cost)} MOFs total ({100*1500/median_cost/(len(df)-100):.1f}% of pool)')

print('\n' + '='*70)
print('POOL EXHAUSTION ANALYSIS')
print('='*70)
print(f'Starting pool size: {len(df) - 100} MOFs')
print(f'\nWith $50/iter (exploration strategy):')
print(f'  - Iter 1: ~63 validated, {len(df)-100-63} remain')
print(f'  - Iter 2: ~63 validated, {len(df)-100-126} remain')
print(f'  - Iter 3: ~63 validated, {len(df)-100-189} remain')
print(f'  → Pool still has {len(df)-100-189} MOFs (plenty left!)')

print(f'\nWith $500/iter (if random sampling):')
remaining = len(df) - 100 - int(500/median_cost)
if remaining < 0:
    print(f'  - Iter 1: ~{int(500/median_cost)} validated → WOULD EXHAUST POOL!')
    print(f'  - Cannot complete 3 iterations')
else:
    print(f'  - Iter 1: ~{int(500/median_cost)} validated, {remaining} remain')

print('\n' + '='*70)
print('AGD ACTUAL BEHAVIOR')
print('='*70)
print('AGD validation numbers from demo_results.json:')
print('  - Iter 1: 12 MOFs at $465 → $38.75/MOF avg')
print('  - Iter 2: 11 MOFs at $464 → $42.18/MOF avg')
print('  - Iter 3: 10 MOFs at $422 → $42.20/MOF avg')
print('  - Total: 33 MOFs validated at $1351 total')
print('\nWhy $40/MOF instead of $0.78/MOF?')
print('  → AGD uses exploration bonus + portfolio constraint')
print('  → Selects high-uncertainty, high-value candidates')
print('  → May select MOFs with expensive metals/linkers')

print('\n' + '='*70)
print('RECOMMENDATION')
print('='*70)
print('\n✓ Use $500/iteration budget for BOTH AGD and baseline because:')
print('  1. Matches original AGD experiment design')
print('  2. AGD only validates ~10-12 MOFs/iter (selective sampling)')
print('  3. Pool size (587 MOFs) can support 3 iterations')
print('  4. Fair comparison: same budget, same strategy (exploration)')
print('\n✗ $50/iteration would:')
print('  1. Validate ~63 MOFs/iter with exploration (too aggressive)')
print('  2. Not match AGD\'s selective, quality-focused approach')
print('  3. Exhaust pool faster (189 MOFs in 3 iters)')
print('  4. Miss expensive high-performers that AGD finds')

print('\n🎯 FINAL SETUP FOR FAIR COMPARISON:')
print('  - Budget: $500/iteration')
print('  - Strategy: exploration (cost_aware_uncertainty)')
print('  - Iterations: 3')
print('  - AGD: Real + Generated candidates')
print('  - Baseline: Real candidates only')
print('  - Everything else identical')
