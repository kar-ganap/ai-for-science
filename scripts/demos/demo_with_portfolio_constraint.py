"""
Demo with Portfolio Constraint

Ensures balanced allocation between real and generated MOFs
regardless of acquisition scores.
"""

import numpy as np
from pathlib import Path

# Simulate portfolio-constrained selection
def select_with_portfolio_constraint(
    acquisitions,
    sources,
    costs,
    budget=500,
    max_generated_pct=0.85,  # Cap at 85% generated
    min_generated_pct=0.70   # Floor at 70% generated
):
    """
    Select MOFs within budget with portfolio constraints

    Args:
        acquisitions: Array of acquisition scores
        sources: Array of source labels ('real' or 'generated')
        costs: Array of costs
        budget: Total budget
        max_generated_pct: Maximum % of generated MOFs
        min_generated_pct: Minimum % of generated MOFs

    Returns:
        selected_indices: List of selected indices
    """
    n_total = len(acquisitions)
    sorted_indices = np.argsort(acquisitions)[::-1]

    selected = []
    spent = 0
    n_generated = 0

    for idx in sorted_indices:
        if spent >= budget:
            break

        source = sources[idx]
        cost = costs[idx]

        # Check if adding this MOF would violate constraints
        current_n = len(selected)

        if source == 'generated':
            # Would this push us over max_generated_pct?
            new_gen_pct = (n_generated + 1) / (current_n + 1)
            if new_gen_pct > max_generated_pct and current_n >= 5:  # Allow flexibility early
                continue  # Skip this generated MOF
        else:  # real
            # Would this push us under min_generated_pct?
            new_gen_pct = n_generated / (current_n + 1)
            if new_gen_pct < min_generated_pct and current_n >= 5:
                continue  # Skip this real MOF

        # Check budget
        if spent + cost <= budget:
            selected.append(idx)
            spent += cost
            if source == 'generated':
                n_generated += 1

    return selected


# Demo
print("="*70)
print("PORTFOLIO-CONSTRAINED SELECTION")
print("="*70 + "\n")

# Generate sample data (simplified)
np.random.seed(42)
n_real = 100
n_generated = 20

# Real MOFs have slightly lower acquisition on average
real_acq = np.random.normal(0.19, 0.02, n_real)
# Generated MOFs have slightly higher acquisition
gen_acq = np.random.normal(0.21, 0.03, n_generated)

# Add exploration bonus
bonus = 1.0
gen_acq_bonus = gen_acq + bonus

# Combine
all_acq = np.concatenate([real_acq, gen_acq_bonus])
sources = ['real'] * n_real + ['generated'] * n_generated
costs = np.full(len(all_acq), 40)

print(f"Pool: {n_real} real, {n_generated} generated")
print(f"Budget: $500")
print(f"Exploration bonus: {bonus}")
print()

# Method 1: Pure acquisition-based (unconstrained)
sorted_idx = np.argsort(all_acq)[::-1]
selected_unconstrained = []
spent = 0
for idx in sorted_idx:
    if spent + costs[idx] <= 500:
        selected_unconstrained.append(idx)
        spent += costs[idx]

n_gen_uncon = sum(1 for i in selected_unconstrained if sources[i] == 'generated')
print("Unconstrained Selection:")
print(f"  Selected: {len(selected_unconstrained)} MOFs")
print(f"  Generated: {n_gen_uncon} ({100*n_gen_uncon/len(selected_unconstrained):.1f}%)")
print(f"  Real: {len(selected_unconstrained) - n_gen_uncon}")
print()

# Method 2: Portfolio-constrained (70-85% generated)
selected_constrained = select_with_portfolio_constraint(
    all_acq, sources, costs,
    budget=500,
    max_generated_pct=0.85,
    min_generated_pct=0.70
)

n_gen_con = sum(1 for i in selected_constrained if sources[i] == 'generated')
print("Portfolio-Constrained Selection (70-85% generated):")
print(f"  Selected: {len(selected_constrained)} MOFs")
print(f"  Generated: {n_gen_con} ({100*n_gen_con/len(selected_constrained):.1f}%)")
print(f"  Real: {len(selected_constrained) - n_gen_con}")
print()

print("="*70)
print("RECOMMENDATION")
print("="*70 + "\n")

print("Use portfolio constraints to ensure balanced allocation:")
print(f"  Target: 70-85% generated, 15-30% real")
print(f"  This hedges against:")
print(f"    - VAE generating poor MOFs")
print(f"    - Surrogate being overconfident")
print(f"    - Model failures")
print()
print("Implementation:")
print("  1. Sort by acquisition (with exploration bonus)")
print("  2. Select greedily, but skip if violates portfolio constraint")
print("  3. Ensures minimum diversity in validation set")
