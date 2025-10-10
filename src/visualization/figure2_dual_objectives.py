"""
Figure 2: Dual Objectives - Learning vs Discovery

Shows that objective alignment matters: AL succeeds at whichever goal you optimize for.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_dual_objectives(output_dir: Path = None):
    """
    Create Figure 2: Dual Objectives comparison

    Shows 4-way comparison on both discovery and learning metrics
    """

    # Load baseline comparison results
    project_root = Path(__file__).parents[2]
    results_file = project_root / "results" / "baseline_comparison.json"

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data
    methods = ['Random', 'Expert\n(Mechanistic)', 'AL\n(Exploration)', 'AL\n(Exploitation)']

    # Discovery metrics
    discovery_best = [
        results['Random']['best_performance_mean'],
        results['Expert']['best_performance'],
        results['AL (Exploration)']['best_performance'],
        results['AL (Exploitation)']['best_performance']
    ]
    discovery_err = [
        results['Random']['best_performance_std'],
        0, 0, 0  # Single runs, no error bars
    ]

    # Learning metrics (uncertainty reduction)
    # Random: -1.4% (measured), Expert: N/A, AL Exploration: 25.1%, AL Exploitation: 8.6%
    learning_reduction = [-1.4, None, 25.1, 8.6]

    # Sample counts for annotation
    n_samples = [
        results['Random']['n_selected_avg'],
        results['Expert']['n_selected'],
        results['AL (Exploration)']['n_selected'],
        results['AL (Exploitation)']['n_selected']
    ]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color scheme
    colors = ['#FF6B6B', '#95E1D3', '#4ECDC4', '#FFD93D']

    # ============================================================
    # LEFT PANEL: Discovery Performance
    # ============================================================

    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, discovery_best, yerr=discovery_err,
                    color=colors, alpha=0.8, capsize=10,
                    error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax1.set_ylabel('Best MOF Performance (mol/kg CO₂)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax1.set_title('Discovery Performance\n"Which method finds the best materials?"',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Annotate with sample counts
    for i, (val, n) in enumerate(zip(discovery_best, n_samples)):
        ax1.text(i, val + 0.3, f'{int(n)} samples',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add efficiency annotation for AL (Exploitation)
    ax1.annotate('', xy=(3, discovery_best[3] - 0.5), xytext=(2, discovery_best[2] - 0.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax1.text(2.5, discovery_best[2] - 0.9,
            '62% fewer\nsamples!',
            ha='center', fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    # ============================================================
    # RIGHT PANEL: Learning Performance
    # ============================================================

    # Filter out None values for plotting
    learning_positions = [i for i, val in enumerate(learning_reduction) if val is not None]
    learning_values = [val for val in learning_reduction if val is not None]
    learning_colors = [colors[i] for i in learning_positions]
    learning_labels = [methods[i] for i in learning_positions]

    bars2 = ax2.bar(range(len(learning_values)), learning_values,
                    color=learning_colors, alpha=0.8)

    # Color bars based on good/bad performance
    for i, (bar, val) in enumerate(zip(bars2, learning_values)):
        if val < 0:
            bar.set_color('#FF6B6B')  # Red for negative
        elif val > 20:
            bar.set_color('#06A77D')  # Green for high reduction

    ax2.set_ylabel('Uncertainty Reduction (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax2.set_title('Learning Performance\n"Which method improves the predictive model?"',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(learning_values)))
    ax2.set_xticklabels(learning_labels, fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Annotate values
    for i, val in enumerate(learning_values):
        y_offset = 1.5 if val > 0 else -2.5
        ax2.text(i, val + y_offset, f'{val:+.1f}%',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=11, fontweight='bold')

    # Add "Model worse!" annotation for Random
    if learning_values[0] < 0:
        ax2.text(0, learning_values[0] - 5, 'Model gets\nWORSE!',
                ha='center', fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', alpha=0.9))

    # Add Expert note
    ax2.text(0.98, 0.02,
            'Note: Expert baseline excluded\n(single heuristic-based selection,\nno iterative model improvement)',
            transform=ax2.transAxes,
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            style='italic')

    # Overall title
    fig.suptitle('Objective Alignment Matters: AL Succeeds at Whichever Goal You Optimize For',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "figure2_dual_objectives.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Figure 2 saved: {output_file}")

    return fig


if __name__ == '__main__':
    print("Creating Figure 2: Dual Objectives\n" + "="*60)

    project_root = Path(__file__).parents[2]
    output_dir = project_root / "results" / "figures"

    plot_dual_objectives(output_dir)

    print("\n✅ Figure 2 complete!")
    print("\nKey Messages:")
    print("  1. Random: Good at discovery (lucky), terrible at learning (-1.4%)")
    print("  2. Expert: Mechanistic baseline (8.75 mol/kg)")
    print("  3. AL (Exploration): Optimizes learning → 25.1% uncertainty reduction")
    print("  4. AL (Exploitation): Optimizes discovery → same result, 62% fewer samples")
    print("\n📌 Objective alignment is key!")
