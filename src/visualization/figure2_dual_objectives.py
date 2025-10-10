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

    # Color scheme - vibrant colors matching Figure 1 theme
    colors = ['#FF4444', '#FFA500', '#06A77D', '#9B59B6']  # Red, Orange, Green, Purple

    # ============================================================
    # LEFT PANEL: Discovery Performance
    # ============================================================

    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, discovery_best, yerr=discovery_err,
                    color=colors, alpha=0.9, capsize=10, edgecolor='black', linewidth=1.5,
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
    # Position arrow between the two bars, use navy to contrast with green bar
    arrow_y = discovery_best[2] - 1.8
    ax1.annotate('', xy=(3, arrow_y), xytext=(2, arrow_y),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color='navy'))
    # Position text well below arrow to avoid overlap
    ax1.text(2.5, arrow_y - 0.9,
            '62% fewer\nsamples!',
            ha='center', fontsize=11, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='navy', alpha=0.95, edgecolor='white', linewidth=2))

    # ============================================================
    # RIGHT PANEL: Learning Performance
    # ============================================================

    # Filter out None values for plotting
    learning_positions = [i for i, val in enumerate(learning_reduction) if val is not None]
    learning_values = [val for val in learning_reduction if val is not None]
    learning_colors = [colors[i] for i in learning_positions]
    learning_labels = [methods[i] for i in learning_positions]

    bars2 = ax2.bar(range(len(learning_values)), learning_values,
                    color=learning_colors, alpha=0.9, edgecolor='black', linewidth=1.5)

    # Color bars based on good/bad performance (keep vibrant colors)
    for i, (bar, val) in enumerate(zip(bars2, learning_values)):
        if val < 0:
            bar.set_color('#FF4444')  # Vibrant red for negative
        elif val > 20:
            bar.set_color('#06A77D')  # Green for high reduction
        elif val > 5:
            bar.set_color('#9B59B6')  # Purple for moderate

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
        y_offset = 1.5 if val > 0 else -4.0  # Increased offset for negative values to avoid axis overlap
        ax2.text(i, val + y_offset, f'{val:+.1f}%',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=11, fontweight='bold')

    # Add "Model gets WORSE!" annotation for Random
    # Position in lower left corner to avoid overlap with bar and value
    if learning_values[0] < 0:
        ax2.text(0.05, 0.08, 'Random makes\nmodel WORSE!',
                transform=ax2.transAxes,
                ha='left', va='bottom',
                fontsize=11, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#FF4444',
                         alpha=0.95, edgecolor='white', linewidth=2))

    # Add Expert note - position in upper right to avoid overlap
    ax2.text(0.97, 0.95,
            'Expert baseline excluded:\nSingle heuristic selection,\nno iterative improvement',
            transform=ax2.transAxes,
            fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC',
                     alpha=0.9, edgecolor='gray', linewidth=1),
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
