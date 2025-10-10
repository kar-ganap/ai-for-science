"""
Visualization Module for Economic Active Learning

Creates publication-quality plots for:
1. Cost tracking over AL iterations
2. Uncertainty reduction (validates epistemic uncertainty)
3. Pareto frontier (performance vs synthesis cost)
4. Budget compliance and efficiency metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class EconomicALVisualizer:
    """
    Visualization suite for Economic Active Learning results
    """

    def __init__(self, history_df: pd.DataFrame, output_dir: Optional[Path] = None):
        """
        Initialize visualizer

        Args:
            history_df: DataFrame with AL iteration metrics
            output_dir: Directory to save plots (default: results/figures/)
        """
        self.history = history_df

        if output_dir is None:
            output_dir = Path(__file__).parents[2] / "results" / "figures"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cost_tracking(self, save: bool = True) -> plt.Figure:
        """
        Plot cost tracking over iterations

        Shows:
        - Cumulative validation cost
        - Cost per iteration
        - Budget compliance

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Economic Active Learning: Cost Tracking',
                     fontsize=16, fontweight='bold', y=0.995)

        iterations = self.history['iteration'].values

        # 1. Cumulative cost over iterations
        ax = axes[0, 0]
        ax.plot(iterations, self.history['cumulative_cost'],
                'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cumulative Validation Cost ($)', fontsize=12)
        ax.set_title('A. Total Discovery Budget Spent', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, self.history['cumulative_cost'])):
            ax.text(x, y + 2, f'${y:.0f}', ha='center', va='bottom', fontsize=9)

        # 2. Cost per iteration (with budget line)
        ax = axes[0, 1]
        iteration_costs = self.history['iteration_cost'].values
        ax.bar(iterations, iteration_costs, color='#A23B72', alpha=0.7, edgecolor='black')

        # Budget line (assume first iteration cost as budget)
        budget = iteration_costs[0] * 1.02  # 2% tolerance
        ax.axhline(y=budget, color='red', linestyle='--', linewidth=2, label=f'Budget: ${budget:.0f}')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Validation Cost per Iteration ($)', fontsize=12)
        ax.set_title('B. Budget Compliance', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, iteration_costs)):
            ax.text(x, y + 0.5, f'${y:.1f}', ha='center', va='bottom', fontsize=9)

        # 3. Average cost per sample
        ax = axes[1, 0]
        avg_costs = self.history['avg_cost_per_sample'].values
        ax.plot(iterations, avg_costs, 's-', linewidth=2, markersize=8, color='#F18F01')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Cost per MOF ($)', fontsize=12)
        ax.set_title('C. Cost Efficiency', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, avg_costs)):
            ax.text(x, y + 0.01, f'${y:.2f}', ha='center', va='bottom', fontsize=9)

        # 4. Cumulative MOFs validated
        ax = axes[1, 1]
        cumulative_validated = self.history['n_validated'].cumsum().values
        ax.plot(iterations, cumulative_validated, 'o-', linewidth=2, markersize=8, color='#6A994E')
        ax.fill_between(iterations, 0, cumulative_validated, alpha=0.3, color='#6A994E')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cumulative MOFs Validated', fontsize=12)
        ax.set_title('D. Discovery Progress', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, cumulative_validated)):
            ax.text(x, y + 1, f'{int(y)}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "cost_tracking.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig

    def plot_uncertainty_reduction(self, save: bool = True) -> plt.Figure:
        """
        Plot uncertainty reduction over iterations

        Validates epistemic uncertainty quantification

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Economic Active Learning: Uncertainty Reduction',
                     fontsize=16, fontweight='bold')

        iterations = self.history['iteration'].values
        mean_unc = self.history['mean_uncertainty'].values
        max_unc = self.history['max_uncertainty'].values

        # 1. Mean uncertainty over time
        ax = axes[0]
        ax.plot(iterations, mean_unc, 'o-', linewidth=2, markersize=8,
                color='#D62828', label='Mean Uncertainty')
        ax.fill_between(iterations, 0, mean_unc, alpha=0.3, color='#D62828')

        # Calculate reduction percentage
        initial_unc = mean_unc[0]
        final_unc = mean_unc[-1]
        reduction_pct = (initial_unc - final_unc) / initial_unc * 100

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Uncertainty (Ensemble Std)', fontsize=12)
        ax.set_title(f'A. Epistemic Uncertainty Reduction ({reduction_pct:.1f}%)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add reduction arrow
        ax.annotate('', xy=(iterations[-1], final_unc), xytext=(iterations[0], initial_unc),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))
        ax.text(iterations[len(iterations)//2], (initial_unc + final_unc)/2,
               f'↓ {reduction_pct:.1f}%', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # 2. Max uncertainty over time
        ax = axes[1]
        ax.plot(iterations, max_unc, 's-', linewidth=2, markersize=8,
                color='#F77F00', label='Max Uncertainty')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Max Uncertainty (Ensemble Std)', fontsize=12)
        ax.set_title('B. Highest Uncertainty in Pool', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "uncertainty_reduction.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig

    def plot_performance_discovery(self, save: bool = True) -> plt.Figure:
        """
        Plot best performance discovered over iterations

        Shows discovery progress

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Economic Active Learning: Performance Discovery',
                     fontsize=16, fontweight='bold')

        iterations = self.history['iteration'].values
        best_perf = self.history['best_predicted_performance'].values
        mean_perf = self.history['mean_predicted_performance'].values

        # 1. Best predicted performance
        ax = axes[0]
        ax.plot(iterations, best_perf, 'o-', linewidth=2, markersize=8, color='#06A77D')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('CO₂ Uptake (mol/kg)', fontsize=12)
        ax.set_title('A. Best MOF Discovered', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, best_perf)):
            ax.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

        # 2. Mean predicted performance
        ax = axes[1]
        ax.plot(iterations, mean_perf, 's-', linewidth=2, markersize=8, color='#118AB2')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('CO₂ Uptake (mol/kg)', fontsize=12)
        ax.set_title('B. Mean Pool Performance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "performance_discovery.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig

    def plot_training_growth(self, save: bool = True) -> plt.Figure:
        """
        Plot training set growth and pool depletion

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = self.history['iteration'].values
        n_train = self.history['n_train'].values
        n_pool = self.history['n_pool'].values

        ax.plot(iterations, n_train, 'o-', linewidth=2, markersize=8,
                color='#06A77D', label='Training Set Size')
        ax.plot(iterations, n_pool, 's-', linewidth=2, markersize=8,
                color='#D62828', label='Pool Size (Remaining)')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Number of MOFs', fontsize=12)
        ax.set_title('Active Learning Progress: Training Set Growth',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "training_growth.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig

    def create_summary_dashboard(self, save: bool = True) -> plt.Figure:
        """
        Create comprehensive summary dashboard

        All key metrics in one figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        fig.suptitle('Economic Active Learning: Complete Summary Dashboard',
                     fontsize=18, fontweight='bold', y=0.995)

        iterations = self.history['iteration'].values

        # Row 1: Cost Metrics
        # 1.1 Cumulative cost
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(iterations, self.history['cumulative_cost'], 'o-',
                linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Cost ($)')
        ax.set_title('Total Budget Spent', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 1.2 Cost per iteration
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(iterations, self.history['iteration_cost'],
               color='#A23B72', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost ($)')
        ax.set_title('Validation Cost per Iteration', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 1.3 Cost efficiency
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(iterations, self.history['avg_cost_per_sample'], 's-',
                linewidth=2, markersize=8, color='#F18F01')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost per MOF ($)')
        ax.set_title('Average Cost per Sample', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Row 2: Uncertainty & Learning
        # 2.1 Uncertainty reduction
        ax = fig.add_subplot(gs[1, 0])
        mean_unc = self.history['mean_uncertainty'].values
        reduction = (mean_unc[0] - mean_unc[-1]) / mean_unc[0] * 100
        ax.plot(iterations, mean_unc, 'o-', linewidth=2, markersize=8, color='#D62828')
        ax.fill_between(iterations, 0, mean_unc, alpha=0.3, color='#D62828')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Uncertainty')
        ax.set_title(f'Uncertainty Reduction ({reduction:.1f}%)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2.2 Training set growth
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(iterations, self.history['n_train'], 'o-',
                linewidth=2, markersize=8, color='#06A77D', label='Training')
        ax.plot(iterations, self.history['n_pool'], 's-',
                linewidth=2, markersize=8, color='#D62828', label='Pool')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of MOFs')
        ax.set_title('Training Set Growth', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2.3 Samples validated per iteration
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(iterations, self.history['n_validated'],
               color='#6A994E', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MOFs Validated')
        ax.set_title('Samples per Iteration', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Row 3: Performance Discovery
        # 3.1 Best performance
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(iterations, self.history['best_predicted_performance'], 'o-',
                linewidth=2, markersize=8, color='#06A77D')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CO₂ Uptake (mol/kg)')
        ax.set_title('Best MOF Performance', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3.2 Mean performance
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(iterations, self.history['mean_predicted_performance'], 's-',
                linewidth=2, markersize=8, color='#118AB2')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CO₂ Uptake (mol/kg)')
        ax.set_title('Mean Pool Performance', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3.3 Summary statistics (text box)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')

        # Calculate summary stats
        total_cost = self.history['cumulative_cost'].iloc[-1]
        total_validated = self.history['n_validated'].sum()
        avg_cost = total_cost / total_validated
        final_unc = mean_unc[-1]
        initial_unc = mean_unc[0]
        unc_reduction = (initial_unc - final_unc) / initial_unc * 100
        best_mof = self.history['best_predicted_performance'].max()

        summary_text = f"""
        ━━━━━━━━━━━━━━━━━━━━━━━
        SUMMARY STATISTICS
        ━━━━━━━━━━━━━━━━━━━━━━━

        Total Cost: ${total_cost:.2f}
        MOFs Validated: {int(total_validated)}
        Avg Cost/MOF: ${avg_cost:.2f}

        Uncertainty:
          Initial: {initial_unc:.3f}
          Final: {final_unc:.3f}
          Reduction: {unc_reduction:.1f}%

        Best MOF: {best_mof:.2f} mol/kg

        Iterations: {len(iterations)}
        Final Training Size: {int(self.history['n_train'].iloc[-1])}
        ━━━━━━━━━━━━━━━━━━━━━━━
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))

        if save:
            output_path = self.output_dir / "summary_dashboard.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig

    def generate_all_plots(self) -> Dict[str, plt.Figure]:
        """
        Generate all plots and save them

        Returns:
            Dictionary of figure names to Figure objects
        """
        print("\n" + "="*60)
        print("Generating Economic AL Visualizations")
        print("="*60 + "\n")

        figures = {}

        print("[1/5] Cost tracking...")
        figures['cost_tracking'] = self.plot_cost_tracking(save=True)

        print("[2/5] Uncertainty reduction...")
        figures['uncertainty_reduction'] = self.plot_uncertainty_reduction(save=True)

        print("[3/5] Performance discovery...")
        figures['performance_discovery'] = self.plot_performance_discovery(save=True)

        print("[4/5] Training set growth...")
        figures['training_growth'] = self.plot_training_growth(save=True)

        print("[5/5] Summary dashboard...")
        figures['summary_dashboard'] = self.create_summary_dashboard(save=True)

        print("\n" + "="*60)
        print(f"✅ All visualizations saved to: {self.output_dir}")
        print("="*60 + "\n")

        return figures


    def plot_pareto_frontier(self, mof_data: pd.DataFrame,
                            selected_indices: Optional[List[int]] = None,
                            save: bool = True) -> plt.Figure:
        """
        Plot Pareto frontier of performance vs synthesis cost

        Args:
            mof_data: DataFrame with columns: co2_uptake_mean, synthesis_cost
            selected_indices: Indices of MOFs selected by AL (highlight)
            save: Whether to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # All MOFs (gray)
        ax.scatter(mof_data['synthesis_cost'], mof_data['co2_uptake_mean'],
                  s=30, alpha=0.4, color='lightgray', label='All MOFs')

        # Compute Pareto frontier (maximize uptake, minimize cost)
        # For each MOF, check if any other MOF dominates it
        is_pareto = np.ones(len(mof_data), dtype=bool)
        costs = mof_data['synthesis_cost'].values
        performance = mof_data['co2_uptake_mean'].values

        for i in range(len(mof_data)):
            # Check if dominated: another MOF has lower cost AND higher performance
            is_dominated = np.any(
                (costs < costs[i]) & (performance > performance[i])
            )
            is_pareto[i] = not is_dominated

        pareto_mofs = mof_data[is_pareto]

        # Plot Pareto frontier
        ax.scatter(pareto_mofs['synthesis_cost'], pareto_mofs['co2_uptake_mean'],
                  s=100, alpha=0.8, color='#D62828', marker='D',
                  edgecolors='black', linewidths=1.5,
                  label=f'Pareto Frontier ({is_pareto.sum()} MOFs)', zorder=5)

        # Highlight selected MOFs if provided
        if selected_indices is not None:
            selected = mof_data.iloc[selected_indices]
            ax.scatter(selected['synthesis_cost'], selected['co2_uptake_mean'],
                      s=150, alpha=0.9, color='#06A77D', marker='*',
                      edgecolors='black', linewidths=1.5,
                      label=f'Selected by AL ({len(selected_indices)} MOFs)', zorder=6)

        ax.set_xlabel('Synthesis Cost ($/gram)', fontsize=13)
        ax.set_ylabel('CO₂ Uptake (mol/kg)', fontsize=13)
        ax.set_title('Pareto Frontier: Performance vs Economic Viability',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Add quadrant labels
        mid_cost = mof_data['synthesis_cost'].median()
        mid_perf = mof_data['co2_uptake_mean'].median()

        ax.axhline(y=mid_perf, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=mid_cost, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Add text annotations for quadrants
        ax.text(0.02, 0.98, 'Low Cost\nHigh Performance\n(TARGET)',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.text(0.98, 0.02, 'High Cost\nLow Performance\n(AVOID)',
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "pareto_frontier.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {output_path}")

        return fig


if __name__ == '__main__':
    # Test with integration test results
    print("Testing Economic AL Visualizer\n" + "="*60)

    # Load results
    project_root = Path(__file__).parents[2]
    results_file = project_root / "results" / "economic_al_crafted_integration.csv"
    mof_data_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run the integration test first:")
        print("   uv run python tests/test_economic_al_crafted.py")
        exit(1)

    history_df = pd.read_csv(results_file)
    print(f"✓ Loaded {len(history_df)} iterations of results\n")

    # Create visualizer
    visualizer = EconomicALVisualizer(history_df)

    # Generate all plots
    figures = visualizer.generate_all_plots()

    # Generate Pareto frontier if MOF data available with costs
    if mof_data_file.exists():
        mof_data = pd.read_csv(mof_data_file)
        if 'synthesis_cost' in mof_data.columns and 'co2_uptake_mean' in mof_data.columns:
            print("\n[6/6] Pareto frontier...")
            figures['pareto_frontier'] = visualizer.plot_pareto_frontier(mof_data, save=True)
        else:
            print(f"\n⚠️  MOF data missing synthesis_cost column")
            print("   Skipping Pareto frontier plot")
            print("   Run dual-cost integration to add cost data")
    else:
        print(f"\n⚠️  MOF data not found: {mof_data_file}")
        print("   Skipping Pareto frontier plot")

    print("\n📊 Visualization test complete!")
    print(f"   Generated {len(figures)} figures")
    print(f"   Output directory: {visualizer.output_dir}")
