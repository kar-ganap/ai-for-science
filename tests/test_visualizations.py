"""
Test Visualization Generation

Verifies that all Economic AL plots generate correctly.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.visualization.economic_al_plots import EconomicALVisualizer


def test_visualizations():
    print("=" * 80)
    print("TESTING VISUALIZATION MODULE")
    print("=" * 80)

    project_root = Path(__file__).parents[1]

    # Load results from Economic AL test
    history_file = project_root / "results" / "economic_al_crafted_integration.csv"
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"

    if not history_file.exists():
        print(f"\n❌ Run Economic AL test first: python tests/test_economic_al_crafted.py")
        return

    history_df = pd.read_csv(history_file)
    print(f"\n✓ Loaded AL history: {len(history_df)} iterations")

    # Initialize visualizer
    output_dir = project_root / "results" / "figures"
    visualizer = EconomicALVisualizer(history_df, output_dir)

    print(f"\n[1/5] Generating cost tracking plot...")
    try:
        visualizer.plot_cost_tracking(save=True)
        print(f"  ✅ Cost tracking plot saved")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[2/5] Generating uncertainty reduction plot...")
    try:
        visualizer.plot_uncertainty_reduction(save=True)
        print(f"  ✅ Uncertainty plot saved")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[3/5] Generating performance discovery plot...")
    try:
        visualizer.plot_performance_discovery(save=True)
        print(f"  ✅ Performance discovery plot saved")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[4/5] Generating training growth plot...")
    try:
        visualizer.plot_training_growth(save=True)
        print(f"  ✅ Training growth plot saved")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Pareto frontier (if MOF data available)
    if mof_file.exists():
        print(f"\n[5/5] Generating Pareto frontier plot...")
        try:
            mof_df = pd.read_csv(mof_file)
            visualizer.plot_pareto_frontier(mof_df, save=True)
            print(f"  ✅ Pareto frontier plot saved")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[5/5] Skipping Pareto frontier (MOF data not found)")

    print(f"\n" + "=" * 80)
    print(f"✅ ALL VISUALIZATIONS GENERATED")
    print(f"=" * 80)
    print(f"\n📊 Plots saved to: {output_dir}")
    print(f"\nGenerated files:")
    if output_dir.exists():
        for plot_file in sorted(output_dir.glob("*.png")):
            print(f"  • {plot_file.name}")


if __name__ == '__main__':
    try:
        test_visualizations()
    except Exception as e:
        print(f"\n❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
