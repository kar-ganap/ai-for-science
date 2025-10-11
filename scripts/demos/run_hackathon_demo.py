#!/usr/bin/env python3
"""
Hackathon Demo Runner
=====================

Executes the entire Economic AL pipeline and generates all visualizations.
Use this script to:
1. Verify everything works before the hackathon presentation
2. Regenerate results if needed
3. Quick demo of the full pipeline

Usage:
    python run_hackathon_demo.py [--quick]

Options:
    --quick     Skip re-running AL iterations, just regenerate figures from existing results
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def print_step(number, total, description):
    """Print a step indicator"""
    print(f"\n[{number}/{total}] {description}")
    print("-" * 80)

def run_pipeline(quick_mode=False):
    """Run the complete hackathon demo pipeline"""

    start_time = time.time()

    print_header("ECONOMIC AL HACKATHON DEMO")
    print(f"Mode: {'QUICK (figures only)' if quick_mode else 'FULL (complete pipeline)'}")
    print(f"Project root: {project_root}")

    # Step 1: Verify data exists
    print_step(1, 4, "Verifying data files...")

    data_file = project_root / "data" / "processed" / "crafted_mofs_co2.csv"
    if not data_file.exists():
        print(f"❌ ERROR: Missing data file: {data_file}")
        print(f"   Run data preprocessing first!")
        return False

    print(f"✅ Data file found: {data_file}")

    # Step 2: Run AL pipeline (or skip in quick mode)
    if not quick_mode:
        print_step(2, 4, "Running Economic AL pipeline...")
        print("   This will take ~2-3 minutes...")

        # Import and run main integration test
        from tests.test_economic_al_crafted import test_economic_al_crafted_integration

        try:
            history = test_economic_al_crafted_integration()
            print("\n✅ AL pipeline complete!")
        except Exception as e:
            print(f"\n❌ ERROR in AL pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Run expected value strategy
        print("\n   Running AL (Exploitation strategy)...")
        from tests.test_economic_al_expected_value import test_economic_al_expected_value

        try:
            test_economic_al_expected_value()
            print("\n✅ Exploitation strategy complete!")
        except Exception as e:
            print(f"\n❌ ERROR in exploitation strategy: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print_step(2, 4, "Skipping AL pipeline (quick mode)...")

        # Verify results exist
        results_file = project_root / "results" / "economic_al_crafted_integration.csv"
        if not results_file.exists():
            print(f"❌ ERROR: Missing results file: {results_file}")
            print(f"   Run in full mode first to generate results!")
            return False

        print(f"✅ Using existing results: {results_file}")

    # Step 3: Generate all visualizations
    print_step(3, 4, "Generating visualizations...")

    # Generate hackathon figures
    print("\n   Creating Figure 1 (ML Ablation Study)...")
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.visualization.figure1_ml_ablation"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("   ✅ Figure 1 complete")
        else:
            print(f"   ❌ Figure 1 failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error generating Figure 1: {e}")
        return False

    print("\n   Creating Figure 2 (Dual Objectives)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.visualization.figure2_dual_objectives"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("   ✅ Figure 2 complete")
        else:
            print(f"   ❌ Figure 2 failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error generating Figure 2: {e}")
        return False

    # Generate supplementary plots
    print("\n   Creating supplementary visualizations...")
    from tests.test_visualizations import test_visualizations

    try:
        test_visualizations()
        print("\n   ✅ All supplementary plots complete")
    except Exception as e:
        print(f"\n   ❌ Error in supplementary plots: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Summary
    print_step(4, 4, "Summary")

    figures_dir = project_root / "results" / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        print(f"\n✅ Generated {len(figures)} figures:")
        for fig in sorted(figures):
            print(f"   • {fig.name}")

    results_dir = project_root / "results"
    if results_dir.exists():
        csvs = list(results_dir.glob("*.csv"))
        print(f"\n✅ Generated {len(csvs)} result files:")
        for csv in sorted(csvs)[:5]:  # Show first 5
            print(f"   • {csv.name}")
        if len(csvs) > 5:
            print(f"   ... and {len(csvs) - 5} more")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!".center(80))
    print("=" * 80)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\n📊 Figures: {figures_dir}")
    print(f"📊 Results: {results_dir}")
    print(f"📖 Narrative: {project_root / 'HACKATHON_NARRATIVE.md'}")

    return True


def main():
    """Main entry point"""

    quick_mode = "--quick" in sys.argv or "-q" in sys.argv

    try:
        success = run_pipeline(quick_mode=quick_mode)

        if success:
            print("\n🎉 Ready for hackathon!")
            sys.exit(0)
        else:
            print("\n❌ Demo failed - check errors above")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
