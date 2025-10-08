"""
Test script for MOF Cost Estimator
"""

from src.cost.estimator import MOFCostEstimator


def test_known_mofs():
    """Test cost estimation on well-known MOFs"""
    estimator = MOFCostEstimator()

    test_mofs = [
        {
            'name': 'MOF-5',
            'metal': 'Zn',
            'linker': 'terephthalic acid',
            'expected_range': (0.5, 1.5),  # Expected cost range
        },
        {
            'name': 'HKUST-1',
            'metal': 'Cu',
            'linker': 'trimesic acid',
            'expected_range': (0.8, 2.0),
        },
        {
            'name': 'UiO-66',
            'metal': 'Zr',
            'linker': 'terephthalic acid',
            'expected_range': (1.0, 3.0),  # Zr is expensive
        },
    ]

    print("\nTesting MOF Cost Estimator")
    print("=" * 60)

    for mof in test_mofs:
        print(f"\n{mof['name']}:")
        print("-" * 60)

        # Base cost (100% yield)
        cost = estimator.estimate_synthesis_cost(mof)
        print(f"  Base cost (100% yield): ${cost['total_cost_per_gram']:.2f}/g")
        print(f"    - Metal ({mof['metal']}): ${cost['metal_contribution']:.2f}")
        print(f"    - Linker: ${cost['linker_contribution']:.2f}")
        print(f"    - Solvent: ${cost['solvent_contribution']:.2f}")

        # Adjusted for yield
        cost_adj = estimator.estimate_batch_cost(mof, yield_percent=70)
        print(f"  Actual cost (70% yield): ${cost_adj['actual_cost_per_gram']:.2f}/g")

        # Cost-efficiency (assuming 10 mmol/g CO2 uptake)
        assumed_performance = 10  # mmol/g
        efficiency = assumed_performance / cost_adj['actual_cost_per_gram']
        print(f"  Cost-efficiency: {efficiency:.2f} mmol CO2 per $")

        # Validation
        expected_min, expected_max = mof['expected_range']
        if expected_min <= cost_adj['actual_cost_per_gram'] <= expected_max:
            print(f"  ✅ Within expected range ${expected_min}-${expected_max}/g")
        else:
            print(f"  ⚠️  Outside expected range ${expected_min}-${expected_max}/g")

    print("\n" + "=" * 60)
    print("✅ Cost estimator tests complete!")
    print("\nKey insights:")
    print("  - MOF-5 (Zn-BDC): Cheapest, common reagents")
    print("  - HKUST-1 (Cu-BTC): Moderate cost, trimesic acid more expensive")
    print("  - UiO-66 (Zr-BDC): Most expensive due to Zr")
    print("\n📊 These cost estimates will be used in Economic Active Learning!")


if __name__ == '__main__':
    test_known_mofs()
