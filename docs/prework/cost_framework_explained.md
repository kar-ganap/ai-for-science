# Dual-Cost Framework for Economic Active Learning

**Purpose:** Clarify two distinct costs in MOF discovery pipeline

**Date:** October 8, 2025

---

## Two Types of Costs

### 1. Validation Cost (Discovery Phase)
**What:** Cost to confirm/measure a MOF's performance
**When:** During active learning - deciding which MOFs to test
**Budget constraint:** Lab validation budget per iteration

### 2. Synthesis Cost (Production Phase)
**What:** Cost to produce MOF at scale for deployment
**When:** After validation - commercialization decision
**Selection criterion:** Economic viability for industrial use

---

## Cost Estimates (Realistic)

### Validation Cost Options

#### A. GCMC Simulation (Computational)
**Scenario:** Run molecular dynamics to predict CO2 uptake

**Cost breakdown:**
```
Cluster compute: 8 hours × 16 cores × $1/core-hour = $128/MOF
   - Structure optimization: 2 hours
   - GCMC equilibration: 3 hours
   - GCMC production: 3 hours

Academic cluster: $0 monetary, but ~$50 opportunity cost
Cloud (AWS/Azure): $50-150/MOF depending on detail level

Realistic range: $20-100/MOF
Demo value: $50/MOF (middle estimate)
```

**Complexity factors:**
- Large unit cell → longer compute
- Multiple conditions (T, P) → multiply cost
- Higher accuracy → more cycles

#### B. Experimental Validation (Lab)
**Scenario:** Synthesize MOF and measure CO2 uptake

**Cost breakdown:**
```
REAGENTS & MATERIALS
  Metal precursor:        $10-50
  Organic linker:         $20-150
  Solvents (DMF, etc):    $20-50
  Activation materials:   $10-30
                          ─────────
Subtotal:                 $60-280

CHARACTERIZATION
  Powder XRD:            $50-100 (structure confirmation)
  N2/Ar physisorption:   $80-150 (BET surface area)
  TGA:                   $40-80 (thermal stability)
  SEM/TEM (optional):    $100-200 (morphology)
                         ─────────
Subtotal:                $270-530

GAS ADSORPTION TESTING
  CO2 isotherm:          $150-300 (instrument time)
  Multi-temperature:     $300-600 (if needed)
  Selectivity (CO2/N2):  $200-400 (mixed gas)
                         ─────────
Subtotal:                $150-600

LABOR
  Grad student/postdoc:  $500-1500 (40-60 hours)
  PI oversight:          $100-300
                         ─────────
Subtotal:                $600-1800

═══════════════════════════════════
TOTAL:                   $1080-3210/MOF

Realistic range: $1000-3000/MOF
Demo value: $1500/MOF (typical)
```

**Success rate factors:**
- First attempt success: ~60-70%
- Failed synthesis → need retry → 1.5× cost
- Realistic average: $1500-2000/MOF

#### C. Literature Lookup
**Cost:** ~$0 (but availability <5% for novel MOFs)

---

### Synthesis Cost (Production Scale)

**Scenario:** Produce MOF at kg scale for deployment

**Our existing estimator:**
```python
from src.cost.estimator import MOFCostEstimator

# Examples (per gram, 70% yield):
MOF-5 (Zn-BDC):    $0.50-1.50/g
HKUST-1 (Cu-BTC):  $0.80-2.00/g
UiO-66 (Zr-BDC):   $1.00-3.00/g
```

**Scale factors:**
- Lab scale (1-10g):     1.0× base cost
- Bench scale (100g):    0.7× (bulk reagents)
- Pilot scale (1kg):     0.5× (efficiency gains)
- Industrial (>10kg):    0.3× (full optimization)

**Realistic range for production: $0.30-5.00/g**

---

## Implementation for Economic AL

### Framework Design

```python
class EconomicActiveLearner:
    def __init__(self, ..., validation_cost_model='simulation'):
        """
        Args:
            validation_cost_model: 'simulation', 'experimental', or 'hybrid'
        """
        self.validation_cost_model = validation_cost_model

    def estimate_validation_cost(self, mof_composition):
        """Cost to validate (test/confirm) a MOF"""

        if self.validation_cost_model == 'simulation':
            # GCMC simulation cost
            base_cost = 50  # $50 baseline

            # Complexity factors
            complexity = self._estimate_complexity(mof_composition)
            # 1.0 = simple (Zn-BDC)
            # 2.0 = complex (mixed-metal, large unit cell)

            return base_cost * complexity

        elif self.validation_cost_model == 'experimental':
            # Lab synthesis + testing cost
            base_cost = 1500  # $1500 baseline

            # Synthesis difficulty
            difficulty = self._estimate_synthesis_difficulty(mof_composition)
            # 0.7 = easy (MOF-5, HKUST-1)
            # 1.0 = moderate (most MOFs)
            # 1.5 = difficult (mixed-metal, air-sensitive)

            return base_cost * difficulty

        elif self.validation_cost_model == 'hybrid':
            # Simulate first, then experiment if promising
            sim_cost = self.estimate_validation_cost_simulation(mof_composition)

            # Probability of experimental follow-up (if promising)
            p_experiment = 0.2  # Top 20% go to experiment
            exp_cost = 1500

            expected_cost = sim_cost + (p_experiment * exp_cost)
            return expected_cost

    def estimate_synthesis_cost(self, mof_composition):
        """Cost to produce at scale (from existing estimator)"""
        return self.cost_estimator.estimate_synthesis_cost(mof_composition)

    def economic_selection(self, budget_per_iteration, strategy='cost_aware'):
        """
        Select MOFs within validation budget

        Returns:
            selected_mofs: List of MOFs to validate
            validation_cost: Total validation cost
            synthesis_costs: Individual synthesis costs (for reporting)
        """
        # Predict with uncertainty
        pool_mean, pool_std = self.predict_with_uncertainty(self.X_pool)

        # Estimate VALIDATION cost (what we spend in AL)
        validation_costs = [
            self.estimate_validation_cost(comp)
            for comp in self.pool_compositions
        ]

        # Estimate SYNTHESIS cost (for filtering/reporting)
        synthesis_costs = [
            self.estimate_synthesis_cost(comp)
            for comp in self.pool_compositions
        ]

        if strategy == 'cost_aware_uncertainty':
            # Maximize: information gain per validation dollar
            acquisition = pool_std / (validation_costs + 1e-6)

        elif strategy == 'economic_value':
            # Consider both validation cost AND synthesis cost
            # High performance, cheap to validate, cheap to make
            acquisition = (pool_mean * pool_std) / (
                (validation_costs + synthesis_costs) + 1e-6
            )

        # Greedy selection within validation budget
        selected = []
        total_validation_cost = 0

        for idx in np.argsort(acquisition)[::-1]:
            cost = validation_costs[idx]
            if total_validation_cost + cost <= budget_per_iteration:
                selected.append(idx)
                total_validation_cost += cost

        return selected, total_validation_cost, synthesis_costs
```

---

## Recommended Configuration for Demo

### For CRAFTED Dataset (Simulation Data)

**Use "simulation" validation cost model:**

```python
# Validation cost = GCMC simulation cost
base_validation_cost = $50/MOF

# Complexity factor based on unit cell
complexity = volume / 2000  # Normalized
# Small MOF (1000 Å³): 0.5× = $25
# Medium (2000 Å³):    1.0× = $50
# Large (4000 Å³):     2.0× = $100

validation_cost = base_validation_cost * complexity
```

**Plus synthesis cost from existing estimator:**
```python
synthesis_cost = MOFCostEstimator.estimate_synthesis_cost(...)
# Range: $0.50-2.00/g
```

**Budget constraint:**
```python
validation_budget_per_iteration = $500
# Can validate ~10 MOFs per iteration (avg $50 each)
```

---

## Tracked Metrics (Dual-Cost)

### Per Iteration:
```python
metrics = {
    # Validation costs (AL budget)
    'validation_cost_iteration': $523,
    'validation_cost_cumulative': $1580,
    'avg_validation_cost_per_mof': $52,

    # Synthesis costs (for selected MOFs)
    'avg_synthesis_cost_selected': $0.85/g,
    'min_synthesis_cost_selected': $0.50/g,
    'max_synthesis_cost_selected': $1.20/g,

    # Performance
    'best_predicted_performance': 8.2 mol/kg,
    'avg_predicted_performance': 4.5 mol/kg,

    # Efficiency metrics
    'cost_efficiency': 8.2 mol/kg per $ synthesis,
    'validation_efficiency': 0.16 mol/kg per $ validation,
}
```

### Visualization:
```
┌────────────────────────────────────────────┐
│ Dual-Cost Tracking                         │
├────────────────────────────────────────────┤
│                                            │
│ Validation Budget (AL iterations)         │
│  $1500 ┤                        ●         │
│  $1000 ┤              ●                   │
│   $500 ┤    ●                             │
│      0 ┴────────────────────────────      │
│        Iter1   Iter2   Iter3             │
│                                            │
│ Synthesis Cost of Selected MOFs           │
│  $2.0/g ┤ ○                               │
│  $1.5/g ┤ ○ ○                             │
│  $1.0/g ┤ ○ ○ ○ ●                         │
│  $0.5/g ┤ ○ ○ ○ ○ ●●●                     │
│      0  ┴────────────────────────────      │
│         Pool    Selected (cheap!)         │
└────────────────────────────────────────────┘
```

---

## Narrative for Hackathon

### Problem Statement:
> "Discovering new MOFs requires answering two questions:
> 1. **Which MOFs should we test?** (validation cost)
> 2. **Which MOFs can we afford to make?** (synthesis cost)
>
> Traditional AL ignores both costs. Economic AL optimizes both."

### Our Innovation:
> "Economic Active Learning makes validation decisions under budget constraints,
> while tracking synthesis cost for commercialization planning.
>
> We don't just find the best MOF - we find the best **affordable** MOF
> that we can **afford to discover**."

### Metrics to Highlight:
```
✓ Validated 150 MOFs within $1500 budget (10 MOFs/iter)
✓ Selected MOFs cost $0.60/g avg (vs $0.85/g pool avg)
✓ Found high performers (>6 mol/kg) with low synthesis cost
✓ 2× more efficient than random selection
```

---

## Alternative: Simplified Single-Cost Model

**If dual-cost is too complex for demo:**

Use **validation cost only**, but make it realistic:

```python
# Assume validation cost ∝ synthesis complexity
validation_cost = 20 + (30 × synthesis_cost)

# Zn-MOF ($0.80/g): $20 + $24 = $44 to validate
# Zr-MOF ($1.50/g): $20 + $45 = $65 to validate

# Captures: expensive MOFs are harder to validate too
```

**Budget:** $500/iteration → ~10 MOFs

**Simpler story:** "Cost-aware AL finds affordable MOFs to test"

---

## Recommendation

**For hackathon:** Use **dual-cost model** with:

1. **Validation cost:** $50/MOF baseline × complexity factor
   - Based on unit cell volume
   - Represents GCMC simulation cost
   - Budget: $500/iteration → ~10 MOFs

2. **Synthesis cost:** Existing estimator ($0.50-2.00/g)
   - Reagent-based calculation
   - For commercialization filtering
   - Report as "production cost"

**Why both:**
- More realistic
- Differentiates your project
- Addresses real-world constraints
- Richer visualizations (2D cost space)

**Implementation time:** ~30 minutes to update code

---

## Next Steps

1. Update `EconomicActiveLearner` with dual-cost framework
2. Add validation cost estimator based on complexity
3. Update integration test to track both costs
4. Create dual-cost visualizations

**Shall I implement this?**
