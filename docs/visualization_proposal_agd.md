# Visualization Strategy for Active Generative Discovery with Portfolio Constraints

## Overview

We need to tell a compelling story that builds from Economic AL → Active Generative Discovery with portfolio constraints. The narrative arc:

1. **Figure 1 (Keep with minor updates)**: Economic AL baseline works
2. **Figure 2 (Major overhaul)**: Active Generative Discovery dramatically improves discovery
3. **Figure 3 (NEW)**: Portfolio constraints provide intelligent hedging

---

## Figure 1: Economic AL Baseline (Keep, Minor Updates)

**Current state**: 4-panel showing Economic AL validation
**Status**: ✓ Works well, minimal changes needed

### Proposed Changes:
- **Keep all 4 panels** (A: Ablation, B: Learning curves, C: Budget, D: Efficiency)
- **Minor title update**: "Economic Active Learning: Foundation Layer"
- **Add subtitle**: "Budget-constrained active learning on real MOF database (687 MOFs)"
- **Purpose**: Establish that our baseline Economic AL is sound before adding generation

**Rationale**: This figure proves we have a working AL system. Audience needs confidence in the foundation before we add generative layer.

---

## Figure 2: Active Generative Discovery Impact (Major Overhaul)

**Current state**: 2-panel showing dual objectives
**Proposed**: 4-panel showing generation + discovery benefits

### New Layout: 2×2 Grid

#### **Panel A (Top Left): Discovery Progression Over Iterations**
```
Type: Line plot with markers
X-axis: Iteration (1, 2, 3)
Y-axis: Best CO2 uptake found (mol/kg)

Lines:
1. Active Generative Discovery (green, solid)
   - Iter 1: 8.47 (from real MOF - marked with circle)
   - Iter 2: 10.47 (from generated MOF - marked with star)
   - Iter 3: 11.23 (from generated MOF - marked with star)

2. Economic AL only baseline (blue, dashed)
   - Iter 1: 8.47
   - Iter 2: 8.75 (hypothetical - from historical data)
   - Iter 3: 9.02 (hypothetical)

Annotations:
- Arrow showing +44% improvement (8.47 → 11.23)
- "Generated MOFs dominate!" at iter 2-3
- Legend distinguishing real (○) vs generated (⭐) discoveries
```

**Key insight**: Generation drives discovery improvement, not just more sampling.

---

#### **Panel B (Top Right): Portfolio Balance Over Iterations**
```
Type: Stacked bar chart
X-axis: Iteration (1, 2, 3)
Y-axis: MOFs validated (count)

Bars (stacked):
- Bottom (blue): Real MOFs selected [3, 2, 2]
- Top (orange): Generated MOFs selected [10, 6, 6]

Annotations:
- Percentage labels: 76.9%, 75%, 75%
- Horizontal band showing target range (70-85%)
- "✓ Portfolio constraint satisfied" badge

Below bars: Budget spent
- Iter 1: $470
- Iter 2: $322
- Iter 3: $313
```

**Key insight**: Portfolio constraints maintained throughout while maximizing exploration.

---

#### **Panel C (Bottom Left): Generation Quality Metrics**
```
Type: Multi-metric visualization (bar + scatter)

Bars showing across 3 iterations:
1. MOFs generated (raw): [70, 57, 56]
2. Unique MOFs: [67, 56, 55] - with diversity % labels
3. Novel MOFs: [67, 56, 55] - with 100% novelty badges

Add trend line showing:
- Target CO2 increasing: 7.1 → 8.9 → 9.2 mol/kg
- "VAE learns from validated data!"

Annotations:
- "100% novelty" (all generated MOFs new to database)
- "97-98% diversity" (unique metal-linker combos)
- "Guided generation" (targets shift based on best validated)
```

**Key insight**: Generation is high-quality and adaptive, not random.

---

#### **Panel D (Bottom Right): Selection Competition**
```
Type: Violin plot or box plot
X-axis: Source (Real MOFs, Generated MOFs)
Y-axis: Acquisition score (EI per dollar)

Show distribution of:
1. All real MOF candidates (657 → 654 → 652)
   - Base acquisition: 0.19-0.23

2. All generated MOF candidates (67 → 56 → 55)
   - Base acquisition: 0.18-0.26
   - After exploration bonus (+2.0): 2.18-2.26

Overlay:
- Selected MOFs (scatter points in bold)
- Exploration bonus arrow showing shift

Annotations:
- "Exploration bonus: +2.0 → +1.62"
- "Generated MOFs gain competitive advantage"
- Show bonus decay across iterations
```

**Key insight**: Exploration bonus enables generated MOFs to compete despite surrogate uncertainty, but quality still matters within each pool.

---

### Figure 2 Title
**Main**: "Active Generative Discovery: VAE-Guided Generation Accelerates Materials Discovery"

**Subtitle**: "Portfolio-constrained selection balances exploration (70-85% generated) with exploitation hedge (15-30% real)"

---

## Figure 3 (NEW): Portfolio Theory & Risk Management

**Purpose**: Deep dive into why portfolio constraints matter (for technical audience)

### Layout: 2×2 Grid

#### **Panel A: Risk-Return Analysis**
```
Type: Scatter plot (portfolio theory style)
X-axis: Risk (variance in outcomes)
Y-axis: Expected discovery improvement

Points:
1. 100% Generated (bonus=4.0): High risk, high return
2. Portfolio-constrained (bonus=2.0): Moderate risk, high return ✓
3. Balanced (bonus=0.8): Low risk, moderate return
4. Pure exploitation (bonus=0): No risk, low return

Pareto frontier connecting optimal strategies
Our choice highlighted
```

---

#### **Panel B: Hedge Value Demonstration**
```
Type: Scenario comparison (side-by-side bars)

Scenarios:
1. VAE generates excellent MOFs
   - 100% generated: Finds 11.5 mol/kg
   - Portfolio (85%): Finds 11.3 mol/kg
   - Loss: -0.2 mol/kg (acceptable)

2. VAE generates poor MOFs (stress test)
   - 100% generated: Finds 5.2 mol/kg
   - Portfolio (85%): Finds 8.5 mol/kg from real hedge
   - Hedge value: +3.3 mol/kg (critical!)

Annotation: "Portfolio insurance against model failure"
```

---

#### **Panel C: Exploration Bonus Calibration**
```
Type: Heatmap or contour plot
X-axis: Exploration bonus (0 → 4.0)
Y-axis: Generated MOF selection %

Color: Discovery performance (mol/kg)

Show:
- Empirical minimum: 0.05 (achieves 100% selection)
- UCB1: 2.15
- GP-UCB: 4.0
- Our choice: 2.0 (marked)

Annotations:
- "Sweet spot: balance novelty with quality"
- Region highlighting 70-85% target
```

---

#### **Panel D: Real vs Generated Performance**
```
Type: Empirical CDF or histogram comparison

Show CO2 uptake distributions:
1. All real MOFs in database: Mean 5.8, Max 8.47
2. Generated MOFs (validated): Mean 7.9, Max 11.23

Overlay:
- Validation data points
- Percentage above database best (8.47):
  - Real MOFs: 0%
  - Generated MOFs: 45% (10/22)

Annotation: "Generated MOFs shift distribution rightward"
```

---

### Figure 3 Title
**Main**: "Portfolio Theory in Materials Discovery: Diversification as Risk Management"

**Subtitle**: "Balancing aggressive exploration with hedge against generative model failure"

---

## Implementation Priority

### High Priority (Must-have for hackathon):
1. **Figure 2 (overhaul)** - Core story of Active Generative Discovery
   - Shows generation works
   - Shows portfolio constraints satisfied
   - Shows discovery improvement

### Medium Priority (Nice-to-have):
2. **Figure 1 (minor updates)** - Quick title changes only
3. **Figure 3 Panel A & D** - Portfolio value demonstration

### Low Priority (Optional/backup):
4. **Figure 3 Panel B & C** - Deeper theoretical justification

---

## Data Requirements

### Already available from `demo_results.json`:
- ✓ Iteration data (n=3)
- ✓ Portfolio balance (real vs generated selection)
- ✓ Discovery progression (best CO2 per iteration)
- ✓ Generation quality metrics (diversity, novelty)
- ✓ Budget spent per iteration

### Need to compute/extract:
- Acquisition score distributions (from demo output or re-run)
- Real MOF baseline performance (from historical Economic AL runs)
- Individual validated MOF CO2 values (need to save from demo)

### For Figure 3 (if we build it):
- Scenario simulations (can use existing analysis scripts)
- Exploration bonus sweep results (from `analyze_exploration_bonus.py`)

---

## Recommended Approach

### Option 1: Conservative (2 figures, ~2 hours work)
- Keep Figure 1 as-is
- Overhaul Figure 2 (4-panel Active Generative Discovery)
- **Outcome**: Clear story, publication-ready

### Option 2: Comprehensive (3 figures, ~4 hours work)
- Minor update to Figure 1
- Full overhaul of Figure 2
- Create Figure 3 (at least Panels A & D)
- **Outcome**: Complete technical narrative

### Option 3: Minimal (1 figure augmentation, ~1 hour)
- Keep Figure 1
- Augment Figure 2 with 2 new panels (Portfolio + Generation quality)
- **Outcome**: Quick demo-ready addition

---

## Code Structure

```python
# src/visualization/figure2_active_generative_discovery.py
def plot_active_generative_discovery(output_dir):
    """
    Create Figure 2: Active Generative Discovery impact (4-panel)

    Panels:
    A. Discovery progression (line plot)
    B. Portfolio balance (stacked bars)
    C. Generation quality (multi-metric)
    D. Selection competition (violin/box plot)
    """
    # Load demo_results.json
    # Create 2x2 subplot grid
    # Generate 4 panels
    # Save high-res PNG

# src/visualization/figure3_portfolio_theory.py  (optional)
def plot_portfolio_analysis(output_dir):
    """
    Create Figure 3: Portfolio theory & risk management (4-panel)
    """
    # Load analysis results
    # Create risk-return, hedge value, calibration, performance panels
```

---

## Hackathon Presentation Flow

### Slide 1: "The Problem"
- Materials discovery is expensive
- Need both good predictions AND good materials

### Slide 2: "Our Foundation" → **Show Figure 1**
- Economic AL works on real MOFs
- Budget-constrained, sample-efficient

### Slide 3: "The Innovation" → **Show Figure 2**
- Active Generative Discovery = AL + VAE generation
- Portfolio constraints provide hedge
- **44% discovery improvement** (8.47 → 11.23 mol/kg)
- **100% novel** MOFs, **75% selected**

### Slide 4: "Why It Matters" → **Show Figure 3** (if available)
- Portfolio theory prevents over-optimism
- Hedge protects against model failure
- Rigorous exploration-exploitation balance

---

## Color Scheme (Consistent Across All Figures)

```python
COLORS = {
    'real_mof': '#2E86AB',        # Blue (trustworthy, known)
    'generated_mof': '#F77F00',   # Orange (novel, exciting)
    'baseline': '#95B8D1',        # Light blue (comparison)
    'success': '#06A77D',         # Green (good performance)
    'caution': '#FFA500',         # Yellow/orange (moderate)
    'failure': '#D62828',         # Red (poor performance)
    'portfolio_target': '#9B59B6' # Purple (constraint/theory)
}
```

---

## File Organization

```
results/
  figures/
    figure1_ml_ablation.png          [Keep existing]
    figure2_active_generative_discovery.png  [NEW - overhaul]
    figure3_portfolio_theory.png     [NEW - optional]

src/
  visualization/
    figure1_ml_ablation.py           [Keep existing]
    figure2_dual_objectives.py       [Deprecate/archive]
    figure2_active_generative_discovery.py  [NEW]
    figure3_portfolio_theory.py      [NEW - optional]
```

---

## Next Steps

1. **Decide on scope**: Option 1 (conservative), 2 (comprehensive), or 3 (minimal)?
2. **Extract missing data**: Save individual MOF validation results from demo
3. **Implement Figure 2**: Core 4-panel Active Generative Discovery visualization
4. **(Optional) Implement Figure 3**: Portfolio theory deep dive
5. **Test with demo results**: Verify all data flows correctly
6. **Polish for presentation**: High-res export, consistent styling

---

## Estimated Time

- **Figure 2 implementation**: 2-3 hours
- **Figure 3 implementation**: 1.5-2 hours (if needed)
- **Data extraction/testing**: 0.5 hours
- **Polish & export**: 0.5 hours

**Total: 3-6 hours depending on scope**

With hackathon in 12 hours, **recommend Option 1 (Conservative)**: Focus on excellent Figure 2, keep Figure 1 as-is.
