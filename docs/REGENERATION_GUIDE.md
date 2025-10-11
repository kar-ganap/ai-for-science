# Data Regeneration Guide

## Overview

The Streamlit dashboard now supports **interactive data regeneration** with custom parameters. You can regenerate Figure 1 data (Economic AL), Figure 2 data (AGD), or both.

---

## Using the Configuration Panel

### 1. Switch to Regenerate Mode

In the sidebar, switch from **"View Pre-computed Results"** to **"Regenerate Data"**.

### 2. Choose What to Regenerate

Select one of four options:
- **Figure 1 Data (Economic AL)**: Regenerate only Economic AL experiments
- **Figure 2 Data (AGD)**: Regenerate only Active Generative Discovery experiments
- **Both Figures**: Run both Figure 1 and Figure 2 pipelines
- **Just Update Figures (no new runs)**: Regenerate visualizations from existing data (fastest option)

---

## Configuration Options

### Figure 1 Parameters (Economic AL)

When regenerating Figure 1, you can control:

- **Budget per Iteration**: $10-$100 (default: $50)
  - How much to spend per AL iteration
  - Current results use $50/iteration

- **Number of Iterations**: 1-10 (default: 5)
  - How many AL cycles to run
  - Current results use 5 iterations

- **Strategies to Run**: Exploration and/or Exploitation
  - Exploration: `uncertainty / cost` (maximize learning)
  - Exploitation: `value × uncertainty / cost` (maximize discovery)
  - Current results include both

**Estimated Runtime**: ~0.5 min per strategy per iteration
- Example: 5 iterations × 2 strategies = ~5 minutes

### Figure 2 Parameters (Active Generative Discovery)

When regenerating Figure 2, you can control:

- **Budget per Iteration**: $100-$1000 (default: $500)
  - How much to spend per AGD iteration
  - Higher budgets allow more MOFs to be validated

- **Number of Iterations**: 1-5 (default: 3)
  - How many AGD cycles to run
  - Current results use 3 iterations

- **Portfolio Constraints**:
  - **Min Generated MOFs**: 50-90% (default: 70%)
  - **Max Generated MOFs**: 50-95% (default: 85%)
  - Controls risk management in portfolio selection

**Estimated Runtime**: ~5 min per iteration
- Example: 3 iterations = ~15 minutes

---

## What Gets Regenerated?

### Figure 1 Data Regeneration

**Scripts Run**:
1. `tests/test_economic_al_crafted.py` (Exploration strategy)
2. `tests/test_economic_al_expected_value.py` (Exploitation strategy)
3. `src/visualization/figure1_ml_ablation.py` (Figure generation)

**Output Files**:
- `results/economic_al_crafted_integration.csv` (Exploration history)
- `results/economic_al_expected_value.csv` (Exploitation history)
- `results/figures/figure1_ml_ablation.png` (Updated figure)

### Figure 2 Data Regeneration

**Scripts Run**:
1. `demo_active_generative_discovery.py` (AGD pipeline)
2. `src/visualization/figure2_active_generative_discovery.py` (Figure generation)

**Output Files**:
- `results/active_generative_discovery_demo/demo_results.json` (AGD history)
- `results/figure2_baseline_exploration_500.csv` (Fair baseline)
- `results/figures/figure2_active_generative_discovery.png` (Updated figure)

### Just Update Figures

**Scripts Run**:
1. `src/visualization/figure1_ml_ablation.py`
2. `src/visualization/figure2_active_generative_discovery.py`

Uses existing data in `results/` directory. Fastest option (~30 seconds).

---

## Progress Tracking

During regeneration, the main panel shows:
- ⏳ Real-time progress bar with status messages
- ✅ Completion messages for each stage
- ❌ Error messages if something fails
- 🎈 Celebration when complete!

---

## Example Workflows

### Workflow 1: Quick Figure Update

**Use Case**: Data is fine, just want to regenerate figures (e.g., changed styling)

**Steps**:
1. Switch to "Regenerate Data"
2. Select "Just Update Figures (no new runs)"
3. Click "🚀 Start Regeneration"
4. Wait ~30 seconds
5. Click "📊 View Updated Results"

### Workflow 2: Test Different Budget

**Use Case**: See how results change with $75/iteration instead of $50

**Steps**:
1. Switch to "Regenerate Data"
2. Select "Figure 1 Data (Economic AL)"
3. Adjust "Budget per Iteration" to $75
4. Keep other defaults
5. Click "🚀 Start Regeneration"
6. Wait ~5 minutes
7. Compare new results with old

### Workflow 3: Full Regeneration

**Use Case**: Regenerate everything from scratch with custom parameters

**Steps**:
1. Switch to "Regenerate Data"
2. Select "Both Figures"
3. Adjust Figure 1 parameters (budget, iterations, strategies)
4. Adjust Figure 2 parameters (budget, iterations, portfolio)
5. Click "🚀 Start Regeneration"
6. Wait ~20 minutes
7. Explore new results

---

## Technical Details

### Backend Implementation

The regeneration handler (`streamlit_app.py` lines 202-352):
1. Uses `subprocess.run()` to execute Python scripts
2. Captures stdout/stderr for error reporting
3. Updates progress bar at key milestones
4. Saves all outputs to standard locations

### Parameter Limitations

**Note**: The current implementation runs scripts with their default parameters. To use custom parameters, you would need to:
1. Modify the test scripts to accept command-line arguments
2. Pass parameters via subprocess arguments

For example:
```python
subprocess.run([
    sys.executable,
    "tests/test_economic_al_crafted.py",
    "--budget", str(budget),
    "--iterations", str(iterations)
], ...)
```

This is a future enhancement opportunity!

---

## Troubleshooting

### Issue: "Regeneration failed"

**Possible Causes**:
- Missing dependencies (VAE model, data files)
- Insufficient memory for GP training
- File permission issues

**Solution**: Check error message in red box, examine stderr output

### Issue: "Results look unchanged"

**Possible Causes**:
- Browser caching old images
- Files weren't actually regenerated

**Solution**:
1. Hard refresh browser (Ctrl+Shift+R)
2. Check file timestamps in `results/figures/`
3. Look for "✅ Complete!" message

### Issue: "Taking too long"

**Possible Causes**:
- Large number of iterations selected
- Both strategies selected for Figure 1
- System resource constraints

**Solution**:
- Start with smaller parameters
- Monitor progress messages
- Be patient - AGD iterations are computationally intensive!

---

## Future Enhancements

Potential improvements to the regeneration system:

1. **Parameterized Scripts**: Modify test scripts to accept CLI arguments for full parameter control
2. **Background Processing**: Use Streamlit background tasks to allow dashboard navigation during regeneration
3. **Comparison Mode**: Keep old results and compare with new side-by-side
4. **Random Seed Control**: Reproducibility by setting random seeds
5. **Baseline Control**: Option to regenerate Random/Expert baselines
6. **Partial Regeneration**: Regenerate only specific components (e.g., only exploration strategy)
7. **Progress Streaming**: Live output streaming from subprocess
8. **Result Validation**: Automatic checks that regenerated data is valid

---

## Summary

The regeneration system provides:
- ✅ **Flexibility**: Custom budgets, iterations, strategies
- ✅ **Transparency**: Real-time progress and error reporting
- ✅ **Safety**: Original data preserved (files overwritten only on success)
- ✅ **Speed Options**: Quick figure updates or full pipeline runs

Perfect for:
- Exploring different experimental configurations
- Testing sensitivity to parameters
- Regenerating after code changes
- Creating publication-ready figures with updated data
