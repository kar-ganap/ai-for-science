# Processed CRAFTED MOF Data

**Generated:** October 8, 2025
**Source:** CRAFTED 2.0.0 database
**Processing:** SimpleCRAFTEDLoader

---

## File: `crafted_mofs_co2.csv`

**MOFs:** 687 experimental MOFs from CoRE MOF 2014
**Conditions:** 1 bar CO2, 298K

### Columns

| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| `mof_id` | CSD refcode identifier | - | e.g., PUPNAQ, HIFVOI |
| `co2_uptake_mean` | Mean CO2 uptake | mol/kg | Average across 12 methods |
| `co2_uptake_std` | Uncertainty in uptake | mol/kg | Std dev across 12 methods |
| `n_methods` | Number of methods | - | 2 FF × 6 charge schemes = 12 |
| `metal` | Primary metal type | - | Zn, Fe, Ca, etc. or Unknown |
| `cell_a` | Unit cell length a | Å | From CIF |
| `cell_b` | Unit cell length b | Å | From CIF |
| `cell_c` | Unit cell length c | Å | From CIF |
| `volume` | Unit cell volume | Å³ | From CIF |

### Statistics (@ 1 bar, 298K)

**CO2 Uptake:**
- Mean: 3.15 ± 1.86 mol/kg
- Range: 0.00 - 11.23 mol/kg
- Median: 2.89 mol/kg

**Uncertainty (Epistemic):**
- Mean: 0.793 mol/kg
- Max: 11.080 mol/kg
- This reflects force field and charge scheme sensitivity

**Metal Distribution:**
- Zn: 523 MOFs (76%)
- Fe: 67 MOFs (10%)
- Unknown: 42 MOFs (6%)
- Ca: 31 MOFs (5%)
- Al: 21 MOFs (3%)
- Others: 3 MOFs (<1%)

**Methods Coverage:**
- All MOFs have 12/12 methods (2 force fields × 6 charge schemes)
- This enables robust uncertainty quantification

---

## Methods (12 combinations)

**Force Fields (2):**
1. UFF (Universal Force Field)
2. DREIDING

**Charge Schemes (6):**
1. DDEC (Density Derived Electrostatic and Chemical)
2. EQeq (Extended Charge Equilibration)
3. MPNN (Message Passing Neural Network)
4. NEUTRAL (No charges)
5. PACMOF (Partial Atomic Charges for MOFs)
6. Qeq (Charge Equilibration)

**Total combinations:** 2 × 6 = 12 per MOF per temperature

---

## Uncertainty Interpretation

### What `co2_uptake_std` Represents

**Epistemic Uncertainty:**
- Uncertainty due to force field and charge scheme choice
- Reflects "model uncertainty" - different methods predict different values
- **Reducible** with better models (but not with more data in this case)

**Example:**
```python
# MOF with low uncertainty (FF-robust)
PUPNAQ: mean=1.48 mol/kg, std=0.58 mol/kg
→ All 12 methods agree reasonably well

# MOF with high uncertainty (FF-sensitive)
LIBJAJ: mean=4.01 mol/kg, std=1.61 mol/kg
→ Different methods predict very different values
→ Be cautious with predictions for this MOF
```

### Implications for Active Learning

**High uncertainty MOFs:**
- Predictions less reliable
- May want to validate experimentally
- Or exclude from high-confidence recommendations

**Low uncertainty MOFs:**
- Predictions more reliable
- Multiple methods agree
- Higher confidence for deployment

---

## Usage Example

```python
import pandas as pd

# Load data
df = pd.read_csv('data/processed/crafted_mofs_co2.csv')

# Filter high-performers with low uncertainty
candidates = df[
    (df['co2_uptake_mean'] > 5.0) &  # High CO2 uptake
    (df['co2_uptake_std'] < 1.0) &   # Low FF uncertainty
    (df['metal'].isin(['Zn', 'Fe']))  # Common metals
]

print(f"Found {len(candidates)} promising MOF candidates")
```

---

## For Economic Active Learning

### Ready-to-use Features

**Input features (X):**
- `cell_a`, `cell_b`, `cell_c`: Unit cell dimensions
- `volume`: Unit cell volume
- `metal` (one-hot encoded): Metal type

**Target variable (y):**
- `co2_uptake_mean`: CO2 uptake at 1 bar, 298K

**Uncertainty for AL:**
- `co2_uptake_std`: Use as baseline uncertainty
- Or compute ensemble uncertainty from ML model

**Cost estimation:**
- `metal` → link to MOFCostEstimator
- E.g., Zn-MOFs typically $0.50-1.50/g

### Integration with Cost Estimator

```python
from src.cost.estimator import MOFCostEstimator

estimator = MOFCostEstimator()

# Map metal to cost
df['synthesis_cost'] = df['metal'].map({
    'Zn': estimator.estimate_synthesis_cost({'metal': 'Zn', 'linker': 'terephthalic acid'})['total_cost_per_gram'],
    'Cu': estimator.estimate_synthesis_cost({'metal': 'Cu', 'linker': 'trimesic acid'})['total_cost_per_gram'],
    'Fe': estimator.estimate_synthesis_cost({'metal': 'Fe', 'linker': 'terephthalic acid'})['total_cost_per_gram'],
    # Add more as needed
})

# Now ready for Economic AL!
```

---

## Data Quality Notes

### Strengths ✅

1. **Experimental MOFs** - All 687 are from CoRE MOF 2014 (experimentally synthesized)
2. **Uncertainty quantified** - 12 methods per MOF enables robust error bars
3. **Reproducible** - GCMC simulations with published force fields
4. **High coverage** - All MOFs have complete 12-method data

### Limitations ⚠️

1. **Geometric features limited** - Only unit cell parameters extracted
   - For better features: use CRAFTED_MOF_geometric.csv or parse CIFs with proper tools

2. **Metal extraction imperfect** - 6% "Unknown" due to:
   - Complex CIF structures
   - Mixed-metal MOFs (assigned to first found)
   - Should manually verify for critical applications

3. **Single pressure point** - Extracted uptake at 1 bar only
   - Full isotherms available in CRAFTED if needed
   - Can extract at other pressures (0.15 bar for DAC, 15 bar for PSA, etc.)

4. **Single temperature** - 298K only in this dataset
   - CRAFTED has 273K and 323K available

5. **Uncertainty is FF-based** - Not experimental uncertainty
   - Reflects computational model disagreement
   - Real experimental error is different (typically ±5-10%)

---

## Next Steps

**To improve dataset:**

1. **Add full geometric features:**
   ```python
   geom = pd.read_csv('CRAFTED-2.0.0/RAC_DBSCAN/CRAFTED_MOF_geometric.csv')
   df_enriched = df.merge(geom, left_on='mof_id', right_on='FrameworkName')
   ```
   This adds: LCD, PLD, ASA, void fraction, pore volume, etc.

2. **Manual metal verification:**
   - For high-value MOFs, manually verify metal composition
   - Check CIF files or CoRE MOF 2014 database

3. **Add linker information:**
   - Parse CIF for organic linker
   - Or match against CoRE MOF database

4. **Multi-pressure extraction:**
   - Extract CO2 uptake at 0.15 bar (DAC conditions)
   - Extract at 15 bar (high-pressure applications)

---

## Citation

If using this data, cite:

**CRAFTED Database:**
```bibtex
@article{crafted2023,
  title={CRAFTED: An exploratory database of simulated adsorption
         isotherms of metal-organic frameworks},
  author={Krishnapriyan, Aditi and ... and others},
  journal={Scientific Data},
  volume={10},
  pages={230},
  year={2023}
}
```

**CoRE MOF 2014:**
```bibtex
@article{coremof2014,
  title={Computation-Ready, Experimental Metal-Organic Frameworks},
  author={Chung, Yongchul G. and ... and Snurr, Randall Q.},
  journal={Chemistry of Materials},
  volume={26},
  pages={6185--6192},
  year={2014}
}
```

---

**Status:** Ready for Economic Active Learning integration! ✅
