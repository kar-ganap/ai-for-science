# Dataset Comparison: MOFX-DB vs. CRAFTED

**Purpose:** Evaluate which MOF CO2 adsorption dataset to use for Economic Active Learning

**Date:** October 8, 2025

---

## Executive Summary

**Recommendation:** **CRAFTED** for prework/hackathon, **MOFX-DB** as stretch goal

**Rationale:** CRAFTED provides experimentally-validated, high-quality data with known uncertainty bounds. MOFX-DB is massive but mostly hypothetical MOFs with single force field assumptions.

---

## Side-by-Side Comparison

| Aspect | MOFX-DB | CRAFTED |
|--------|---------|---------|
| **Size** | 160,000+ MOFs | 690 MOFs |
| **Publication** | J. Chem. Eng. Data (2023) | Nature Sci Data (2023) |
| **Affiliation** | Northwestern/NIST | IBM Research |
| **MOF Sources** | CoRE MOF + hMOF + ToBaCCo | CoRE MOF 2014 only |
| **Experimental vs Hypothetical** | ~13K experimental, ~147K hypothetical | 690 experimental |
| **Force Fields** | UFF + TraPPE (single approach) | UFF + DREIDING (systematic comparison) |
| **Partial Charges** | Typically single scheme | 6 schemes (no charges, Qeq, EQeq, MPNN, PACMOF, DDEC) |
| **Temperatures** | Varies by database | 3 temps (273K, 298K, 323K) |
| **Gases** | H2, CH4, CO2, Xe, Kr, Ar, N2 | CO2, N2 |
| **Data Points** | 3+ million | 99,360 isotherms (49,680 CO2 + 49,680 N2) |
| **Download** | Web interface/API (complex) | Direct download, 54.7 MB (easy) |
| **Reproducibility** | JSON with metadata | CIF files + force field files + notebooks |

---

## Detailed Analysis

### 1. MOF Quality & Experimental Validity

#### MOFX-DB
**Composition:**
- ~13,511 ToBaCCo MOFs (hypothetical, geometrically assembled)
- ~137,953 hMOF structures (hypothetical, computationally generated)
- ~13,000 CoRE MOF structures (experimental)

**Experimental Validation:**
- Only ~8% are experimentally-reported structures
- hMOF and ToBaCCo are "geometrically assembled in silico"
- May not be synthesizable or stable
- No experimental isotherm comparison for hypothetical MOFs

**Quote from research:**
> "Hypothetical MOF datasets such as hMOF and ToBaCCo provide valuable inputs for training generative models, they often lack experimental validation, whereas databases like CoRE MOF are rooted in experimental data."

#### CRAFTED
**Composition:**
- 100% experimental MOFs from CoRE MOF 2014
- Filtered for computation-readiness
- Removed structures with unbound water, counter-ions, incorrect bonds

**Experimental Validation:**
- All 690 MOFs are experimentally reported
- Can be directly compared to synthesis results
- Well-characterized structures

**Selection process:**
1. Started with 2,932 CoRE MOF 2014 structures
2. Selected 726 compatible with both UFF and DREIDING
3. Removed 36 problematic structures
4. Final: 690 high-quality, computation-ready MOFs

---

### 2. Force Field & Simulation Quality

#### MOFX-DB
**Approach:**
- Single force field combination (UFF for framework, TraPPE for adsorbates)
- Lorentz-Berthelot mixing rules
- Typical charge scheme (not systematically varied)

**Strengths:**
- Consistent methodology across all MOFs
- Well-established force field combination
- Reproducible via JSON metadata

**Weaknesses:**
- **No uncertainty quantification** - single FF means no error bars
- UFF is "not specific" - may not capture diverse chemical environments
- Cannot assess force field sensitivity
- Unknown accuracy for specific MOFs

**Critical limitation:**
> "UFF exhibits a lack of specificity, as MOFs exhibit a wide variety of chemical environments and a single set of parameters may not capture the nuances of interactions in all systems, leading to discrepancies between computational predictions and experimental results."

#### CRAFTED
**Approach:**
- **Systematic force field exploration:**
  - 2 force fields: UFF, DREIDING
  - 6 partial charge schemes: none, Qeq, EQeq, MPNN, PACMOF, DDEC
  - 12 combinations per MOF per gas
  - 3 temperatures: 273K, 298K, 323K

**Strengths:**
- ✅ **Uncertainty bounds** - can estimate prediction uncertainty
- ✅ **Force field sensitivity analysis** - know which MOFs are FF-dependent
- ✅ **Charge scheme impact** - understand partial charge effects
- ✅ **Temperature dependence** - model thermal effects

**Purpose (from paper):**
> "CRAFTED provides a convenient platform to explore the sensitivity of simulation outcomes to molecular modeling choices at the material (structure-property relationship) and process levels (structure-property-performance relationship)."

**Critical insight:**
> "The original Lennard-Jones force field parameters were derived and validated using specific partial charge schemes (Qeq for UFF and Gasteiger for DREIDING), thus the combination of these parameters with different charge assignment methodologies, even if more accurate, may not necessarily generate better results."

---

### 3. Data Completeness & Usability

#### MOFX-DB
**What you get:**
- Adsorption isotherms (multiple pressures)
- Structural properties (surface area, pore size)
- MOF structure files (CIF)
- Simulation metadata (JSON format)

**Format:**
- JSON files (NIST-compatible format)
- Requires API calls or web interface
- Standardized but complex

**Reproducibility:**
- Full RASPA input files included
- Exact structure used for simulation
- Metadata enables reproduction

**Download:**
- Not straightforward bulk download
- May need to use Python API
- Large data transfer

#### CRAFTED
**What you get:**
- 49,680 CO2 isotherm files
- 49,680 N2 isotherm files
- 49,680 CO2 enthalpy files
- 49,680 N2 enthalpy files
- Charge-assigned CIF files (1,357 per FF)
- Force field definition files
- Jupyter notebooks for analysis

**Format:**
- Isotherm data files (structured text)
- CIF files (standard crystallographic format)
- Organized by force field / charge scheme

**Reproducibility:**
- ✅ All input files included
- ✅ Force field parameters provided
- ✅ Analysis notebooks included
- ✅ Panel visualizations for exploration

**Download:**
- ✅ Single 54.7 MB tar.xz file
- ✅ Direct download from Zenodo
- ✅ Easy to extract and use

---

### 4. Suitability for Active Learning

#### MOFX-DB
**Pros:**
- ✅ Large pool for exploration (160K candidates)
- ✅ Multiple gases (can explore multi-target AL)
- ✅ Hypothetical MOFs = "what could be synthesized"

**Cons:**
- ❌ Hypothetical MOFs may not be synthesizable
- ❌ No uncertainty → can't validate AL selection
- ❌ Single FF → unknown prediction error
- ❌ Hard to justify "economic AL" on hypothetical materials

**Use case:**
- Generative model training (large dataset)
- Exploratory screening (high-throughput)
- Property prediction models (need volume)

#### CRAFTED
**Pros:**
- ✅ **Uncertainty quantification** - critical for AL!
  - Can compare FF/charge schemes → estimate epistemic uncertainty
  - Know which MOFs have high prediction variance
  - Validate AL reduces uncertainty over iterations

- ✅ **Experimental MOFs** - defensible for Economic AL
  - Can actually synthesize and validate
  - Cost estimation makes sense (not hypothetical)
  - Realistic budget-constrained scenario

- ✅ **Quality over quantity**
  - 690 MOFs sufficient for AL demonstration
  - High-quality, curated structures
  - No "junk" hypothetical MOFs

- ✅ **Force field benchmarking**
  - Can show: "AL selected MOF-5, all 12 FF/charge combos agree"
  - Or: "AL avoided MOF-X due to high FF uncertainty"

**Cons:**
- ⚠️ Smaller pool (690 vs 160K)
- ⚠️ Only CO2/N2 (but that's our target application)

**Use case:**
- **Active learning validation** ← Perfect fit
- Uncertainty-aware screening
- Force field sensitivity studies
- Experimental comparison

---

## Quality Assessment

### MOFX-DB Quality Concerns

**1. Hypothetical MOF Validity**
- No experimental synthesis validation
- May violate physical constraints
- Stability unknown
- Synthesizability unclear

**2. Force Field Uncertainty**
- Single FF = unknown error
- Cannot estimate prediction confidence
- May be systematically biased for certain MOF types

**3. Computational Assumptions**
- UFF "lacks specificity" for diverse chemical environments
- TraPPE validated for small molecules, not all MOF-adsorbate pairs
- Single charge scheme may not be optimal

### CRAFTED Quality Strengths

**1. Experimental Foundation**
- All MOFs experimentally reported
- Filtered for computation-readiness
- Known chemical validity

**2. Uncertainty Quantification**
- Systematic FF/charge variation
- Can estimate prediction error bars
- Identifies FF-sensitive vs FF-robust MOFs

**3. Reproducibility**
- All input files provided
- Force field definitions included
- Analysis notebooks for validation

**4. Designed for Sensitivity Analysis**
- Explicitly created to study FF effects
- Enables robust ML model training
- Can train on FF-invariant features

---

## Recommendation for Economic Active Learning

### Primary Dataset: CRAFTED ✅

**Reasons:**

1. **Experimental Validity**
   - Economic AL requires real synthesis costs
   - Hypothetical MOFs have undefined cost
   - Can justify: "Selected MOF-5 (Zn-BDC) costs $1.12/g to synthesize"

2. **Uncertainty Quantification**
   - AL selects high-uncertainty samples
   - CRAFTED provides 12 predictions per MOF
   - Can validate: "Uncertainty decreases as AL progresses"
   - This is **critical** for validating epistemic uncertainty (see uncertainty_quantification_explained.md)

3. **Defensibility**
   - Judge asks: "How do you know your predictions are accurate?"
   - Answer: "Tested across 2 force fields × 6 charge schemes"
   - Judge asks: "Can these be synthesized?"
   - Answer: "All 690 are experimentally reported MOFs"

4. **Practical Size**
   - 690 MOFs perfect for hackathon demo
   - Can run full AL loop in reasonable time
   - Train: 100 MOFs → Pool: 590 MOFs → Iterate

5. **Quality over Quantity**
   - Better 690 high-quality than 160K questionable
   - Economic AL is about **efficiency**, not scale
   - Message: "Smart selection beats brute force"

### Stretch Goal: MOFX-DB (If Time Permits)

**Use case:**
- "Trained on CRAFTED, deployed to screen MOFX-DB"
- "Economic AL found 5 promising hypothetical MOFs"
- "Here's their predicted cost and performance"

**Value:**
- Shows scalability
- Generative design angle
- Broader impact

**But not essential for core demo.**

---

## Implementation Plan

### Week 2 Prework (Current)

**Download CRAFTED:**
```bash
curl -L -O "https://zenodo.org/records/8190237/files/CRAFTED-2.0.0.tar.xz"
tar -xf CRAFTED-2.0.0.tar.xz
```

**Data structure:**
```
CRAFTED-2.0.0/
├── CIF_FILES/
│   ├── UFF_Qeq/      (1,357 CIF files)
│   ├── UFF_DDEC/
│   └── ...
├── ISOTHERM_FILES/   (97,704 isotherms)
│   ├── CO2/
│   └── N2/
├── ENTHALPY_FILES/
└── notebooks/        (Analysis examples)
```

**Extract features:**
- Geometric: LCD, PLD, ASA, density (from CIFs)
- CO2 uptake: From isotherm files @ 1 bar, 298K
- Metal composition: Parse CIF → map to cost estimator

**Create dataset:**
```python
import pandas as pd

# For each MOF:
#   - Read geometric properties
#   - Extract CO2 uptake (@ 1 bar, 298K, UFF+Qeq as baseline)
#   - Compute uncertainty (std across 12 FF/charge combos)
#   - Parse metal type → estimate synthesis cost

df = pd.DataFrame({
    'mof_id': ...,
    'LCD': ...,
    'PLD': ...,
    'ASA': ...,
    'density': ...,
    'co2_uptake_mean': ...,  # Average across 12 methods
    'co2_uptake_std': ...,   # Uncertainty
    'metal': ...,
    'synthesis_cost': ...,
})
```

**Economic AL integration:**
- Use `co2_uptake_mean` as target
- Validate: `co2_uptake_std` should correlate with ensemble uncertainty
- Cost: Link `metal` → MOFCostEstimator

### Hackathon Day (If Needed)

**If time for MOFX-DB:**
1. Use MOFX-DB API to query larger pool
2. Train model on CRAFTED (690 MOFs)
3. Screen MOFX-DB (thousands of MOFs)
4. Identify top candidates
5. Pitch: "Economic AL enables rapid screening at scale"

---

## Key Takeaways

### MOFX-DB
- ✅ Massive scale (160K MOFs)
- ❌ Mostly hypothetical (92%)
- ❌ No uncertainty quantification
- ❌ Single force field
- ⚠️ Hard to download
- **Best for:** Large-scale screening, generative models

### CRAFTED
- ✅ Experimental MOFs (100%)
- ✅ Uncertainty quantification (12 methods)
- ✅ Force field sensitivity
- ✅ Easy download (54.7 MB)
- ✅ Reproducible
- ⚠️ Smaller (690 MOFs)
- **Best for:** Active learning, uncertainty-aware ML, experimental validation

---

## Final Answer

**For Economic Active Learning prework: Use CRAFTED**

**Why:**
1. Experimental validity → defensible cost estimates
2. Uncertainty quantification → validates AL approach
3. High quality → better than large quantity for demo
4. Easy download → can start immediately
5. Perfect size → 690 MOFs ideal for hackathon

**MOFX-DB is impressive but overkill. CRAFTED is exactly what we need.**

---

## References

1. Bobbitt et al. (2023). MOFX-DB: An Online Database of Computational Adsorption Data for Nanoporous Materials. *J. Chem. Eng. Data* 68(2), 483-498.

2. Krishnapriyan et al. (2023). CRAFTED: An exploratory database of simulated adsorption isotherms of metal-organic frameworks. *Sci Data* 10, 230.

3. Chung et al. (2014). Computation-Ready, Experimental Metal-Organic Frameworks. *Chem. Mater.* 26, 6185-6192.

---

**Decision: Download CRAFTED (54.7 MB) for prework**
