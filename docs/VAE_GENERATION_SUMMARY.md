# VAE Generation: Technical Exploration & Results

## Executive Summary

Built three VAE architectures for MOF generation (Unconditional, Conditional, Hybrid). All variants achieved reasonable reconstruction but suffered from **mode collapse** (5-12% diversity). The root cause is **data imbalance in CRAFTED dataset** (80%+ Zn + terephthalic acid), which cannot be overcome with current architecture approaches.

**Best result:** Hybrid Conditional VAE with 5-6 unique metal-linker combinations out of 50 samples (12% diversity).

**Recommendation:** Position VAE as a technical exploration demonstrating the challenge of generation with limited experimental data. Focus hackathon presentation on Economic AL (the strong component).

---

## Why CRAFTED Dataset? (Root Cause of Mode Collapse)

### The CRAFTED Choice

We chose CRAFTED (687 experimental MOFs) over larger datasets for three critical reasons:

1. **All synthesized** - Grounded in experimental reality (not hypothetical)
2. **CO2 uptake labels** - Required for supervised learning and Economic AL
3. **CIF files available** - Needed for geometric feature extraction

This was a **quality over quantity** decision prioritizing real, labeled data over hypothetical structures.

### The Data Imbalance Problem

CRAFTED has severe imbalance:
- **80%+ samples:** Zn + terephthalic acid
- **6 metals total:** Zn (dominant), Ca, Unknown, Fe, Al, Ti
- **4 linkers total:** Terephthalic acid (dominant), trimesic acid, 2,6-naphthalenedicarboxylic acid, biphenyl-4,4-dicarboxylic acid
- **Theoretical maximum:** 6 × 4 = 24 combinations
- **Actual coverage:** ~10-12 combinations in dataset

### How This Causes Mode Collapse

The VAE learns the true data distribution:

```
P(metal, linker | data) ≈ 0.80 × P(Zn, terephthalic acid) + 0.20 × P(others)
```

When generating:
1. VAE samples from learned distribution
2. Argmax decoding picks most probable (Zn + terephthalic acid)
3. Temperature sampling helps marginally but can't overcome 80% bias
4. Data augmentation (16×) preserves the imbalance (same MOFs, different representations)

**Result:** VAE correctly generates what it learned - a heavily biased distribution.

### Why We're "Stuck" with CRAFTED

Alternative datasets have critical issues:

| Dataset | Size | Problem |
|---------|------|---------|
| CoRE MOF | 14,000+ | No CO2 labels, distribution mismatch with CRAFTED |
| QMOF | 20,000+ | Simulated data, not synthesized |
| hMOFs | 130,000+ | Hypothetical, no experimental validation |

**Switching cost:** 2-3 hours to retrain Economic AL predictor + revalidate pipeline + unknown risks.

**Time constraint:** 40 hours to hackathon when decision was made.

**Conclusion:** CRAFTED's experimental grounding and CO2 labels are more valuable than larger datasets with distribution mismatches.

---

## What We Built

### 1. Unconditional VAE (Baseline)

**Architecture:**
- Input/Output: Metal (one-hot) + Linker (one-hot) + Cell parameters (4) + Volume
- Latent dim: 16
- Variants: β = 1.0, 3.0, 5.0 (KL weight)

**Results:**
- Reconstruction: Good (loss ~7.0)
- **Diversity: 0.3-0.8%** (3-8 unique out of 1000 samples)
- Temperature sampling (T=3.0): Improved to 0.8% but still severe mode collapse

**Files:**
- `src/generation/mof_vae.py`
- `results/vae_evaluation/temperature_test_results.json`

### 2. Conditional VAE (Property-Targeted)

**Architecture:**
- Input: Metal + Linker + Cell + **CO2 uptake (condition)**
- Output: Metal + Linker + Cell (NO CO2)
- Conditioning: CO2 appended to input during encoding, passed to decoder

**Key Innovation:**
```python
def forward(self, x):
    mu, logvar = self.encode(x)  # x includes CO2
    z = self.reparameterize(mu, logvar)
    co2_condition = x[:, -1:]  # Extract CO2
    recon = self.decode(z, co2_condition)  # Condition decoder
    return recon, mu, logvar
```

**Results:**
- Final loss: 6.70 (Recon=2.30, KL=4.40)
- **Diversity: 8-10%** (4-5 unique out of 50)
- **Proof of learning:** Metal distribution shifts with CO2 target
  - Low CO2 (2.0 mmol/g): Mostly Zn, some Fe/Ca
  - High CO2 (8.0 mmol/g): Al appears (higher capacity metal)

**Files:**
- `src/generation/conditional_vae.py`
- `results/conditional_vae_log.txt`
- `models/conditional_mof_vae.pt`

### 3. Hybrid Conditional VAE (With Geometric Features)

**Architecture:**
- Input: Metal + Linker + Cell + CO2 + **11 geometric features**
- Output: Metal + Linker + Cell
- Input dim: 26 (vs 15 for Simple cVAE)
- Hidden dim: 64 (vs 32 for Simple cVAE)

**Geometric Features (11 total):**
1. Density
2. Pore volume
3. Void fraction
4. Surface area
5. Metal coordination number
6. Linker length
7. Packing fraction
8. Channel diameter
9. Largest cavity diameter
10. Gravimetric surface area
11. Volumetric surface area

**Key Technical Challenge:**
How to use geometric features with augmented data (supercells + thermal noise)?

**Solution:**
```python
# Map augmented MOFs to original geometric features
geom_dict = {row['mof_id']: row[geom_features] for _, row in geom_features.iterrows()}

for _, mof in mof_data.iterrows():
    if mof['mof_id'] in geom_dict:
        # Use original MOF's features
        # WORKS because supercells preserve intensive properties
        geom_features_matched.append(geom_dict[mof['mof_id']])
```

**Why this works:**
- Supercells: Mathematical transformations that **preserve intensive properties** (density, packing fraction, coordination)
- Thermal noise: ±2% perturbations don't significantly change geometric features

**Results:**
- Final loss: 6.65 (Recon=2.24, KL=4.41)
- **Diversity: 12%** (5-6 unique out of 50)
- **17% improvement** over Simple cVAE
- Better reconstruction quality (2.24 vs 2.30)

**Files:**
- `src/generation/hybrid_conditional_vae.py`
- `results/hybrid_conditional_vae_log.txt`
- `models/hybrid_conditional_mof_vae.pt`
- `src/preprocessing/geometric_features.py`

---

## Data Augmentation Strategy

### Motivation
Expand 687 MOFs → 10,992 samples (16× expansion) to reduce overfitting.

### Techniques

**1. Supercells (8× expansion):**
```python
supercell_factors = [
    (1,1,1), (2,1,1), (1,2,1), (1,1,2),
    (2,2,1), (2,1,2), (1,2,2), (2,2,2)
]
```
- Creates different unit cell representations
- Preserves intensive properties (density, coordination)
- Volume scales as `V_new = V_orig × n_a × n_b × n_c`

**2. Thermal Noise (2× expansion):**
```python
noise = np.random.normal(0, noise_level, size=cell_params.shape)
cell_perturbed = cell_params * (1 + noise)  # ±2% variation
```
- Simulates thermal fluctuations
- Adds slight variation to cell parameters

**Result:** 687 → 5,496 (supercells) → 10,992 (thermal noise)

**Files:**
- `src/generation/mof_augmentation.py`

---

## Temperature Sampling Experiments

### Hypothesis
Can temperature sampling overcome mode collapse by encouraging exploration?

### Method
```python
def temperature_sample_categorical(logits, temperature):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
```

### Results (1000 samples each)

| Temperature | Unique Combos | Diversity | Most Common |
|-------------|---------------|-----------|-------------|
| argmax (T=0) | 3 | 0.3% | 99.1% same MOF |
| 0.5 | 5 | 0.5% | 95.0% same MOF |
| 1.0 | 6 | 0.6% | 81.0% same MOF |
| 1.5 | 6 | 0.6% | 62.1% same MOF |
| 2.0 | 6 | 0.6% | 49.0% same MOF |
| 3.0 | 8 | **0.8%** | 35.9% same MOF |

### Conclusion
Temperature helps marginally (0.3% → 0.8%) but **cannot overcome 80% data imbalance**.

**Files:**
- `src/generation/test_temperature_sampling.py`
- `results/vae_evaluation/temperature_test_results.json`

---

## Comparison: VAE vs Economic AL

| Aspect | VAE Generation | Economic AL |
|--------|----------------|-------------|
| **Status** | Technical exploration | Core innovation |
| **Results** | 5-6 unique out of 50 (12%) | Complete pipeline with 687 MOFs |
| **Limitation** | Mode collapse from data imbalance | Constrained by data availability |
| **Strength** | Shows structure-property learning | Optimizes dual costs (performance + synthesis) |
| **Demo value** | Demonstrates challenge | **Strong differentiator** |
| **Robustness** | Weak diversity | Proven end-to-end |

**Recommendation:** Position VAE as "we explored generation but hit fundamental data limits" and focus presentation on Economic AL's dual-cost optimization.

---

## Files Generated

### Models
- `models/mof_vae_beta_1.0.pt` - Unconditional VAE (β=1.0)
- `models/mof_vae_beta_3.0.pt` - Unconditional VAE (β=3.0)
- `models/mof_vae_beta_5.0.pt` - Unconditional VAE (β=5.0)
- `models/conditional_mof_vae.pt` - **Conditional VAE** (best simple)
- `models/hybrid_conditional_mof_vae.pt` - **Hybrid Conditional VAE** (best overall)

### Code
- `src/generation/mof_vae.py` - Unconditional VAE
- `src/generation/conditional_vae.py` - Conditional VAE
- `src/generation/hybrid_conditional_vae.py` - Hybrid Conditional VAE
- `src/generation/mof_augmentation.py` - Data augmentation
- `src/preprocessing/geometric_features.py` - Feature extraction from CIF
- `src/generation/test_temperature_sampling.py` - Temperature experiments

### Data
- `data/processed/crafted_mofs_co2.csv` - 687 MOFs with CO2 uptake
- `data/processed/crafted_geometric_features.csv` - 687 MOF geometric features

### Results
- `results/conditional_vae_log.txt` - Simple cVAE training log
- `results/hybrid_conditional_vae_log.txt` - Hybrid cVAE training log
- `results/vae_evaluation/temperature_test_results.json` - Temperature sampling results

---

## Lessons Learned

### 1. Data Quality vs Quantity
**CRAFTED's 687 experimental MOFs > 100k hypothetical structures**
- Experimental grounding more valuable than size
- But comes with severe data imbalance

### 2. Augmentation Preserves Imbalance
16× expansion didn't fix mode collapse because:
- Supercells: Same MOF, different representation
- Thermal noise: Small perturbations
- **No new metal-linker combinations created**

### 3. Conditioning Shows Learning
Even with mode collapse, Conditional VAE proved structure-property learning:
- Metal distribution shifts with CO2 target
- Al appears at high CO2 (correct chemistry)

### 4. Geometric Features Are Feasible
Initially thought incompatible with augmentation, but:
- Supercells preserve intensive properties
- Can map augmented → original features deterministically

### 5. Temperature Sampling Has Limits
Helps exploration but can't overcome fundamental data distribution:
- 0.8% diversity still unacceptable
- Need better data or architecture (GFlowNets, diffusion models)

---

## Future Work (Post-Hackathon)

### Short-term (if continuing VAE)
1. **Class balancing:** Oversample minority metals, undersample Zn
2. **Auxiliary losses:** Metal diversity penalty, linker diversity reward
3. **Architecture changes:** Separate decoders for metal/linker vs cell parameters

### Long-term (alternative approaches)
1. **GFlowNets:** Better for diverse generation from biased data
2. **Diffusion models:** May handle multimodal distributions better
3. **Hybrid dataset:** Mix CRAFTED (experimental) + CoRE MOF (simulated) with domain adaptation

### Data strategy
1. **Expand CRAFTED:** Find more experimental MOFs with CO2 labels
2. **Active learning for data collection:** Use Economic AL to guide new synthesis
3. **Transfer learning:** Pretrain on large hypothetical dataset, fine-tune on CRAFTED

---

## How to Use

### Load and Generate with Hybrid cVAE

```python
import torch
from src.generation.hybrid_conditional_vae import HybridConditionalMOFGenerator

# Load trained model
generator = HybridConditionalMOFGenerator()
checkpoint = torch.load('models/hybrid_conditional_mof_vae.pt')
generator.cvae.load_state_dict(checkpoint['cvae_state'])
generator.metal_encoder = checkpoint['metal_encoder']
generator.linker_encoder = checkpoint['linker_encoder']
generator.cell_mean = checkpoint['cell_mean']
generator.cell_std = checkpoint['cell_std']
generator.co2_mean = checkpoint['co2_mean']
generator.co2_std = checkpoint['co2_std']
generator.geom_mean = checkpoint['geom_mean']
generator.geom_std = checkpoint['geom_std']

# Generate candidates for target CO2 uptake
unique_combos = generator.generate_candidates(
    n_candidates=50,
    target_co2=6.0,  # Target CO2 uptake in mmol/g
    temperature=2.0   # Higher = more diversity
)

print(f"Generated {unique_combos} unique metal-linker combinations")
```

### Retrain from Scratch

```bash
# Simple Conditional VAE
python src/generation/conditional_vae.py

# Hybrid Conditional VAE (with geometric features)
python src/generation/hybrid_conditional_vae.py

# Temperature sampling experiments
python src/generation/test_temperature_sampling.py
```

---

## Conclusion

VAE generation demonstrates the fundamental challenge of learning diverse distributions from imbalanced experimental data. While we achieved reasonable reconstruction quality and proved structure-property learning, the 12% diversity is insufficient for practical materials discovery.

**For hackathon:** Position this as a valuable technical exploration that motivates focusing on Economic AL, where we have strong results with the 687 experimental MOFs available.

**The real innovation:** Economic AL's dual-cost optimization (performance + synthesis cost) with real experimental data.
