# 🔬 Making the Surrogate Model More Robust & Chemistry-Grounded

**Date:** October 10, 2025
**Context:** Surrogate shows weak correlation (r=0.283) for generated MOFs
**Goal:** Improve predictions to make Active Generative Discovery more reliable

---

## 📊 **Current State**

### **What We Have Now:**

**Features (18 dimensions):**
- Metal one-hot encoding (6 metals)
- Linker one-hot encoding (4 linkers)
- Cell parameters (a, b, c, volume)
- Synthesis cost
- Geometric features (5): density, volume_per_atom, packing_fraction, void_fraction_proxy, avg_coordination

**Models Tested:**
- Random Forest baseline: R²=0.234, correlation=-0.04 (FAIL)
- RF + geometric features: R²=0.686, correlation=0.087 (WEAK)
- Gaussian Process + geom features: R²=0.588, correlation=0.283 (VIABLE)

**The Problem:**
- Surrogate works well on real MOFs (R²=0.686)
- Fails to generalize to VAE-generated MOFs (r=0.283)
- Distribution mismatch: real MOFs vs generated MOFs

---

## 🎯 **Improvement Strategies** (Ranked by Impact & Feasibility)

---

## **TIER 1: Quick Wins** (2-4 hours, High Impact)

### **1. Synthesizability Filtering** ⭐ HIGHEST PRIORITY

**What:** Filter out physically impossible MOFs BEFORE they reach the surrogate

**Chemistry Rules:**
- ✅ **Volume sanity** (100-50,000 Å³)
- ✅ **Density range** (0.2-3.0 g/cm³)
- ✅ **Cell parameter ratios** (avoid elongated cells)
- ✅ **Metal-linker compatibility** (coordination geometry)
- ✅ **Linker size vs cell dimensions** (geometric feasibility)

**Implementation:** ✅ **DONE** - `src/validation/synthesizability_filter.py`

**Expected Impact:**
- Remove 20-40% of generated MOFs that are unphysical
- Remaining MOFs should be easier for surrogate to predict
- Improves correlation by filtering outliers

**Test:**
```python
from synthesizability_filter import SynthesizabilityFilter

filter = SynthesizabilityFilter(strict=False)
valid_mofs, rejected = filter.filter_batch(generated_mofs)

print(f"Pass rate: {len(valid_mofs) / len(generated_mofs):.1%}")
```

---

### **2. Chemistry-Informed Features** ⭐⭐ HIGH PRIORITY

**What:** Add features that capture chemical properties, not just structure

**NEW FEATURES TO ADD:**

**A. Electronic Properties (Metal-Based):**
```python
# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    'Zn': 1.65, 'Fe': 1.83, 'Ca': 1.00, 'Al': 1.61,
    'Ti': 1.54, 'Cu': 1.90, 'Zr': 1.33, 'Cr': 1.66
}

# Oxidation state preference
OXIDATION_STATES = {
    'Zn': 2, 'Fe': 2, 'Ca': 2, 'Al': 3,
    'Ti': 4, 'Cu': 2, 'Zr': 4, 'Cr': 3
}

# Ionic radius (6-coordinate, Angstroms)
IONIC_RADII = {
    'Zn': 0.74, 'Fe': 0.78, 'Ca': 1.00, 'Al': 0.54,
    'Ti': 0.61, 'Cu': 0.73, 'Zr': 0.72, 'Cr': 0.62
}

# Hard-Soft Acid-Base classification
HSAB = {
    'Zn': 'borderline', 'Fe': 'borderline', 'Ca': 'hard',
    'Al': 'hard', 'Ti': 'hard', 'Cu': 'borderline',
    'Zr': 'hard', 'Cr': 'hard'
}
```

**B. Linker Properties:**
```python
# Linker length (carboxylate separation, Angstroms)
LINKER_LENGTH = {
    'terephthalic acid': 11.0,
    'trimesic acid': 9.5,
    '2,6-naphthalenedicarboxylic acid': 13.0,
    'biphenyl-4,4-dicarboxylic acid': 15.0
}

# Number of coordination sites
LINKER_DENTICITY = {
    'terephthalic acid': 2,
    'trimesic acid': 3,
    '2,6-naphthalenedicarboxylic acid': 2,
    'biphenyl-4,4-dicarboxylic acid': 2
}

# Linker rigidity (0=flexible, 1=rigid)
LINKER_RIGIDITY = {
    'terephthalic acid': 0.9,
    'trimesic acid': 1.0,
    '2,6-naphthalenedicarboxylic acid': 0.8,
    'biphenyl-4,4-dicarboxylic acid': 0.6  # Can rotate
}

# Aromatic ring count
LINKER_AROMATICITY = {
    'terephthalic acid': 1,
    'trimesic acid': 1,
    '2,6-naphthalenedicarboxylic acid': 2,
    'biphenyl-4,4-dicarboxylic acid': 2
}
```

**C. Derived Chemical Features:**
```python
def compute_chemistry_features(mof):
    metal = mof['metal']
    linker = mof['linker']

    # Metal-linker compatibility score
    compatibility = compute_compatibility(
        ELECTRONEGATIVITY[metal],
        IONIC_RADII[metal],
        LINKER_LENGTH[linker],
        LINKER_DENTICITY[linker]
    )

    # Pore size estimate (chemical intuition)
    pore_size = LINKER_LENGTH[linker] - 2 * IONIC_RADII[metal]

    # Coordination number estimate
    coord_number = estimate_coordination(
        IONIC_RADII[metal],
        LINKER_DENTICITY[linker]
    )

    # Chemical affinity (for CO2)
    co2_affinity = estimate_co2_affinity(
        HSAB[metal],  # Lewis acidity
        LINKER_AROMATICITY[linker],  # π-π interactions
        pore_size  # Accessibility
    )

    return [
        ELECTRONEGATIVITY[metal],
        IONIC_RADII[metal],
        LINKER_LENGTH[linker],
        LINKER_RIGIDITY[linker],
        LINKER_AROMATICITY[linker],
        compatibility,
        pore_size,
        coord_number,
        co2_affinity
    ]
```

**Expected Impact:** +10-20% correlation improvement (r=0.283 → 0.35-0.40)

**Implementation Time:** 2-3 hours

---

### **3. Ensemble with Chemistry-Aware Models** ⭐ MODERATE PRIORITY

**What:** Combine multiple model types, some with built-in chemistry

**Ensemble Strategy:**
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge

# Ensemble with different inductive biases
models = {
    'rf': RandomForestRegressor(),  # Non-linear, feature interactions
    'gb': GradientBoostingRegressor(),  # Sequential learning
    'gp': GaussianProcessRegressor(),  # Uncertainty quantification
    'ridge': Ridge(alpha=1.0)  # Linear baseline (chemistry often linear)
}

# Weighted ensemble based on validation performance
weights = optimize_weights(models, val_set)

# Prediction
prediction = sum(w * model.predict(X) for w, model in zip(weights, models.values()))
```

**Why This Helps:**
- Different models capture different aspects of chemistry
- Ridge might capture simple linear relationships (often true in chemistry)
- RF/GB capture non-linear interactions
- GP provides uncertainty

**Expected Impact:** +5-10% correlation improvement

**Implementation Time:** 2 hours

---

## **TIER 2: Medium-Term Improvements** (1-2 days, Moderate-High Impact)

### **4. Graph Neural Networks** ⭐⭐⭐ HIGHEST LONG-TERM IMPACT

**What:** Represent MOFs as graphs instead of fixed feature vectors

**Why This Is Better:**
- Captures topology (how atoms are connected)
- Learns representations instead of hand-crafted features
- Naturally handles variable-size structures
- State-of-the-art in molecular property prediction

**Implementation (using PyTorch Geometric):**
```python
import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class MOFGraphNet(torch.nn.Module):
    def __init__(self, node_features=10, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Graph-level embedding
        x = global_mean_pool(x, batch)

        # Prediction
        return self.fc(x)

# Graph representation
graph = {
    'x': node_features,  # Metal/linker properties
    'edge_index': bonds,  # Connectivity
    'edge_attr': bond_types  # Single/double/coordination
}
```

**Node Features:**
- Atomic number, electronegativity, radius
- Coordination number
- Local geometry (bond angles)

**Edge Features:**
- Bond type (covalent, coordination)
- Bond length
- Bond order

**Expected Impact:** +20-30% correlation improvement (r=0.283 → 0.40-0.50)

**Challenge:** Requires graph construction from MOF structure (need atomic coordinates)

**Implementation Time:** 1-2 days (if we have coordinates), 1 week (if we need to generate coordinates)

---

### **5. Transfer Learning from Related Datasets** ⭐⭐ MODERATE IMPACT

**What:** Pre-train on larger, related datasets

**Strategy:**
```python
# Phase 1: Pre-train on large dataset (e.g., CoRE MOF database)
pretrain_dataset = load_core_mof_database()  # ~10,000 MOFs
model.pretrain(pretrain_dataset, property='surface_area')

# Phase 2: Fine-tune on CO2 uptake
model.finetune(co2_dataset, property='co2_uptake')

# Phase 3: Domain adaptation for generated MOFs
# Add a loss term to handle distribution shift
loss = mse_loss(predictions, targets) + \
       domain_adaptation_loss(real_mofs, generated_mofs)
```

**Datasets to Consider:**
- CoRE MOF database (~14,000 MOFs)
- CSD MOF subset (~80,000 structures)
- QMOF database (~20,000 with DFT properties)

**Expected Impact:** +10-15% correlation improvement

**Challenge:** Need to acquire and process additional datasets

**Implementation Time:** 2-3 days

---

### **6. Multi-Task Learning** ⭐⭐ MODERATE IMPACT

**What:** Predict multiple properties simultaneously

**Rationale:** Properties are correlated, multi-task learning helps generalization

```python
class MultiTaskSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(...)  # Shared layers

        # Task-specific heads
        self.co2_head = nn.Linear(hidden_dim, 1)
        self.surface_area_head = nn.Linear(hidden_dim, 1)
        self.pore_volume_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            'co2': self.co2_head(h),
            'surface_area': self.surface_area_head(h),
            'pore_volume': self.pore_volume_head(h)
        }

# Loss combines all tasks
loss = w1 * mse(pred_co2, target_co2) + \
       w2 * mse(pred_sa, target_sa) + \
       w3 * mse(pred_pv, target_pv)
```

**Expected Impact:** +5-10% correlation improvement

**Challenge:** Need additional property labels

**Implementation Time:** 1 day (if we have labels)

---

## **TIER 3: Long-Term Solutions** (1-2 weeks, Highest Impact)

### **7. DFT/Molecular Dynamics Screening** ⭐⭐⭐ GOLD STANDARD

**What:** Use quantum chemistry to validate generated MOFs

**Workflow:**
```
Generated MOF → DFT optimization → Energy minimization →
  Stability check → GCMC simulation → CO2 uptake prediction
```

**Tools:**
- VASP / Quantum ESPRESSO (DFT)
- LAMMPS (MD)
- RASPA (GCMC)

**Why This Works:**
- First-principles physics
- No machine learning errors
- Validates synthesizability
- Provides high-quality labels for retraining surrogate

**Expected Impact:** Near-perfect predictions (r > 0.9)

**Challenge:**
- Computational cost (~1-10 CPU-hours per MOF)
- Requires HPC infrastructure
- Need expertise to set up

**Implementation Time:** 1-2 weeks (setup), then ongoing

---

### **8. Physics-Guided Neural Networks** ⭐⭐ HIGH IMPACT

**What:** Incorporate physical laws directly into the model

**Example: Langmuir Isotherm Constraint**
```python
# CO2 uptake often follows Langmuir model
# Q = Q_max * K * P / (1 + K * P)

class PhysicsGuidedSurrogate(nn.Module):
    def forward(self, x):
        # Neural network predicts Langmuir parameters
        q_max = self.nn_qmax(x)  # Maximum capacity
        K = self.nn_K(x)  # Adsorption constant

        # Pressure (fixed for our case, e.g., 1 bar)
        P = 1.0

        # Langmuir equation (physics constraint)
        co2_uptake = q_max * K * P / (1 + K * P)

        return co2_uptake
```

**Why This Helps:**
- Enforces thermodynamic constraints
- Extrapolates better to OOD data
- Interpretable predictions

**Expected Impact:** +15-25% correlation improvement

**Implementation Time:** 3-5 days

---

## **TIER 4: VAE Improvements** (Alternative Approach)

Instead of improving the surrogate, make the VAE generate better MOFs:

### **9. Constrained VAE Generation**

**What:** Add physics constraints to VAE training

```python
# VAE loss with physics penalties
loss = reconstruction_loss + \
       kl_divergence + \
       synthesizability_penalty + \
       diversity_bonus

# Synthesizability penalty
def synthesizability_penalty(generated_mof):
    violations = 0
    if not check_volume_range(mof): violations += 1
    if not check_density_range(mof): violations += 1
    if not check_coordination(mof): violations += 1
    return violations * penalty_weight
```

**Expected Impact:** Better quality generated MOFs → easier for surrogate

**Implementation Time:** 2-3 days (retrain VAE)

---

## 📊 **Recommended Action Plan**

### **For Hackathon (Next 4 hours):**

1. ✅ **Synthesizability filtering** (DONE - 0 hours)
2. **Add chemistry-informed features** (2-3 hours)
   - Electronegativity, ionic radii
   - Linker properties (length, rigidity, aromaticity)
   - Derived features (pore size estimate, compatibility score)
3. **Test ensemble approach** (1 hour)
   - Add GradientBoosting + Ridge to GP
   - Weight by validation performance

**Expected Result:** r = 0.283 → 0.35-0.42 (+20-50% improvement)

---

### **Post-Hackathon (Week 1-2):**

4. **Implement Graph Neural Network** (if coordinates available)
5. **Multi-task learning** (if additional labels available)
6. **Transfer learning from CoRE MOF database**

**Expected Result:** r = 0.40-0.50 (usable for production)

---

### **Production (Month 1-2):**

7. **DFT screening pipeline**
   - Set up on HPC cluster
   - Screen generated MOFs before experimental validation
   - Use DFT results to retrain surrogate

**Expected Result:** r > 0.9 (gold standard)

---

## 🎯 **Bottom Line**

**The Real Problem:** Distribution mismatch (real vs generated MOFs)

**Quick Fixes (Hackathon):**
- ✅ Filter unphysical MOFs
- ✅ Add chemistry-grounded features
- ✅ Use ensemble methods

**Medium-Term (Post-Hackathon):**
- Graph neural networks
- Transfer learning
- Multi-task learning

**Long-Term (Production):**
- DFT screening (gold standard)
- Physics-guided neural networks
- Constrained VAE generation

**Current Status:** With filtering + chemistry features + ensemble, we can likely reach r=0.35-0.42, which is VIABLE for hackathon demo with exploration bonus strategy.
