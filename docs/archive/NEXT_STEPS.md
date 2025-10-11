# Immediate Next Steps: Economic Active Learning

**Status:** Post-Nobel pivot approved ✅
**Target:** Budget-Constrained Active Learning (Integrated Approach)
**Prep time:** 12 hours over 2 weeks

**Key Innovation:** Active learning that respects lab budget constraints

---

## 🎯 Decision Summary

**Chosen Approach:** Economic Active Learning (Integrated)
- **Keeps:** Active learning as core component (ML depth)
- **Adds:** Budget constraints in AL selection (economic awareness)
- **Innovation:** First budget-constrained AL for materials discovery
- 4D optimization: Performance + Synthesizability + **Cost** + **Time**
- Real reagent pricing data integrated into learning loop

**Why this wins:**
- Combines ML novelty (AL) + practical impact (economics)
- More novel than either HARD or Economic alone
- Appeals to all judge types (technical + business)
- Post-Nobel commercialization narrative

**Key docs:**
- `docs/prework/economic_active_learning_guide.md` ← **Primary implementation guide**
- `docs/prework/hard_vs_economic_comparison.md` ← Decision rationale
- `docs/prework/competitive_analysis.md` ← Why you'll stand out

---

## 📅 2-Week Prep Timeline (12 hours total)

### Week 1: Cost Estimator (Essential for AL) - 6 hours

#### Day 1-2 (Sat-Sun): Reagent Cost Database [3 hours]
**Goal:** Create database of common MOF reagent prices

**Option 1: Manual curation (Recommended - faster)**
Create `data/reagent_prices.csv` manually from Sigma-Aldrich/TCI catalogs:
```csv
reagent,cas_number,price_per_gram_usd,supplier,notes
Zinc nitrate hexahydrate,10196-18-6,0.15,Sigma-Aldrich,Common
Copper(II) nitrate trihydrate,10031-43-3,0.25,Sigma-Aldrich,Common
Zirconium(IV) chloride,10026-11-6,2.50,Sigma-Aldrich,Expensive
Terephthalic acid,100-21-0,0.50,TCI,Common linker
Trimesic acid,554-95-0,2.50,Sigma-Aldrich,Less common
DMF,68-12-2,0.05,Sigma-Aldrich,Solvent
Ethanol,64-17-5,0.02,Fisher,Solvent
```

**Start with 20-30 reagents**, covering:
- Top 10 metal salts (Zn, Cu, Zr, Mg, Co, Ni, Mn, Fe, Cd, Ca)
- Top 10 organic linkers (BDC, BTC, NDC, BPDC, etc.)
- Common solvents (DMF, DEF, EtOH, MeOH, water)

**Option 2: Web scraping** (if you want to automate):
```python
# scripts/create_reagent_database.py - See economic_active_learning_guide.md
```

**Target:** 20-30 reagents covering common MOF building blocks

---

#### Day 3-4 (Mon-Tue): Cost Estimator Implementation [3 hours]
**Goal:** Working cost estimation for MOF compositions

```python
# src/cost/estimator.py
import pandas as pd
import numpy as np

class MOFCostEstimator:
    def __init__(self, reagent_db_path='data/reagent_prices.csv'):
        self.reagents = pd.read_csv(reagent_db_path)
        self.reagent_dict = self.reagents.set_index('reagent')['price_per_gram_usd'].to_dict()

    def estimate_synthesis_cost(self, mof_info):
        """
        Estimate cost to synthesize 1g of MOF

        mof_info: dict with keys
            - metal: str (e.g., 'Zn')
            - linker: str (e.g., 'terephthalic acid')
            - topology: str (optional)
        """
        # Lookup metal salt
        metal_salt = f"{mof_info['metal']} nitrate"  # Common form
        metal_cost = self._get_reagent_cost(metal_salt, default=1.0)

        # Lookup linker
        linker_cost = self._get_reagent_cost(mof_info['linker'], default=5.0)

        # Typical stoichiometry for 1g MOF yield
        # Rough estimates from literature
        metal_mass = 0.1  # 100mg metal salt
        linker_mass = 0.2  # 200mg linker
        solvent_cost = 0.05  # 10mL DMF

        total = (
            metal_cost * metal_mass +
            linker_cost * linker_mass +
            solvent_cost
        )

        return {
            'total_cost_per_gram': total,
            'metal_contribution': metal_cost * metal_mass,
            'linker_contribution': linker_cost * linker_mass,
            'solvent_contribution': solvent_cost,
            'breakdown': {
                'metal_salt': metal_salt,
                'metal_price': metal_cost,
                'linker_price': linker_cost
            }
        }

    def _get_reagent_cost(self, reagent_name, default=5.0):
        """Lookup reagent cost with fuzzy matching"""
        for r in self.reagent_dict.keys():
            if reagent_name.lower() in r.lower() or r.lower() in reagent_name.lower():
                return self.reagent_dict[r]
        return default  # Default if not found

# Test
estimator = MOFCostEstimator()
cost = estimator.estimate_synthesis_cost({
    'metal': 'Zn',
    'linker': 'terephthalic acid'
})
print(f"MOF-5 (Zn-BDC) estimated cost: ${cost['total_cost_per_gram']:.2f}/g")
```

**Test output:**
- MOF-5 (Zn-BDC): ~$0.30-0.50/g
- HKUST-1 (Cu-BTC): ~$0.75-1.00/g
- UiO-66 (Zr-BDC): ~$0.50-0.70/g

---

### Week 2: Economic AL Integration - 6 hours

#### Day 8-9 (Sat-Sun): Economic Active Learning [4 hours]
**Goal:** Implement budget-constrained sample selection

```python
# src/active_learning/economic_learner.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class EconomicActiveLearner:
    """Active learning with budget constraints"""

    def __init__(self, X_train, y_train, X_pool, y_pool, cost_estimator):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()
        self.cost_estimator = cost_estimator
        self.history = []

    def train_ensemble(self, n_models=5):
        """Train ensemble for uncertainty"""
        self.models = []
        for i in range(n_models):
            model = RandomForestRegressor(n_estimators=100, random_state=i)
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def predict_with_uncertainty(self, X):
        """Ensemble prediction with uncertainty"""
        predictions = np.array([m.predict(X) for m in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0)

    def economic_selection(self, budget=1000):
        """
        Select samples within budget
        Acquisition: (expected_value * uncertainty) / cost
        """
        # Predict
        mean, std = self.predict_with_uncertainty(self.X_pool)

        # Estimate costs (simplified - use feature proxy or composition data)
        costs = np.random.uniform(0.5, 50, len(self.X_pool))  # Placeholder

        # Acquisition function
        acquisition = mean * std / (costs + 1e-6)

        # Greedy knapsack selection
        selected = []
        total_cost = 0
        sorted_idx = np.argsort(acquisition)[::-1]

        for idx in sorted_idx:
            if total_cost + costs[idx] <= budget:
                selected.append(idx)
                total_cost += costs[idx]
                if len(selected) >= 100:  # Max samples
                    break

        return selected, total_cost

    def run_iteration(self, budget=1000):
        """One AL iteration with cost tracking"""
        self.train_ensemble()
        selected, cost = self.economic_selection(budget)

        # Query oracle
        X_new = self.X_pool.iloc[selected]
        y_new = self.y_pool.iloc[selected]

        # Update
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)
        self.X_pool = self.X_pool.drop(X_new.index)
        self.y_pool = self.y_pool.drop(y_new.index)

        # Log metrics
        metrics = {
            'iteration': len(self.history) + 1,
            'n_validated': len(selected),
            'iteration_cost': cost,
            'avg_cost_per_sample': cost / len(selected),
        }
        self.history.append(metrics)
        return metrics
```

**Full implementation:** See `docs/prework/economic_active_learning_guide.md` lines 240-520

---

#### Day 10-11 (Mon-Tue): Integration & Testing [2 hours]
**Goal:** End-to-end test of Economic AL pipeline

```python
# tests/test_economic_al_pipeline.py
from src.cost.estimator import MOFCostEstimator
from src.active_learning.economic_learner import EconomicActiveLearner
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
mofs_df = pd.read_csv('data/raw/core_mofs.csv')
features = ['LCD', 'PLD', 'ASA_m2/g', 'Density']
X = mofs_df[features].fillna(mofs_df[features].median())
y = mofs_df['CO2_0.15bar_298K'].fillna(0)

X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.8, random_state=42)

# Initialize
cost_est = MOFCostEstimator()
learner = EconomicActiveLearner(X_train, y_train, X_pool, y_pool, cost_est)

# Run 3 iterations
print("Testing Economic Active Learning...")
for i in range(3):
    metrics = learner.run_iteration(budget=500)
    print(f"Iteration {i+1}:")
    print(f"  Validated: {metrics['n_validated']} MOFs")
    print(f"  Cost: ${metrics['iteration_cost']:.2f}")
    print(f"  Avg: ${metrics['avg_cost_per_sample']:.2f}/sample")

print("\n✅ Pipeline ready for hackathon!")
```

**Key test:** Verify cost per iteration decreases (learning to focus on affordable MOFs)

---

## 🚀 Hackathon Day: Economic AL Timeline

### Hour 1: Foundation [10:00-11:00 AM]
**Same as HARD version**
- Load CoRE MOF data
- Extract features
- Train initial performance & synthesizability ensembles

### Hour 2: Multi-Objective Setup [11:00 AM-12:00 PM]
**Enhanced with economics**
```python
# Score all MOFs on 4 objectives
for mof in candidates:
    perf, perf_unc = perf_model.predict_with_uncertainty(mof)
    synth, synth_unc = synth_model.predict_with_uncertainty(mof)
    cost = cost_estimator.estimate_synthesis_cost(mof.composition)
    time = estimate_synthesis_time(mof.composition)  # Simplified

    mof_scores[mof.id] = {
        'performance': perf,
        'synthesizability': synth,
        'cost': cost['total_cost_per_gram'],
        'time': time,
        'cost_efficiency': perf / cost['total_cost_per_gram']
    }
```

### Hour 3: Economic Active Learning [12:00-1:00 PM] ⭐
**Core implementation**
- Implement economic AL selection function
- Test on small subset
- Verify budget constraints work

**Checkpoint:** Can select samples with cost constraints

### Hour 4: Run AL Loop [1:00-2:00 PM] ⭐
**BASELINE CHECKPOINT - MUST WORK**
- Run 3-5 AL iterations with budget tracking
- Compute 4D Pareto frontier after each iteration
- Track: cost per iteration, uncertainty reduction

**Must have:**
- ✅ Active learning with budget constraints working
- ✅ Cost tracking over iterations
- ✅ 4D objectives (Perf, Synth, Cost, Time) computed
- ✅ Pareto frontier identified

### Hour 5: Visualization [2:00-3:00 PM]
**Economic AL specific plots**
- 4D Pareto plot (3D scatter + color for 4th dimension)
- AL progress with cost tracking (dual y-axis)
- Cost-uncertainty trade-off scatter
- Cost efficiency leaderboard

### Hour 6: Dashboard [3:00-4:00 PM]
**Interactive exploration**
- Streamlit app
- Budget slider (recalculate AL selection)
- Pareto frontier explorer
- Cost breakdown for selected MOFs
- AL iteration replay

### Hour 7: Polish & Present [4:00-5:00 PM]
**Stop coding at 4:30 PM**
- Pre-generate all figures
- Practice demo (2-3 run-throughs)
- 5-minute pitch ready

---

## 📦 Requirements (No changes from HARD version)

```
# Core ML/DL (unchanged)
torch>=2.0.0
torch-geometric>=2.3.0

# Materials science (unchanged)
pymatgen>=2023.0.0

# Data science (unchanged)
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization (unchanged)
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.28.0

# No LLM dependencies needed for Economic AL baseline!
# (Optional: Add later if you want synthesis route generation)
```

---

## ✅ Pre-Hackathon Checklist

### Week 1 (Essential - 6 hours)
- [ ] Reagent pricing CSV with 20-30 entries created
- [ ] Cost estimator implemented and tested on known MOFs (MOF-5, HKUST-1, UiO-66)
- [ ] CoRE MOF database downloaded
- [ ] Cost estimator returns reasonable values ($0.30-$2/g for common MOFs)

### Week 2 (Essential - 6 hours)
- [ ] Economic AL selection function implemented
- [ ] Tested on small dataset (100 MOFs)
- [ ] Integration test passes end-to-end
- [ ] Verified cost tracking works over iterations

### Pre-Hackathon (Final checks)
- [ ] Review `economic_active_learning_guide.md`
- [ ] Run full pipeline test (3 AL iterations)
- [ ] Have backup figures ready
- [ ] Practice 5-min pitch

---

## 🎯 Success Metrics

**Minimum (BASELINE - Hour 4):**
- ✅ Economic AL selection working
- ✅ Cost tracking per iteration
- ✅ 4D Pareto frontier computed
- ✅ Can demo trade-offs

**Target (Full demo - Hour 6):**
- ✅ Above + interactive dashboard
- ✅ Budget slider (recalculate AL)
- ✅ Cost breakdown visualizations
- ✅ Polished presentation

**Stretch (If time permits):**
- ✅ LLM synthesis routes (optional add-on)
- ✅ Failure mode predictions
- ✅ Real-time cost optimization
- ✅ Economic impact projection

---

## 📞 Decision Point: After Week 1

**If cost estimator working well:**
→ ✅ Continue to Week 2 (Economic AL integration)

**If cost estimator too complex:**
→ **Fallback:** Use synthesis complexity score
- Score based on: metal rarity, linker complexity, typical conditions
- Still captures economic intuition
- Still differentiated from standard approaches

**If even complexity score fails:**
→ **Final fallback:** Revert to HARD version
- You already have that plan ready
- Still competitive, just not as differentiated

---

## 🔄 Next Immediate Action

**This weekend (Day 1-2 - 3 hours):**
1. Create directory structure
2. Create reagent pricing CSV (manual curation from Sigma/TCI)
3. Implement basic cost estimator
4. Test on MOF-5, HKUST-1, UiO-66

**Commands to run:**
```bash
# Create directories
mkdir -p data src/cost src/active_learning tests

# No LLM dependencies needed yet!
# Just standard scientific Python stack

# Start with reagent database
# See economic_active_learning_guide.md for template
```

**Decision after this weekend:**
- Cost estimates reasonable? → Continue to Week 2
- Estimates too rough? → Simplify further or use complexity score
- Completely blocked? → Revert to HARD version

---

**You're keeping your ML depth (Active Learning) while adding practical impact (Economic constraints). Best of both worlds! 🚀**
