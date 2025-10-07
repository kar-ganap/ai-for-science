# Understanding the "Oracle" in Active Learning

## What is an Oracle?

In active learning, an **oracle** is an entity that provides ground truth labels/values for data points you select.

### In Real-World Materials Discovery:

**Oracle = Expensive Validation Method**

For batteries:
- **DFT Calculation:** You select a structure → Run DFT (costs 1000 CPU-hours) → Get true formation energy, band gap, etc.
- **AIMD Simulation:** Run molecular dynamics → Get true ionic conductivity
- **Experiment:** Actually synthesize the material → Measure properties in lab

For MOFs:
- **GCMC Simulation:** Select a MOF → Run grand canonical Monte Carlo (costs 1-10 CPU-hours) → Get true CO₂ uptake
- **DFT:** Calculate binding energies, partial charges
- **Experiment:** Synthesize MOF → Measure gas adsorption

**Cost:** Each oracle query costs hours to weeks (and $$ for compute/experiments)

**Active Learning Goal:** Minimize oracle queries by asking only about the most informative samples

---

## The Hackathon Problem

### Why We Can't Use Real Oracles:

❌ **DFT takes too long:** 1000 CPU-hours per structure × 50 structures per AL iteration = no way to finish in 6 hours

❌ **GCMC takes hours:** Even 1 hour per MOF × 50 MOFs = can't complete multiple AL iterations

❌ **No compute budget:** Don't know if NVIDIA will give cluster access; can't rely on it

### The Solution: Simulated Oracle with Held-Out Validation Set

---

## How the Held-Out Validation Set Works

### Step 1: Initial Data Split

Start with a dataset where you have ground truth (e.g., Materials Project with DFT energies, or CoRE MOF database with GCMC results).

```
Total Dataset: 100,000 materials with known properties
    ↓
Split into:
├─ Training Set (70%): 70,000 materials
│   └─ Use this to train initial models
│
├─ Validation Set (20%): 20,000 materials
│   └─ THIS IS YOUR "ORACLE"
│   └─ Pretend we don't know these values initially
│   └─ Active learning will "query" this set
│
└─ Test Set (10%): 10,000 materials
    └─ Final evaluation only (never touch during AL)
```

### Step 2: Simulate Oracle Queries

**Key Trick:** The validation set has ground truth values (from DFT/GCMC that was already computed), but we **pretend we don't know them** at the start.

```python
# Initial setup
train_data = dataset[:70000]
validation_pool = dataset[70000:90000]  # Oracle pool
test_data = dataset[90000:]

# Train initial model on training data ONLY
model.train(train_data)

# Active learning loop
for iteration in range(5):
    # Model makes predictions on validation pool
    predictions = model.predict(validation_pool)

    # Calculate uncertainty
    uncertainty = ensemble_disagreement(validation_pool)

    # Active learning selects most uncertain samples
    selected_indices = select_top_k_uncertain(
        predictions,
        uncertainty,
        k=50
    )

    # "ORACLE QUERY" - Look up ground truth from validation pool
    # In real world: This would trigger DFT calculation
    # In hackathon: We just look up the pre-computed value
    ground_truth = validation_pool[selected_indices].get_labels()

    # Add to training set
    train_data.append(validation_pool[selected_indices], ground_truth)

    # Remove from validation pool (can't query same sample twice)
    validation_pool.remove(selected_indices)

    # Retrain model with new data
    model.fine_tune(train_data)
```

---

## Concrete Example: Battery Materials

### Scenario: Predicting Ionic Conductivity

**Dataset:** 10,000 materials from Materials Project with DFT-computed formation energies, band gaps, etc.

**Real-world workflow (what we're simulating):**
1. Chemist wants to find high-conductivity solid electrolytes
2. Chemist has ML model trained on 1000 known materials
3. Model predicts conductivity for 100,000 hypothetical materials
4. Model is uncertain about 500 of them
5. Chemist selects 50 most uncertain high-performers
6. **Oracle query:** Run expensive AIMD simulations (costs $10,000 in compute time)
7. Get true conductivity values for those 50
8. Retrain model with new data
9. Model is now better; repeat

**Hackathon workflow (simulation):**
1. Take Materials Project data: 10,000 materials with known properties
2. Split: 7000 train, 2000 validation (oracle pool), 1000 test
3. Train model on 7000
4. Model predicts properties for validation pool (2000)
5. Model is uncertain about 500 of them
6. Active learning selects 50 most uncertain
7. **"Oracle query":** Look up ground truth from validation pool (instant!)
8. Add to training set (now have 7050 samples)
9. Retrain model
10. Repeat

---

## Why This is Valid for Demonstration

### ✅ **Preserves Active Learning Logic:**

The key question active learning answers: **"If I can only validate 50 materials, which 50 should I choose?"**

- This is the SAME whether you're looking up from a validation set or running DFT
- The algorithm doesn't know the ground truth beforehand
- Selection strategy is identical

### ✅ **Shows Real Metrics:**

You can measure:
- **Sample efficiency:** How many oracle queries needed to reach target accuracy?
- **Uncertainty reduction:** Does confidence improve over iterations?
- **Discovery rate:** What fraction of oracle queries found good materials?

### ✅ **Realistic Simulation:**

The validation set represents materials that:
- Exist in chemical space (real formulas, realistic structures)
- Have properties you'd get from real validation (DFT/GCMC/experiment)
- Are unknown to the model initially

### ✅ **Hackathon-Feasible:**

- Oracle queries are instant (just array lookups)
- Can run 5-10 AL iterations in minutes
- Can experiment with different AL strategies quickly

---

## The Key Abstraction

```
REAL WORLD:
┌─────────────────────────────────────┐
│ Unknown Material                     │
│   ↓                                  │
│ ML Model: "I predict conductivity   │
│            = 0.01 S/cm, but I'm     │
│            uncertain"                │
│   ↓                                  │
│ Oracle Query: Run AIMD ($$$$)       │
│   ↓                                  │
│ Ground Truth: 0.008 S/cm            │
│   ↓                                  │
│ Update Model with True Value        │
└─────────────────────────────────────┘

HACKATHON SIMULATION:
┌─────────────────────────────────────┐
│ Material in Validation Pool          │
│   ↓                                  │
│ ML Model: "I predict conductivity   │
│            = 0.01 S/cm, but I'm     │
│            uncertain"                │
│   ↓                                  │
│ "Oracle Query": Look up from pool   │
│   ↓                                  │
│ Ground Truth: 0.008 S/cm (from DFT  │
│               computed years ago)    │
│   ↓                                  │
│ Update Model with True Value        │
└─────────────────────────────────────┘
```

**The ML algorithm can't tell the difference!** Both cases:
- Model doesn't know ground truth initially
- Makes a prediction
- Requests validation
- Gets ground truth
- Learns from it

---

## Visual Workflow

```
TIME: Before Hackathon (Data Preparation)
═══════════════════════════════════════════════════════════
Materials Project Database
├─ 100,000 materials
├─ All have DFT-computed properties (formation energy, etc.)
└─ Downloaded and processed

Split into:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Training Set    │  │ Validation Set   │  │    Test Set      │
│  70,000 mats     │  │ 20,000 mats      │  │  10,000 mats     │
│                  │  │                  │  │                  │
│  ✅ Use labels   │  │  🔒 Hide labels  │  │  🔒 Hide labels  │
│  to train        │  │  (Oracle pool)   │  │  (Final eval)    │
└──────────────────┘  └──────────────────┘  └──────────────────┘


TIME: During Hackathon (Active Learning Loop)
═══════════════════════════════════════════════════════════

ITERATION 1:
-----------
Model trained on: 70,000 materials

  ┌─────────────────────────────────────┐
  │ Validation Pool: 20,000 materials   │
  │ (Ground truth HIDDEN)               │
  └─────────────────────────────────────┘
              ↓
  Model predicts properties + uncertainty
              ↓
  Select 50 most uncertain high-performers
              ↓
  ┌─────────────────────────────────────┐
  │ "ORACLE": Reveal ground truth for   │
  │ selected 50 materials               │
  └─────────────────────────────────────┘
              ↓
  Add 50 to training set (now: 70,050)
  Remove 50 from validation pool (now: 19,950)


ITERATION 2:
-----------
Model retrained on: 70,050 materials

  ┌─────────────────────────────────────┐
  │ Validation Pool: 19,950 materials   │
  │ (Ground truth still HIDDEN)         │
  └─────────────────────────────────────┘
              ↓
  Model predicts (should be better now!)
              ↓
  Select another 50 uncertain samples
              ↓
  Reveal ground truth → Retrain
              ↓
  Training set: 70,100
  Validation pool: 19,900

... Repeat 3-5 more iterations ...


FINAL:
------
Model trained on: 70,250 materials
Validation pool: 19,750 materials remaining
              ↓
  ┌─────────────────────────────────────┐
  │ Evaluate on Test Set: 10,000        │
  │ Compare to baseline model           │
  │ (trained only on initial 70,000)    │
  └─────────────────────────────────────┘
```

---

## What You Show at the Hackathon

### Demo Narrative:

**Opening:**
"In the real world, validating a material costs $1000 and takes a week. We simulated this using held-out data."

**Visualization 1: Oracle Budget**
```
Oracle Queries Used: ▓▓▓▓▓░░░░░ 250 / 500
Cost Equivalent: $250,000 in DFT compute
Time Equivalent: 6 months of simulations
```

**Visualization 2: Learning Curve**
```
Model Accuracy vs. Oracle Queries
     │
 90% ├─────────────────────●────●────●  Active Learning
     │                   ●●
 80% ├──────────────●●●●
     │          ●●●
 70% ├────●●●●                    Random Sampling
     │  ●●                      ●───────────────────
 60% ├●●                      ●●
     │                      ●●
 50% └─────────────────────────────────────────────
     0    50   100  150  200  250  300  350  400
              Oracle Queries (# of validations)
```

**Key Message:**
"Active learning reached 85% accuracy with 200 validations. Random sampling needed 400. We saved $200,000 in compute!"

---

## Alternative: Real Oracle (If NVIDIA Provides Resources)

If you get access to a compute cluster during the hackathon, you COULD use a real oracle:

```python
def oracle_validation_REAL(structures):
    """Real oracle using CHGNet or GCMC"""
    # Option 1: Structure relaxation with CHGNet
    relaxed_energies = []
    for structure in structures:
        relaxed = chgnet.relax(structure)
        relaxed_energies.append(relaxed.energy)

    # Option 2: GCMC for MOFs (if RASPA is set up)
    co2_uptakes = []
    for mof_structure in structures:
        uptake = raspa.run_gcmc(mof_structure, gas="CO2", pressure=0.15)
        co2_uptakes.append(uptake)

    return relaxed_energies  # or co2_uptakes
```

**This would be BETTER (more impressive)** but RISKIER:
- ✅ Shows real validation pipeline
- ⚠️ Depends on getting cluster access
- ⚠️ RASPA or CHGNet might fail/be slow

**My recommendation:** Have the validation set version working as **backup**, attempt real oracle if resources allow.

---

## Summary

**What "Held-Out Validation Set as Oracle" Means:**

1. **You have a dataset** with ground truth (Materials Project, CoRE MOF, etc.)
2. **You split it:** Train (70%), Validation/Oracle (20%), Test (10%)
3. **You hide the validation labels** initially (pretend you don't know them)
4. **Active learning selects samples** from validation pool based on uncertainty
5. **You "query the oracle"** by looking up the hidden ground truth (instant!)
6. **You retrain the model** with the newly revealed data
7. **Repeat** for multiple iterations

**Why it works:**
- Preserves AL logic (model doesn't know truth beforehand)
- Simulates expensive validation (you're budget-constrained)
- Enables rapid iteration (don't need to wait for real DFT/GCMC)
- Shows realistic metrics (sample efficiency, uncertainty reduction)

**For the hackathon:**
- Use this as your primary approach (guaranteed to work)
- If you get compute resources, switch to real oracle (CHGNet/GCMC)
- Either way, the active learning algorithm is identical!
