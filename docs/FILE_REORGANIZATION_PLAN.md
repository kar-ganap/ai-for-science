# File Reorganization Plan & Risk Analysis

## 🚨 Risk Assessment

### **HIGH RISK - Will Break if Not Fixed**

#### 1. **Streamlit App Subprocess Call** (CRITICAL)
**File**: `streamlit_app.py` (line 315)
```python
subprocess.run([sys.executable, "demo_active_generative_discovery.py"], ...)
```
**Risk**: ❌ Will fail when regenerating Figure 2 data
**Fix Required**: Update to `scripts/demos/demo_active_generative_discovery.py`

#### 2. **Cross-Script Import** (CRITICAL)
**File**: `generate_pseudo_labels.py` (line 21)
```python
from train_gnn_surrogate import MOF_GNN
```
**Risk**: ❌ Import will fail after move
**Fix Required**:
- Option A: Move both to same directory
- Option B: Update import to use relative path
- Option C: Move MOF_GNN to `src/` and import from there

### **MEDIUM RISK - Should Update**

#### 3. **Documentation References** (MEDIUM)
**Files**: Multiple markdown files reference script paths
- `HACKATHON_QUICK_REFERENCE.md`
- `PRE_HACKATHON_CHECKLIST.md`
- `docs/REGENERATION_GUIDE.md`
- Multiple session summaries

**Risk**: ⚠️ User confusion, but won't break code
**Fix Required**: Update documentation with new paths

### **LOW RISK - No Action Needed**

#### 4. **Test Imports** (LOW)
**File**: `run_hackathon_demo.py` imports from `tests/`
```python
from tests.test_economic_al_crafted import test_economic_al_crafted_integration
```
**Risk**: ✅ Safe - tests directory not changing
**Fix Required**: None

---

## 🛠️ Refactoring Strategy

### Phase 1: Prepare (No File Moves Yet)

1. **Create directory structure**
   - `scripts/demos/`
   - `scripts/analysis/`
   - `scripts/data_generation/`
   - `scripts/training/`
   - `scripts/data_preprocessing/`
   - `tests/exploratory/`
   - `docs/archive/`
   - `docs/guides/`
   - `docs/hackathon/`
   - `docs/technical/`

2. **Fix imports BEFORE moving**
   - Fix `generate_pseudo_labels.py` import issue
   - Fix `streamlit_app.py` subprocess calls

### Phase 2: Move Files

3. **Move files in batches**
   - Start with docs (safest)
   - Then test scripts
   - Then analysis/data_generation scripts
   - Finally demo scripts (most risky)

4. **Test after each batch**

### Phase 3: Validate

5. **Run comprehensive tests**
   - Test streamlit app regeneration
   - Test all demos
   - Verify imports work

---

## 📋 Detailed Refactoring Plan

### Step 1: Fix Streamlit App Subprocess Calls

**File**: `streamlit_app.py`

**Changes needed**:
```python
# Line 315: OLD
subprocess.run([sys.executable, "demo_active_generative_discovery.py"], ...)

# Line 315: NEW
subprocess.run([sys.executable, "scripts/demos/demo_active_generative_discovery.py"], ...)
```

### Step 2: Fix generate_pseudo_labels.py Import

**Option A: Move both to same directory** (RECOMMENDED)
```
scripts/training/
  ├── train_gnn_surrogate.py
  └── (related training scripts)

scripts/data_generation/
  ├── generate_pseudo_labels.py
  └── (other generation scripts)
```
Then update import:
```python
# In generate_pseudo_labels.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
from train_gnn_surrogate import MOF_GNN
```

**Option B: Move MOF_GNN to src/** (CLEANER)
```python
# Move class to src/models/gnn_surrogate.py
# Then update imports in both files:
from src.models.gnn_surrogate import MOF_GNN
```

### Step 3: File Move Mapping

#### Documentation Files
```
ROOT                                    → docs/archive/
  SESSION_SUMMARY_*.md                  → docs/archive/
  FINAL_STATUS.md                       → docs/archive/
  LYNCHPIN_ANALYSIS_SUMMARY.md         → docs/archive/
  NEXT_STEPS.md                         → docs/archive/
  ACTIVE_GENERATIVE_DISCOVERY_SUMMARY.md → docs/archive/

ROOT                                    → docs/guides/
  QUICKSTART.md                         → docs/guides/
  DEMO.md                               → docs/guides/
  STREAMLIT_DEMO.md                     → docs/guides/
  GNN_IMPLEMENTATION_GUIDE.md          → docs/guides/

ROOT                                    → docs/hackathon/
  HACKATHON_*.md                        → docs/hackathon/
  PRE_HACKATHON_CHECKLIST.md           → docs/hackathon/

ROOT                                    → docs/technical/
  EXPLORATION_BONUS_*.md                → docs/technical/
  SURROGATE_IMPROVEMENT_OPTIONS.md     → docs/technical/
```

#### Python Scripts
```
ROOT                                    → scripts/demos/
  demo_active_generative_discovery.py   → scripts/demos/
  demo_exploration_bonus_strategy.py    → scripts/demos/
  demo_with_portfolio_constraint.py     → scripts/demos/
  run_hackathon_demo.py                 → scripts/demos/

ROOT                                    → scripts/analysis/
  analyze_budget_tradeoffs.py           → scripts/analysis/
  analyze_exploration_bonus.py          → scripts/analysis/
  theoretical_bonus_justification.py    → scripts/analysis/
  final_surrogate_diagnosis.py          → scripts/analysis/

ROOT                                    → scripts/data_generation/
  generate_baseline_for_figure2.py      → scripts/data_generation/
  generate_baseline_with_actual_co2.py  → scripts/data_generation/
  extract_baseline_progression.py       → scripts/data_generation/
  generate_pseudo_labels.py             → scripts/data_generation/
  create_linker_assignments.py          → scripts/data_generation/

ROOT                                    → scripts/training/
  train_compositional_vae.py            → scripts/training/
  train_dual_cvae_production.py         → scripts/training/
  train_dual_cvae_variants.py           → scripts/training/
  train_gnn_surrogate.py                → scripts/training/
  comprehensive_vae_sweep.py            → scripts/training/

ROOT                                    → scripts/data_preprocessing/
  build_gnn_dataset.py                  → scripts/data_preprocessing/
  build_gnn_dataset_fast.py             → scripts/data_preprocessing/

ROOT                                    → tests/exploratory/
  test_cif_parsing.py                   → tests/exploratory/
  test_gaussian_process_surrogate.py    → tests/exploratory/
  test_gp_chemistry.py                  → tests/exploratory/
  test_hybrid_surrogate.py              → tests/exploratory/
  test_surrogate_generalization.py      → tests/exploratory/
  test_surrogate_with_geom_features.py  → tests/exploratory/
```

---

## ✅ Validation Checklist

After reorganization, verify:

- [ ] **Streamlit app launches**: `uv run streamlit run streamlit_app.py`
- [ ] **Figure 1 regeneration works**: Test "Just Update Figures"
- [ ] **Figure 2 regeneration works**: Test full AGD regeneration
- [ ] **Demo scripts run**: `python scripts/demos/demo_active_generative_discovery.py`
- [ ] **Training scripts accessible**: Can import from scripts/training
- [ ] **No import errors**: Run Python `-c "import sys; import streamlit_app"`
- [ ] **Documentation accurate**: Spot-check key documentation files

---

## 🎯 Execution Order (Safe)

### Batch 1: Documentation (Safest)
1. Create `docs/` subdirectories
2. Move all `.md` files (except README.md)
3. ✅ **Test**: Nothing should break (docs are passive)

### Batch 2: Test Scripts
1. Create `tests/exploratory/`
2. Move test_*.py files
3. ✅ **Test**: Run existing tests in tests/ to ensure nothing broke

### Batch 3: Training & Data Scripts
1. Create `scripts/training/`, `scripts/data_preprocessing/`, `scripts/data_generation/`
2. Fix `generate_pseudo_labels.py` import
3. Move files
4. ✅ **Test**: Try importing from new locations

### Batch 4: Analysis Scripts
1. Create `scripts/analysis/`
2. Move analysis files
3. ✅ **Test**: Run one analysis script

### Batch 5: Demo Scripts (Most Risky)
1. Create `scripts/demos/`
2. **FIX streamlit_app.py subprocess call FIRST**
3. Move demo files
4. ✅ **Test**: Streamlit regeneration feature

---

## 🔄 Rollback Plan

If something breaks:
1. Git has all changes tracked
2. Can revert with: `git checkout -- <file>`
3. Or: `git reset --hard HEAD` (if committed)

**Recommendation**: Commit after each successful batch

---

## 📊 Impact Summary

| Component | Risk | Mitigation | Test Required |
|-----------|------|------------|---------------|
| Streamlit App | 🔴 HIGH | Update subprocess calls | Full regeneration test |
| generate_pseudo_labels.py | 🔴 HIGH | Fix import path | Run script |
| Documentation | 🟡 MEDIUM | Update paths | Visual review |
| Demo Scripts | 🟢 LOW | Move only | Run one demo |
| Training Scripts | 🟢 LOW | Move only | Import test |
| Analysis Scripts | 🟢 LOW | Move only | None needed |
| Test Scripts | 🟢 LOW | Move only | None needed |

---

## 🎯 Recommendation

**Proceed with reorganization** using the phased approach. The risks are:
- **Manageable**: Only 2 critical fixes needed
- **Testable**: Can verify at each step
- **Reversible**: Git tracking allows rollback
- **Valuable**: Much cleaner project structure

**Estimated Time**:
- Preparation & fixes: 15 minutes
- Moving files: 20 minutes
- Testing: 15 minutes
- **Total**: ~50 minutes

**Success Criteria**:
- ✅ Streamlit app launches and regenerates data
- ✅ All demos run from new locations
- ✅ No import errors
- ✅ Project is cleaner and more maintainable
