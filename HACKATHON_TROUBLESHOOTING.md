# Hackathon Demo Troubleshooting Guide

**Quick Reference for Common Issues During Presentation**

---

## Pre-Demo Checklist ✅

Run before the hackathon starts:

```bash
# 1. Verify all figures exist
ls -la results/figures/*.png

# 2. Quick regeneration (30 seconds)
python run_hackathon_demo.py --quick

# 3. Check key results files
ls -la results/*.csv
```

Expected files:
- `results/figures/figure1_ml_ablation.png` ✅
- `results/figures/figure2_dual_objectives.png` ✅
- `results/economic_al_crafted_integration.csv` ✅
- `results/economic_al_expected_value.csv` ✅

---

## Common Issues & Instant Fixes

### Issue 1: Figure Not Displaying

**Symptom:** Figure file won't open or appears blank

**Quick Fix:**
```bash
# Regenerate just that figure
uv run python src/visualization/figure1_ml_ablation.py
# or
uv run python src/visualization/figure2_dual_objectives.py
```

**Backup Plan:** Use the figures already in `results/figures/` - they're pre-generated!

---

### Issue 2: Font/Character Warnings

**Symptom:** Warnings about missing glyphs or subscript characters (CO₂)

**Impact:** ⚠️ Cosmetic only - figures still render correctly

**What to say:** "These are font warnings - the subscripts still render correctly as you can see"

**If subscripts don't render:**
- Open figure files in Preview/browser - they render fine when saved as PNG

---

### Issue 3: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Quick Fix:**
```bash
# Ensure you're in project root
cd /path/to/ai-for-science

# Try running with uv
uv run python tests/test_economic_al_crafted.py
```

---

### Issue 4: Missing Data File

**Symptom:** `❌ Missing data file: data/processed/crafted_mofs_co2.csv`

**Quick Fix:**
```bash
# Check if file exists
ls -la data/processed/

# If missing, re-run preprocessing
uv run python src/preprocessing/load_crafted.py
```

---

### Issue 5: Code Crashes During Live Demo

**Backup Plan:**

✅ **Option 1: Use Pre-Generated Figures**
- All figures already exist in `results/figures/`
- Just walk through the static figures - safer than live code!

✅ **Option 2: Show Pre-Computed Metrics**
```python
# Quick metrics summary (no code execution needed)
print("Key Results:")
print("  • Uncertainty reduction: 25.1% (0.122 → 0.091)")
print("  • Sample efficiency: 62% fewer samples (72 vs 188)")
print("  • Budget compliance: 100% (all 3 iterations under $50)")
print("  • Total cost: $148.99")
```

✅ **Option 3: Refer to Documentation**
- Open `HACKATHON_NARRATIVE.md`
- Walk through the narrative without live code

---

## Demo Execution Strategies

### Strategy 1: Static Figures Only (SAFEST)

**Time:** 5 minutes
**Risk:** ⭐ (lowest)

1. Open pre-generated figures in Preview/image viewer
2. Walk through Figure 1 and Figure 2
3. Reference metrics from `HACKATHON_NARRATIVE.md`
4. No live code execution

**Advantage:** Zero risk of code failure

---

### Strategy 2: Quick Regeneration (RECOMMENDED)

**Time:** 5 minutes + 30 seconds setup
**Risk:** ⭐⭐ (low)

```bash
# Before demo starts
python run_hackathon_demo.py --quick
```

1. Show command output (proves it works)
2. Open newly generated figures
3. Walk through narrative

**Advantage:** Shows code works without re-running full pipeline

---

### Strategy 3: Full Pipeline (HIGH RISK)

**Time:** 5 minutes + 3 minutes execution
**Risk:** ⭐⭐⭐⭐ (high)

```bash
python run_hackathon_demo.py
```

**Only do this if:**
- You have extra time
- Someone explicitly asks "can you run it live?"
- You're confident in network/compute

**Risk:** Pipeline takes 2-3 minutes, could fail mid-demo

---

## Q&A Quick Answers

### Q: "Can you show me the code?"

**Answer:**
```bash
# Open main Economic AL class
open src/active_learning/economic_al.py

# Highlight key functions:
# - run_iteration() (line ~120)
# - compute_acquisition_scores() (line ~180)
# - select_batch_within_budget() (line ~220)
```

**Backup:** Refer to `docs/economic_active_learning_guide.md`

---

### Q: "How accurate are your cost estimates?"

**Answer:**
```bash
# Show validation
uv run python tests/test_cost_estimator.py

# Key results to quote:
# MOF-5: $1.12/g (validated against literature)
# HKUST-1: $1.51/g
# UiO-66: $1.57/g
```

**Backup:** Refer to `docs/cost_framework_explained.md`

---

### Q: "What if Random baseline changes with different seed?"

**Answer:**
"Great question! That's exactly why we ran 20 independent trials and computed statistics. The 10.28 ± 1.08 mol/kg shows Random's high variance - sometimes lucky, sometimes not. AL is consistent."

**Show:** Figure 2, right panel - Random's error bars are huge

---

### Q: "Can you run this on different data?"

**Answer:**
"Absolutely! The framework is dataset-agnostic. We use CRAFTED because it has experimental CO₂ labels and real synthesis costs. You could apply this to any materials dataset with:
1. Feature vectors (geometric, compositional, etc.)
2. Target property labels
3. Cost estimation (or actual costs)"

---

## Nuclear Option: Demo Failure Recovery

**If everything breaks:**

1. **Stay calm** - this is research code, failures happen

2. **Switch to static demo:**
   ```
   "Let me show you the results we generated earlier -
    this is safer than running code live during a demo!"
   ```

3. **Open pre-generated figures** in Preview/browser

4. **Walk through HACKATHON_NARRATIVE.md** section by section

5. **Emphasize the science, not the code:**
   - "The key insight is objective alignment..."
   - "As you can see in Figure 1, Panel B..."
   - "This demonstrates that acquisition functions matter..."

6. **Offer to share code afterwards:**
   ```
   "I'd be happy to walk through the code implementation
    after the presentation if you're interested!"
   ```

---

## Environment Quick Checks

### Before Demo:

```bash
# 1. Verify Python/uv working
uv run python --version

# 2. Check key imports
uv run python -c "import pandas; import sklearn; import matplotlib; print('✅ All imports OK')"

# 3. Verify project structure
ls -la src/active_learning/economic_al.py
ls -la results/figures/

# 4. Check git status (optional)
git status
git log -1 --oneline
```

---

## Backup Materials Checklist

✅ **Pre-generated figures** (do NOT delete these!)
- `results/figures/*.png`

✅ **Pre-computed results** (do NOT delete these!)
- `results/*.csv`

✅ **Documentation** (always available offline)
- `HACKATHON_NARRATIVE.md`
- `docs/PROJECT_STATUS_HOLISTIC.md`
- `README.md`

✅ **This troubleshooting guide**
- Keep open in a separate terminal during demo

---

## Time Management

### If Running Behind:

**Cut these parts:**
1. ❌ Skip VAE discussion (only if asked)
2. ❌ Skip supplementary plots walkthrough
3. ❌ Skip technical implementation details

**Keep these parts:**
1. ✅ Problem statement (30 seconds)
2. ✅ Figure 1 walkthrough (90 seconds)
3. ✅ Figure 2 walkthrough (60 seconds)
4. ✅ Key insight: 62% sample efficiency (15 seconds)
5. ✅ Q&A (remaining time)

**Minimum viable demo:** 3 minutes
- Hook: "Materials discovery costs money, we optimize it"
- Figure 1 Panel B: "AL learns (25.1%), Random doesn't (-1.4%)"
- Figure 2 Left: "Same outcome, 62% fewer samples"
- Close: "Economic AL enables budget-constrained discovery"

---

## Final Checklist (Print This!)

**30 minutes before demo:**
- [ ] Run `python run_hackathon_demo.py --quick`
- [ ] Open `HACKATHON_NARRATIVE.md` in browser
- [ ] Open `results/figures/figure1_ml_ablation.png`
- [ ] Open `results/figures/figure2_dual_objectives.png`
- [ ] Open this troubleshooting guide
- [ ] Close all other applications
- [ ] Silence phone/notifications
- [ ] Test screen sharing (if virtual)

**During demo:**
- [ ] Start with static figures (safest)
- [ ] Reference narrative document for talking points
- [ ] Have this troubleshooting guide open in background
- [ ] Keep demo to 5 minutes max
- [ ] Save 2+ minutes for Q&A

**If asked to run code live:**
- [ ] Use `python run_hackathon_demo.py --quick` (30s)
- [ ] NOT the full pipeline (3 min risk)

---

## Contact Info for Help

**If demo computer fails completely:**
- Have figures backed up on phone/cloud
- Can show figures from phone if needed
- Narrative document also available on phone

**Repository backup:**
- GitHub: [your-username]/ai-for-science
- All figures and docs available in repo

---

## Success Metrics (What Matters)

✅ **Audience understands the problem:** Materials discovery is expensive
✅ **Audience understands the solution:** Economic AL with budget constraints
✅ **Audience sees the evidence:** 25.1% uncertainty reduction, 62% sample efficiency
✅ **Audience gets the insight:** Objective alignment matters (exploration vs exploitation)

Everything else is secondary! Focus on the story, not the code.

---

**Remember:** Pre-generated figures exist for a reason. Use them! 🎯
