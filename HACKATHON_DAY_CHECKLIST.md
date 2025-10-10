# Hackathon Day Checklist

**Last Updated:** October 10, 2025
**Status:** ✅ All systems ready

---

## ⏰ 30 Minutes Before Demo

### System Verification (5 minutes)

```bash
# Run from project root
cd /path/to/ai-for-science

# Quick regeneration test (14 seconds)
python run_hackathon_demo.py --quick

# Verify figures exist
ls -la results/figures/figure*.png
```

**Expected output:**
```
✅ DEMO COMPLETE!
Total time: ~14 seconds
🎉 Ready for hackathon!
```

---

### Open These Files (5 minutes)

**In separate tabs/windows:**

1. ✅ **Figure 1** - `results/figures/figure1_ml_ablation.png`
   - 4-panel ML ablation study

2. ✅ **Figure 2** - `results/figures/figure2_dual_objectives.png`
   - 2-panel dual objectives

3. ✅ **Narrative** - `HACKATHON_NARRATIVE.md`
   - Full presentation script with talking points

4. ✅ **Metrics Card** - `HACKATHON_METRICS_CARD.md`
   - Quick reference for Q&A

5. ✅ **Troubleshooting** - `HACKATHON_TROUBLESHOOTING.md`
   - Emergency fixes (keep in background)

---

### Environment Prep (2 minutes)

- [ ] Close all unnecessary applications
- [ ] Silence phone notifications
- [ ] Disable system notifications (macOS: Do Not Disturb)
- [ ] Test screen sharing (if virtual)
- [ ] Have water nearby
- [ ] Set browser zoom to 125-150% for visibility

---

### Mental Prep (3 minutes)

Read these once:

**One-sentence summary:**
> "We developed the first budget-constrained active learning framework for materials discovery, achieving 25.1% better predictions with 62% sample efficiency."

**Three key numbers:**
- **25.1%** - Uncertainty reduction
- **62%** - Sample efficiency improvement
- **100%** - Budget compliance

**Core message:**
> "Objective alignment matters: optimize for what you actually want (learning vs discovery), and you get 62% efficiency gains."

---

## 📋 Demo Execution Plan

### Strategy: Static Figures (RECOMMENDED)

**Time:** 5 minutes
**Risk:** ⭐ (lowest)
**Setup:** Open Figure 1 and Figure 2 in Preview/browser

**Flow:**
1. Hook (30s): "Materials discovery costs money..."
2. Figure 1 walkthrough (2 min): Panels A, B, C, D
3. Figure 2 walkthrough (1.5 min): Discovery vs Learning
4. Key insight (30s): "Objective alignment = 62% efficiency"
5. Q&A (remaining time)

**Advantage:** Zero risk, focuses on science not code

---

### Backup: Quick Regeneration (if asked)

**Time:** 5 min + 15 seconds setup
**Command:** `python run_hackathon_demo.py --quick`

Only use if someone explicitly asks "can you show it works?"

---

## 🎯 Presentation Checklist

### Opening (30 seconds)

- [ ] State the problem clearly
- [ ] Use the hook: "Like searching for a needle... each piece of hay costs $1"
- [ ] Set expectations: "I'll show 2 figures and key insights"

---

### Figure 1 Walkthrough (2 minutes)

**Panel A:**
- [ ] Point out: Exploration (25.1%), Exploitation (8.6%), Random (-1.4%)
- [ ] Emphasize: "Random makes model WORSE"

**Panel B:**
- [ ] Show: AL curves go down, Random stays flat
- [ ] Say: "AL systematically learns, Random doesn't"

**Panel C:**
- [ ] Point to budget line at $50
- [ ] Say: "100% budget compliance - constraint optimization works"

**Panel D:**
- [ ] Highlight: Different cost-efficiency trade-offs
- [ ] Say: "Pareto frontier - choose your strategy based on objective"

---

### Figure 2 Walkthrough (1.5 minutes)

**Left Panel:**
- [ ] Point out: All methods find similar performance
- [ ] Emphasize: **"But exploitation uses 62% fewer samples!"**
- [ ] Mention: Random got lucky (high variance)

**Right Panel:**
- [ ] Contrast: Random (-1.4%) vs AL (+25.1%)
- [ ] Say: "Random wins on discovery by luck, fails on learning"
- [ ] Core message: **"Objective alignment matters"**

---

### Closing (30 seconds)

- [ ] Restate core innovation: "First budget-constrained AL for materials"
- [ ] State impact: "62% sample efficiency, 25.1% better predictions"
- [ ] Open for questions: "Happy to dive deeper into any aspect"

---

## 💬 Q&A Preparation

### Expected Questions & Responses

**Q: "Why only 687 MOFs?"**
> ✅ "Quality over quantity - all experimental with CO₂ labels. Enables real cost estimation. Hypothetical datasets lack synthesis cost data."

**Q: "How accurate are costs?"**
> ✅ "Validated on MOF-5 ($1.12/g), HKUST-1 ($1.51/g). Reagent cost is 60-80% of total."

**Q: "Can Random really make models worse?"**
> ✅ "Yes! Random samples don't target epistemic uncertainty—they add noise. That's why -1.4%."

**Q: "What about neural networks?"**
> ✅ "Small dataset (687). RandomForest better for uncertainty with ensembles at this scale."

**Q: "Can you show the code?"**
> ✅ "Absolutely!" → Open `src/active_learning/economic_al.py`

---

### Difficult Questions & Responses

**Q: "Random found better materials (10.28 vs 9.18)?"**
> ✅ "Good catch! Random got lucky (±1.08 high variance). We ran 20 trials - sometimes wins, sometimes loses. AL is consistent. Plus, Random fails on learning (-1.4%)."

**Q: "What about synthesizability?"**
> ✅ "All CRAFTED MOFs are already synthesized (=1.0). Meaningful for hypothetical datasets - future work."

**Q: "Isn't this just standard AL with cost?"**
> ✅ "No - we optimize BOTH validation budget (discovery) AND synthesis cost (production). That dual-cost framework is novel."

---

## 🚨 Emergency Protocols

### If Code Breaks

1. **Stay calm:** "Let me show pre-generated results - safer for demo!"
2. **Switch to static figures**
3. **Walk through HACKATHON_NARRATIVE.md**
4. **Focus on science, not code**

---

### If Running Behind on Time

**Cut these:**
- ❌ Panel D (sample efficiency)
- ❌ Technical implementation details
- ❌ VAE discussion

**Keep these:**
- ✅ Problem statement
- ✅ Panel B (learning dynamics)
- ✅ Figure 2 left (62% efficiency)
- ✅ Core insight (objective alignment)

**Minimum viable (3 minutes):**
1. Hook: "Materials discovery costs money"
2. Show Panel B: "AL learns (25.1%), Random doesn't (-1.4%)"
3. Show Figure 2 Left: "Same result, 62% fewer samples"
4. Close: "Economic AL enables budget-constrained discovery"

---

## ✅ Final Checklist

### Before Demo Starts:

- [ ] `python run_hackathon_demo.py --quick` completed successfully
- [ ] Figure 1 open in Preview
- [ ] Figure 2 open in Preview
- [ ] HACKATHON_NARRATIVE.md open in browser
- [ ] HACKATHON_METRICS_CARD.md open in browser
- [ ] Phone silenced
- [ ] System notifications disabled
- [ ] Water nearby
- [ ] Deep breath taken 😊

### During Demo:

- [ ] Start with static figures (safest)
- [ ] Keep to 5 minutes max
- [ ] Save 2+ minutes for Q&A
- [ ] Reference metrics card for numbers
- [ ] Use troubleshooting guide if needed

### After Demo:

- [ ] Offer to share code/repo
- [ ] Exchange contact info with interested people
- [ ] Note any good questions for future work

---

## 📊 Success Criteria

**Audience should leave knowing:**

1. ✅ **Problem:** Materials discovery is expensive, budgets are real
2. ✅ **Solution:** Economic AL with budget constraints
3. ✅ **Evidence:** 25.1% uncertainty reduction, 62% sample efficiency, 100% budget compliance
4. ✅ **Insight:** Objective alignment matters (exploration vs exploitation)
5. ✅ **Innovation:** First budget-constrained AL for materials discovery

**Everything else is bonus!**

---

## 🎯 Confidence Check

Rate yourself on these (should all be YES):

- [ ] I can explain the problem in 30 seconds
- [ ] I can walk through Figure 1 without notes
- [ ] I can walk through Figure 2 without notes
- [ ] I can answer "Why only 687 MOFs?"
- [ ] I can answer "How accurate are costs?"
- [ ] I can answer "Can Random make models worse?"
- [ ] I know where all files are located
- [ ] I've practiced the 5-minute flow at least once
- [ ] I can recover if code breaks
- [ ] I know the 3 key numbers (25.1%, 62%, 100%)

**If all checked: You're ready!** 🚀

---

## 📞 Quick Links

**Key Documents:**
- Narrative: `HACKATHON_NARRATIVE.md`
- Metrics: `HACKATHON_METRICS_CARD.md`
- Troubleshooting: `HACKATHON_TROUBLESHOOTING.md`
- Project Status: `docs/PROJECT_STATUS_HOLISTIC.md`

**Key Figures:**
- Figure 1: `results/figures/figure1_ml_ablation.png`
- Figure 2: `results/figures/figure2_dual_objectives.png`

**Quick Commands:**
```bash
# Regenerate everything (14 seconds)
python run_hackathon_demo.py --quick

# Verify files
ls -la results/figures/figure*.png
```

---

**Remember:** You've overdelivered on the plan. You have strong evidence (25.1%, 62%, 100%). You have a clear narrative. You're ready to win. 🏆

**Final advice:** Focus on the story, not the code. Pre-generated figures are your friend. You've got this! 💪
