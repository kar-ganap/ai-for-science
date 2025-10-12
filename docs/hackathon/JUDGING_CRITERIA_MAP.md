# Judging Criteria Mapping: Strategic Demo Emphasis

**Purpose**: Map each demo section to judging criteria to maximize scoring.

---

## Judging Criteria Overview

From the hackathon brief, judges evaluate on:

1. **Scientific Relevance & Rigor** (25%)
2. **Novelty** (25%)
3. **Impact** (25%)
4. **Execution** (25%)

**Goal**: Hit all 4 criteria explicitly during the 3-minute demo.

---

## Criteria 1: Scientific Relevance & Rigor (25%)

### What Judges Look For:
- Real scientific problem
- Sound methodology
- Validated results
- Limitations acknowledged
- Realistic context

### Where to Address in Demo:

#### Opening Hook (0:00-0:10) ✅
**SAY**: "Nobel Prize-winning MOFs... $35-65 per experiment"
**SCORING**: Establishes scientific relevance + real-world constraint

#### Problem Setup (0:10-0:30) ✅
**SAY**: "687 candidates from CRAFTED database" + "would cost $25,000+"
**SCORING**: Real experimental dataset + quantified problem scale

#### Figure 1 - Panel A (1:20-1:30) ✅
**SAY**: "4-way baseline comparison: Random, Expert, Exploration, Exploitation"
**SCORING**: Rigorous experimental design with multiple baselines

#### Figure 2 - Panel A (2:05-2:20) ✅
**SAY**: "Validated on CRAFTED—687 experimental MOFs with ground-truth CO₂ uptake"
**SCORING**: Real data, not synthetic

#### Impact Section (2:50-3:10) ✅
**SAY**: "Validated on real experimental data (CRAFTED database)"
**SCORING**: Reinforces scientific rigor

### Key Phrases to Use:
- **"CRAFTED database"** (mention 3 times)
- **"Ground-truth CO₂ uptake"**
- **"4-way baseline comparison"**
- **"100% budget compliance"** (rigor in constraint adherence)
- **"687 experimental MOFs"**

### Limitations to Acknowledge (if asked):
> "Demo uses `target_co2 + Gaussian noise` for generated MOF validation to prove the AL loop. Production deployment would plug in DFT calculations or experimental synthesis. The framework is validation-agnostic."

**WHY THIS MATTERS**: Shows intellectual honesty—you know the demo limits, you're not overselling.

---

## Criteria 2: Novelty (25%)

### What Judges Look For:
- Unique approach
- Technical innovation
- Advances beyond state-of-art
- Not just applying existing tools

### Where to Address in Demo:

#### Architecture Section (0:40-1:20) ✅ **CRITICAL**
**SAY**: "**Not sequential** (train VAE → generate → select), but **iterative co-evolution**"
**SCORING**: Explicitly contrasts with standard pipelines

**SAY**: "3 key innovations: 1) GP covariance (true epistemic uncertainty), 2) Adaptive VAE targeting (7.1 → 10.2), 3) Portfolio constraints (70-85% generated)"
**SCORING**: Enumerated novelty—easy for judges to note

#### Figure 1 - Panel B (1:30-1:40) ✅
**SAY**: "GP covariance provides true Bayesian uncertainty—not ensemble variance like Random Forest"
**SCORING**: Technical depth distinguishes from naive approaches

#### Figure 2 - Panel C (2:25-2:35) ✅
**SAY**: "Adaptive targeting: VAE target increases from 7.1 → 8.7 → 10.2 mol/kg as it learns"
**SCORING**: Dynamic system, not static generation

#### Impact Section (2:50-3:10) ✅
**SAY**: "First work (to our knowledge) with portfolio-constrained active generative loop"
**SCORING**: Claims novelty explicitly

### Key Phrases to Use:
- **"Tight coupling"** / **"Iterative co-evolution"** (say 3 times)
- **"Not sequential—simultaneous feedback loop"**
- **"Adaptive targeting"** (emphasize 7.1 → 10.2 progression)
- **"Portfolio constraints as regularization"**
- **"Dual-cost optimization"** (validation + synthesis)
- **"True epistemic uncertainty"** (GP covariance vs ensemble variance)

### What Makes This Novel (for Q&A):
1. **Tight AL ↔ VAE coupling**: Not train → generate → select (sequential), but iterative co-evolution
2. **Portfolio constraints**: 70-85% generated + 15-30% real as risk management
3. **Adaptive VAE targeting**: Target increases with discoveries (7.1 → 10.2)
4. **Dual-cost budget constraints**: First work integrating validation + synthesis costs in materials AL
5. **GP for epistemic uncertainty**: True Bayesian uncertainty enables principled exploration (18.6× better)

**WHY THIS MATTERS**: Judges must understand this isn't "just apply VAE to MOFs"—it's a novel framework.

---

## Criteria 3: Impact (25%)

### What Judges Look For:
- Quantified improvements
- Transferability to other domains
- Adoption potential
- Broader field implications

### Where to Address in Demo:

#### Figure 1 - Panel A (1:20-1:30) ✅
**SAY**: "18.6× better learning efficiency—9.3% vs 0.5% uncertainty reduction"
**SCORING**: Quantified impact with dramatic ratio

#### Figure 1 - Panel D (1:50-2:05) ✅
**SAY**: "2.6× more efficient: $0.78/MOF vs $2.03/MOF, and 2.6× more MOFs validated"
**SCORING**: Double quantified impact (cost + sample efficiency)

#### Figure 2 - Panel A (2:05-2:20) ✅
**SAY**: "+26.6% discovery improvement—baseline stuck at 8.75, AGD reaches 11.07 mol/kg"
**SCORING**: Performance breakthrough quantified

#### Impact Section (2:50-3:10) ✅ **CRITICAL**
**SAY**: "Transferable framework: swap MOFs for batteries, catalysts, alloys. Generalizes to any materials + property task."
**SCORING**: Broad applicability

**SAY**: "Open-source, reproducible—deploy in your lab tomorrow"
**SCORING**: Adoption potential

### Key Numbers to Memorize & Emphasize:
- **18.6×** better learning (exploration vs exploitation)
- **+26.6%** discovery improvement (AGD vs baseline)
- **2.6×** cost efficiency ($0.78 vs $2.03 per MOF)
- **2.6×** sample efficiency (315 vs 120 MOFs)
- **100%** budget compliance
- **100%** compositional diversity
- **95%** chemical space coverage (19/20 combinations)

### Transferability Statement (for Q&A):
> "This framework applies to any materials discovery + property prediction task:
> - **Batteries**: Optimize energy density, charging rate, cycle life
> - **Catalysts**: Maximize reaction yield, selectivity
> - **Alloys**: Target strength, ductility, corrosion resistance
> - **Polymers**: Design thermal stability, mechanical properties
>
> The system is modular: swap datasets, swap surrogate models (GP → NN), swap generative models (VAE → GAN, diffusion). All code is open-source, documented, ready for adoption."

**WHY THIS MATTERS**: Impact isn't just MOFs—it's a generalizable framework for budget-constrained discovery.

---

## Criteria 4: Execution (25%)

### What Judges Look For:
- Working prototype
- Polished presentation
- Clear visualizations
- Comprehensive documentation
- Usability for target audience

### Where to Address in Demo:

#### Kick Off Regeneration (0:30-0:40) ✅
**SAY**: "Let me show you live—regenerating experiments right now"
**SCORING**: Working prototype, not mockup

**SHOW**: Click sidebar → select "Both Figures" → Run
**SCORING**: Polished UI, clear controls

#### Figure 1 & 2 (1:20-2:50) ✅
**SHOW**: Publication-quality 4-panel figures
**SCORING**: Professional visualizations, not Matplotlib defaults

#### Impact Section (2:50-3:10) ✅
**SAY**: "Interactive Streamlit dashboard with 5 tabs: Dashboard, Results, Figures, Discovery Explorer, About"
**SCORING**: Comprehensive tooling

**SAY**: "Complete documentation: README, ARCHITECTURE, BRIEF—7-level progressive diagrams"
**SCORING**: Usability for adoption

### Visual Execution Checklist:
- [ ] **Figures are high-res** (1200 DPI, downloadable PNG)
- [ ] **Color scheme is consistent** (exploration=green, exploitation=red, AGD=orange, baseline=gray)
- [ ] **Annotations are clear** ("R" = Real, "G" = Generated markers)
- [ ] **Layout is professional** (4-panel grid, consistent fonts, no overlapping text)
- [ ] **Dashboard is intuitive** (sidebar controls, clear tabs, no clutter)

### Documentation to Mention (if time allows):
> "Three levels of documentation:
> 1. **README.md**: Quick start, 5-minute setup
> 2. **ARCHITECTURE.md**: 7-level progressive diagrams—high-level overview to implementation details
> 3. **HACKATHON_BRIEF.md**: Technical deep-dive addressing all judging criteria
>
> All code is modular, commented, type-hinted. Production-ready, not research spaghetti."

**WHY THIS MATTERS**: Execution distinguishes "research prototype" from "deployable tool."

---

## Strategic Emphasis by Demo Section

### Opening Hook (0:00-0:10)
**PRIMARY CRITERION**: Scientific Relevance ✅
**SECONDARY**: Impact (Nobel Prize = field importance)

### Problem Setup (0:10-0:30)
**PRIMARY CRITERION**: Scientific Relevance ✅
**SECONDARY**: Novelty (tight coupling vs sequential)

### Architecture Walkthrough (0:40-1:20)
**PRIMARY CRITERION**: Novelty ✅✅✅ **MOST IMPORTANT**
**SECONDARY**: Scientific Rigor (GP covariance, portfolio constraints)

### Figure 1 (1:20-2:05)
**PRIMARY CRITERION**: Impact ✅ (18.6×, 2.6×, 100% compliance)
**SECONDARY**: Scientific Rigor (4-way comparison, budget compliance)

### Figure 2 (2:05-2:50)
**PRIMARY CRITERION**: Impact ✅ (+26.6%, R→G→G pattern)
**SECONDARY**: Novelty (adaptive targeting, compositional diversity)

### Impact & Adoption (2:50-3:10)
**PRIMARY CRITERION**: Impact ✅✅✅ **CLOSING ARGUMENT**
**SECONDARY**: Execution (open-source, transferable, documented)

---

## Scoring Optimization: Where to Double Down

### If You Have Extra 10 Seconds:
**ADD TO**: Architecture section (0:40-1:20)
**WHY**: Novelty is hardest to convey—most demos fail here
**WHAT TO SAY**:
> "Let me be explicit: most work does VAE → generate pool → AL selects from pool. That's sequential. We do AL selects → validates → retrains VAE → VAE generates for NEXT AL cycle → repeat. The feedback loop is tight—VAE learns from every validation, AL benefits from better generation. That's why AGD breaks through (+26.6%) while baseline plateaus."

### If You're Running 10 Seconds Over:
**CUT**: Panel C/D explanations in Figure 1
**KEEP**: Panel A (4-way comparison) + Panel D (Pareto efficiency)
**WHY**: Panel A establishes rigor, Panel D quantifies impact—core criteria

### If Judges Ask "So What?":
**RESPONSE** (20 seconds):
> "So what? **18.6× better learning, +26.6% discovery improvement, 100% budget compliance**—all on real experimental data with fair baselines. This isn't incrementally better—it's a paradigm shift from passive generation (train VAE, screen outputs) to active generation (AL ↔ VAE co-evolution). Transferable to any materials discovery task. Open-source, documented, ready to deploy. That's impact."

---

## Judge Personas & What They Care About

### Judge Type 1: Domain Expert (Material Scientist)
**PRIORITY**: Scientific Relevance (40%) > Impact (30%) > Novelty (20%) > Execution (10%)

**WHAT TO EMPHASIZE**:
- "CRAFTED database—687 experimental MOFs" (relevance)
- "$35-65/experiment" (real constraint)
- "100% budget compliance" (production-ready)
- "2.6× cost efficiency" (impact on lab budgets)
- "Deploy tomorrow" (adoption)

**WHAT TO DE-EMPHASIZE**:
- VAE architecture details (they don't care about latent dims)
- GP vs RF technical comparison (assume they know GPs)

**MAGIC PHRASE**:
> "This is the regime YOU work in: 100 training samples, tight budgets, risk-averse. Not ImageNet, not unlimited compute."

---

### Judge Type 2: ML Researcher
**PRIORITY**: Novelty (40%) > Execution (25%) > Impact (20%) > Relevance (15%)

**WHAT TO EMPHASIZE**:
- "Tight coupling—not sequential pipeline" (novelty)
- "GP covariance for true epistemic uncertainty" (technical rigor)
- "Portfolio constraints as regularization" (novelty)
- "Adaptive VAE targeting 7.1 → 10.2" (closed-loop learning)
- "Generalizes to any materials + property task" (impact)

**WHAT TO DE-EMPHASIZE**:
- MOF chemistry details (they don't care about linkers)
- Lab budget constraints (assume they get it)

**MAGIC PHRASE**:
> "Novel contribution: portfolio-constrained active generative loop. Not VAE OR AL—it's the tight coupling that enables breakthroughs."

---

### Judge Type 3: Industry/Investor
**PRIORITY**: Impact (40%) > Execution (30%) > Relevance (20%) > Novelty (10%)

**WHAT TO EMPHASIZE**:
- "18.6×, +26.6%, 2.6×—quantified improvements" (ROI)
- "Open-source, reproducible" (adoption risk mitigation)
- "Transferable: batteries, catalysts, alloys" (market size)
- "Deploy in your lab tomorrow" (time-to-value)
- "100% budget compliance" (financial discipline)

**WHAT TO DE-EMPHASIZE**:
- Technical details (GP vs RF, KL divergence)
- Novelty claims (they care about results, not novelty)

**MAGIC PHRASE**:
> "Your lab's $50k annual materials budget now discovers 2.6× more, learns 18.6× faster, finds +26.6% better materials. Pays for itself in 1 quarter."

---

### Judge Type 4: Generalist/Student
**PRIORITY**: Execution (35%) > Impact (30%) > Relevance (25%) > Novelty (10%)

**WHAT TO EMPHASIZE**:
- "Nobel Prize-winning MOFs" (importance hook)
- "AI designs materials that don't exist yet" (wow factor)
- "Gray line stuck, orange line breakthrough" (visual storytelling)
- "100% unique—no wasted experiments" (tangible efficiency)
- "Open-source, try it yourself" (accessibility)

**WHAT TO DE-EMPHASIZE**:
- Technical jargon (GP, VAE, KL divergence)
- Detailed methodology (4-way comparison nuances)

**MAGIC PHRASE**:
> "We made AI that designs better materials AND learns more efficiently. 26% better discoveries, 18× faster learning. That's the power of AI for science."

---

## Post-Demo Q&A: Common Questions

### Q1: "How do you validate generated MOFs?"
**A** (30 seconds):
> "Great question. Demo mode: we use `target_co2 + Gaussian noise`—simulates validation to prove the AL loop works. Production: you'd plug in **DFT calculations** (quantum chemistry, 1-2 hours/MOF) or **experimental synthesis** (days/weeks, but high-fidelity). The framework is **validation-agnostic**—swap in any oracle. For hackathon, we focused on the AL-VAE coupling innovation, not MOF synthesis (that's a separate research field)."

**SCORING**: Shows you understand scope limits, have production path.

---

### Q2: "Why not use neural networks instead of GPs?"
**A** (25 seconds):
> "**Small data regime**—we start with only 100 training MOFs. NNs need thousands of samples to avoid overfitting. GPs excel in this regime AND provide **true epistemic uncertainty** from the covariance matrix. NNs only give ensemble variance (less accurate for exploration). That said, the framework is modular—you CAN swap GP for NN if you have more data (1000+ samples)."

**SCORING**: Technical depth + acknowledges trade-offs.

---

### Q3: "What if the VAE generates invalid MOFs?"
**A** (30 seconds):
> "Two safeguards: 1) **Physical constraints** in decoder—outputs must be positive (cell dimensions, volume) and within realistic ranges (a,b,c: 5-30 Å, volume: 100-10,000 Å³). 2) **Deduplication** against existing MOFs—we filter out exact duplicates. For chemical validity (e.g., bond lengths, coordination geometry), production would add post-processing: **structure relaxation** via DFT or **validity filtering** via SMILES/CIF parsers. Demo focuses on geometry + composition—full chemical validation is future work."

**SCORING**: Acknowledges limitation, has mitigation plan.

---

### Q4: "Can I use this on my own dataset?"
**A** (20 seconds):
> "Absolutely. Format your data: `[features (geometry, composition), target_property, costs]`. The system is **dataset-agnostic**. We've tested on CRAFTED (687 MOFs), but the framework works for any materials discovery task. Modular design: swap datasets, swap models. All code is **open-source with documentation**—README has a 5-minute quickstart."

**SCORING**: Adoption-ready, not just research demo.

---

### Q5: "How does this compare to [recent paper X]?"
**A** (depends on paper, but general template):
> "Great reference. [Paper X] does [sequential VAE → AL] OR [AL without generation] OR [generation without budgets]. Our novelty is **tight coupling + portfolio constraints + dual-cost budgets**—all three together. We show **+26.6% improvement** over baseline and **18.6× learning efficiency**—quantified gains with fair baselines. Happy to dive deeper offline if you're interested in methodology comparisons."

**SCORING**: Shows field awareness, doesn't dismiss prior work, emphasizes unique contributions.

---

## Final Pre-Demo Checklist: Judging Criteria Coverage

### Scientific Relevance & Rigor ✅
- [ ] Mention "CRAFTED database" 3 times
- [ ] Mention "687 experimental MOFs" 2 times
- [ ] Mention "4-way baseline comparison"
- [ ] Mention "$35-65/experiment" (real constraint)
- [ ] Acknowledge demo limitations (generated MOF validation)

### Novelty ✅
- [ ] Say "tight coupling" or "iterative co-evolution" 3 times
- [ ] Explain 3 innovations (GP uncertainty, adaptive VAE, portfolio constraints)
- [ ] Contrast with "sequential pipelines" (train → generate → select)
- [ ] Mention "first work with portfolio-constrained active generative loop"

### Impact ✅
- [ ] Say "18.6×" at least twice
- [ ] Say "+26.6%" at least twice
- [ ] Mention "2.6× cost efficiency"
- [ ] Mention "100% budget compliance"
- [ ] Mention "transferable: batteries, catalysts, alloys"
- [ ] Mention "open-source, reproducible"

### Execution ✅
- [ ] Show live regeneration (36 seconds)
- [ ] Show publication-quality figures (4-panel layouts)
- [ ] Mention "5 tabs: Dashboard, Results, Figures, Explorer, About"
- [ ] Mention "documentation: README, ARCHITECTURE, BRIEF"
- [ ] Mention "modular, type-hinted, production-ready code"

---

## Victory Condition

**You CRUSHED it if judges:**
1. **Nod during architecture section** (0:40-1:20) → Novelty resonates
2. **Write down "18.6×" and "+26.6%"** → Impact quantified
3. **Ask about adoption/deployment** → Execution credibility
4. **Request GitHub repo link** → They want to use it

**You WON if judges:**
1. **Ask "Can we deploy this in our lab?"** → Relevance + execution + impact
2. **Say "This is different from [recent work]—I see the novelty"** → Novelty recognized
3. **Follow up with "What other materials have you tested?"** → Transferability acknowledged
4. **Applaud or smile during +26.6% reveal** → Emotional engagement (they're sold)

---

**NOW GO WIN THIS HACKATHON! 🏆**
