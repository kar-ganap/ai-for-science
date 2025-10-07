# MOF Project: Baseline, Hard, and Ambitious Formulations

## Overview: The Core Concept

**"Active Inverse Design for Synthesizable CO₂-Capturing MOFs"**

Generate MOFs with target properties → Score on performance + synthesizability + confidence → Use active learning to validate uncertain predictions → Refine and regenerate

---

## 🥉 BASELINE: "Multi-Objective Screening with Uncertainty"

### Scope: **No generation, focus on screening + multi-objective + AL**

**What you do:**
1. Take existing MOF database (CoRE MOF: 12K structures)
2. Train/use GNN to predict CO₂ uptake
3. Train synthesizability classifier (from SynMOF data)
4. Implement uncertainty estimation (ensemble or dropout)
5. Multi-objective Pareto frontier: Performance vs. Synthesizability vs. Confidence
6. Active learning loop: Select uncertain MOFs → Validate from held-out set → Retrain
7. Simple visualization dashboard

**What you skip:**
- ❌ No generative model (no inverse design)
- ❌ No new MOF structures created
- ❌ Use existing database only

### Timeline (6-7 hours)

**Hour 1: Data & Setup**
- Download CoRE MOF database (~12K structures with GCMC results)
- Download SynMOF database (synthesis conditions from literature)
- Setup: PyTorch Geometric, basic libraries

**Hour 2: Property Prediction Model**
- Use pre-trained MatGL/M3GNet OR
- Train simple CGCNN on CoRE MOF data (CO₂ uptake prediction)
- Target: Predict uptake at 0.15 bar, 298K (flue gas conditions)

**Hour 3: Synthesizability Model**
- Extract features from SynMOF (metal nodes, linker complexity, topology)
- Train Random Forest classifier: synthesizable vs. not
- Binary classification (reported in literature = 1, else = 0)

**Hour 4: Uncertainty Estimation**
- Train ensemble (5 models) for uptake prediction
- Uncertainty = standard deviation of ensemble predictions
- Alternative: Single model with MC dropout

**Hour 5: Multi-Objective Optimization**
- Score all 12K MOFs on 3 objectives:
  - CO₂ uptake (maximize)
  - Synthesizability score (maximize)
  - Confidence = 1/(1+uncertainty) (maximize)
- Compute Pareto frontier (NSGA-II or simple non-dominated sorting)
- Visualize 3D scatter plot

**Hour 6: Active Learning Loop**
- Split data: 10K train, 2K validation (oracle pool)
- AL iteration 1: Select 100 uncertain high-performers
- "Validate" from oracle pool
- Retrain models
- Repeat 2-3 iterations

**Hour 7: Dashboard & Presentation**
- Streamlit app with:
  - Pareto frontier visualization (interactive Plotly)
  - Learning curve (accuracy vs. oracle queries)
  - Top 10 recommended MOFs with structures
- Prepare slides

### Deliverables

✅ **Working demo** showing multi-objective trade-offs
✅ **Active learning** reduces uncertainty over iterations
✅ **Quantitative metrics:**
- Pareto frontier evolution
- Uncertainty reduction: Start 30% ± 15% → End 30% ± 5%
- AL sample efficiency: Reach target accuracy with 50% fewer oracle queries

### Strengths
- ✅ **Guaranteed to work** (no generative model risk)
- ✅ **Shows all key concepts** (multi-objective + AL)
- ✅ **Real data, real metrics**

### Weaknesses
- ⚠️ **No inverse design** (less "AI generates new materials" wow factor)
- ⚠️ **Limited novelty** (screening existing database)
- ⚠️ **Smaller story** ("better screening" vs. "AI designs materials")

### Win Probability: 🏆🏆🏆 (3/5)
- Solid technical execution
- May lose to flashier projects with generation
- Best for: Risk-averse team, wants guaranteed working demo

---

## 🥈 HARD: "Full Pipeline with Conditional Generation"

### Scope: **Everything - generation + multi-objective + AL**

**What you do:**
1. Use pre-trained generative model (CDVAE or MOF-specific)
2. Conditional generation: Target CO₂ uptake > 10 mmol/g
3. Generate 1000-5000 candidate MOFs
4. Property prediction (GNN for CO₂ uptake)
5. Synthesizability prediction (ML classifier)
6. Uncertainty estimation (ensemble)
7. Multi-objective Pareto frontier
8. Active learning: Validate uncertain candidates → Update models → Regenerate
9. Interactive dashboard with generation controls

**What makes it hard:**
- ⚠️ Generative model integration (CDVAE can be finicky)
- ⚠️ Generated structures need validation (stability, validity)
- ⚠️ More moving parts = higher risk

### Timeline (6-7 hours)

**Hour 0-1: Setup & Pre-work (CRITICAL)**
- **DO THIS BEFORE HACKATHON:**
  - Download and test CDVAE (or find MOF-specific generative model)
  - Ensure it runs on your hardware
  - Pre-download CoRE MOF, SynMOF datasets
  - Test structure file I/O (CIF format)

**Hour 1-2: Conditional Generation**
- Load pre-trained CDVAE (trained on Materials Project or MOF subset)
- Implement conditional sampling:
  ```python
  target_properties = {"co2_uptake": 10}  # mmol/g target
  candidates = cdvae.sample(
      n_samples=1000,
      condition=target_properties
  )
  ```
- Validate generated structures:
  - Charge neutrality check
  - Reasonable bond lengths (0.5-3 Å)
  - Porosity check (accessible volume > 0)
- Filter: Keep ~500-800 valid structures

**Hour 2-3: Property Prediction Pipeline**
- Option A: Use pre-trained MatGL/M3GNet (universal, slower)
- Option B: Train CGCNN on CoRE MOF (faster, domain-specific)
- Predict CO₂ uptake for generated + existing MOFs
- Build ensemble (3-5 models) for uncertainty

**Hour 3-4: Synthesizability Model**
- Features from SynMOF:
  - Metal node type (Zn, Cu, Mg → easy; rare earths → hard)
  - Linker complexity (# functional groups, molecular weight)
  - Topology family (common like pcu, dia → easy)
- Train classifier (Random Forest or XGBoost)
- Predict synthesizability for generated MOFs

**Hour 4-5: Multi-Objective + Active Learning**
- Score all MOFs (generated + existing) on 3 objectives
- Compute Pareto frontier
- **AL Loop (2-3 iterations):**
  - Iteration 1: Select 50 uncertain high-performers
    - Criteria: High uptake (>8 mmol/g) + High synth score (>0.7) + High uncertainty
  - Validate using oracle (held-out GCMC results for existing MOFs)
  - For generated MOFs: Run quick RASPA GCMC (if time) OR use GNN as proxy
  - Retrain property predictor
  - **Regenerate:** Sample new MOFs with updated model as guidance

**Hour 5-6: Visualization & Iteration**
- 3D Pareto frontier (animated evolution across AL iterations)
- Uncertainty heatmap (chemical space visualization)
- Top 10 discovered MOFs:
  - Show structure (py3Dmol visualization)
  - Show predicted properties
  - Show confidence score
  - Show synthesis recommendation (solvent, temperature from SynMOF model)

**Hour 6-7: Polish & Presentation**
- Interactive Streamlit dashboard:
  - User input: Target CO₂ uptake slider
  - "Generate" button → Creates new MOFs
  - Live Pareto frontier update
  - "Run Active Learning" → Executes AL loop
- Prepare presentation with:
  - Problem statement (synthesizability gap)
  - Demo walkthrough
  - Metrics (Pareto improvement, uncertainty reduction, AL efficiency)

### Deliverables

✅ **Full closed-loop system:**
- Generate → Score → Validate → Learn → Regenerate

✅ **Conditional inverse design:**
- User specifies properties → AI generates MOFs

✅ **Multi-objective with 3D Pareto frontier**

✅ **Active learning improves over iterations**

✅ **Quantitative metrics:**
- Generated X novel MOFs
- Y% passed stability/validity checks
- Top candidates: Z mmol/g predicted uptake, W% synthesis confidence
- AL reduced uncertainty from A% to B%
- Pareto frontier expanded by C%

### Technical Risks

⚠️ **Risk 1: CDVAE Integration (Medium-High)**
- Mitigation: Test before hackathon, have backup (use existing MOF database)

⚠️ **Risk 2: Generated MOFs Invalid (Medium)**
- Mitigation: Strict filtering, show % valid in demo ("generated 1000, 600 passed checks")

⚠️ **Risk 3: Time Pressure (Medium)**
- Mitigation: Skip GCMC validation, use GNN predictions as oracle

### Strengths
- ✅ **Complete story** (full pipeline)
- ✅ **Inverse design** (high wow factor)
- ✅ **Novel integration** (all three techniques together)
- ✅ **Addresses THE problem** (synthesizability gap)

### Weaknesses
- ⚠️ **Higher risk** (more can go wrong)
- ⚠️ **Generated MOFs may not be novel** (CDVAE might generate close to training data)
- ⚠️ **Validation is simulated** (not real GCMC)

### Win Probability: 🏆🏆🏆🏆 (4/5)
- Strong technical demonstration
- Complete narrative
- May win if execution is solid
- Best for: Confident team with ML/materials experience

---

## 🥇 AMBITIOUS: "Closed-Loop Discovery with Real Validation"

### Scope: **Everything + real GCMC validation + reticular chemistry**

**What you do (everything from HARD, plus):**
1. **Real oracle:** Run actual RASPA GCMC simulations for validation
2. **Reticular design:** Enumerate MOFs from building blocks (metal nodes + linkers)
3. **Hierarchical generation:** Generate topology → Select nodes → Place linkers
4. **Process-aware scoring:** Add regeneration energy as 4th objective
5. **Synthesis planning:** Predict full synthesis recipe (solvent, temp, time, modulator)
6. **Experimental handoff:** Output CIF files + synthesis protocols for top 3

**What makes it ambitious:**
- 🔥 Real GCMC (computationally intensive)
- 🔥 Reticular chemistry (domain expertise needed)
- 🔥 More sophisticated generation
- 🔥 Synthesis planning (not just synthesizability)

### Timeline (6-7 hours) - **TIGHT**

**Pre-Hackathon (Essential):**
- ✅ RASPA installed and tested
- ✅ CDVAE working and tested
- ✅ All datasets downloaded and preprocessed
- ✅ Code templates for each module ready
- ✅ Team roles assigned (1 person on generation, 1 on GCMC, 1 on dashboard)

**Hour 1: Reticular Design Space**
- Define building blocks:
  - Metal nodes: Zn²⁺, Cu²⁺, Mg²⁺, etc. (10 types)
  - Organic linkers: BDC, BTB, BTC, etc. (20 types)
  - Topologies: pcu, dia, sod, etc. (10 common)
- Enumerate: 10 × 20 × 10 = 2000 possible combinations
- Use heuristics to filter: charge balance, size matching
- Generate ~500 candidate MOFs using reticular assembly

**Hour 1-2: Conditional Generation (Parallel)**
- CDVAE generates another 500 from learned distribution
- Total pool: 1000 candidate MOFs

**Hour 2-3: Property Prediction & Uncertainty**
- GNN ensemble predicts CO₂ uptake + uncertainty
- Fast pre-screening: Keep top 200 by predicted uptake

**Hour 3-4: Multi-Objective Scoring**
- Objective 1: CO₂ uptake (from GNN)
- Objective 2: Synthesizability (from SynMOF model)
- Objective 3: Confidence (from ensemble)
- Objective 4: Regeneration energy (from GNN or heuristic: f(binding_strength))
- 4D Pareto frontier (visualize as 2D projections or parallel coordinates)

**Hour 4-5: Active Learning with REAL GCMC**
- **AL Iteration 1:**
  - Select 20 uncertain high-performers
  - Run RASPA GCMC in parallel (20 CPU cores, ~30 min wall time)
  - Get true CO₂ uptake isotherms
  - Retrain GNN with validated data

- **AL Iteration 2:**
  - Regenerate 200 new MOFs (guided by updated model)
  - Select another 20 uncertain
  - Run GCMC (parallel)
  - Retrain

**Hour 5-6: Synthesis Planning**
- For top 10 Pareto-optimal MOFs:
  - Use SynMOF model to predict synthesis conditions:
    - Solvent (DMF, methanol, etc.)
    - Temperature (80-150°C)
    - Time (24-72 hrs)
    - Modulator (acetic acid, etc.)
  - Generate synthesis protocol text (using template)
  - Export CIF files (structure files for experimentalists)

**Hour 6-7: Advanced Visualization**
- Interactive dashboard with:
  - **4D Pareto frontier** (parallel coordinates plot)
  - **Chemical space map** (t-SNE of MOF embeddings, color by performance)
  - **Live GCMC viewer** (show isotherms updating in real-time)
  - **Uncertainty evolution** (animated heatmap across AL iterations)
  - **Synthesis protocol generator** (click MOF → get full recipe)
  - **Experimental package** (download button for CIF + protocol)

### Deliverables

✅ **Complete discovery platform:**
- Reticular design + generative model
- Real GCMC validation (not simulated)
- 4-objective optimization
- Closed-loop learning

✅ **Lab-ready outputs:**
- Top 3 MOFs with:
  - Structure (CIF file)
  - Predicted performance (with confidence intervals)
  - Full synthesis protocol
  - Experimental validation plan

✅ **Advanced metrics:**
- Reticular space explored: X% of possible combinations
- GCMC validations: Y structures (cost equivalent: $Z)
- Model improvement: RMSE improved from A to B meV/atom
- Pareto frontier: Discovered W non-dominated MOFs
- Success rate: X% of validated MOFs met performance target

### Technical Risks

🔥 **Risk 1: GCMC Time (HIGH)**
- Each GCMC: 30-60 min even optimized
- 40 total (2 iterations × 20) = up to 20 GPU-hours (if parallelized) or 40 CPU-hours
- **Mitigation:** Pre-compute some, use very short GCMC runs (10⁴ steps instead of 10⁶)

🔥 **Risk 2: Complexity (HIGH)**
- Many moving parts, any failure breaks the pipeline
- **Mitigation:** Modular design, each component can demo independently

🔥 **Risk 3: Time (CRITICAL)**
- This is realistically a 2-day project compressed to 7 hours
- **Mitigation:** Exceptional team (3-4 people), perfect pre-work, no debugging

### Strengths
- ✅ **Maximum impact** (complete system, lab-ready)
- ✅ **Real validation** (not simulated)
- ✅ **Novel integration** (reticular + generative + AL)
- ✅ **Publishable quality** (could write paper after hackathon)
- ✅ **Experimental handoff** (bridges computation → lab)

### Weaknesses
- 🔥 **Very high risk** (many failure points)
- 🔥 **Tight timeline** (no room for error)
- 🔥 **Compute intensive** (needs cluster or many cores)

### Win Probability: 🏆🏆🏆🏆🏆 (5/5 IF it works, 2/5 if it breaks)
- High-risk, high-reward
- If executed well: **Winner**
- If execution has issues: May lose to simpler projects
- Best for: Expert team, perfect preparation, access to compute

---

## Comparison Table

| Feature | Baseline | Hard | Ambitious |
|---------|----------|------|-----------|
| **Generative Model** | ❌ No | ✅ CDVAE | ✅ CDVAE + Reticular |
| **Property Prediction** | ✅ GNN | ✅ GNN Ensemble | ✅ GNN Ensemble |
| **Synthesizability** | ✅ Binary | ✅ Binary | ✅ Full Protocol |
| **Uncertainty** | ✅ Ensemble | ✅ Ensemble | ✅ Ensemble |
| **Multi-Objective** | ✅ 3D Pareto | ✅ 3D Pareto | ✅ 4D Pareto |
| **Active Learning** | ✅ Simulated | ✅ Simulated | ✅ Real GCMC |
| **Validation** | Held-out set | Held-out set | RASPA GCMC |
| **Novel MOFs** | 0 | 500-1000 | 1000+ |
| **Lab-Ready Output** | ❌ No | ⚠️ Structures only | ✅ CIF + Protocol |
| **GPU Hours** | 1-2 | 2-3 | 3-4 |
| **CPU Hours** | 2-3 | 3-4 | 20-40 |
| **Risk Level** | ✅ Low | ⚠️ Medium | 🔥 High |
| **Win Probability** | 🏆🏆🏆 | 🏆🏆🏆🏆 | 🏆🏆🏆🏆🏆 or 🏆🏆 |
| **Team Size** | 1-2 | 2-3 | 3-4 |
| **Prep Work** | 2 hours | 4-6 hours | 10+ hours |

---

## Recommendations by Team Profile

### Team Profile 1: Solo or Duo, First Hackathon
→ **Go BASELINE**
- Guaranteed working demo
- Learn all concepts
- Build confidence
- Can always add generation if time permits

### Team Profile 2: Experienced Team, Good Prep Time
→ **Go HARD**
- Sweet spot of ambition vs. risk
- Shows complete pipeline
- High chance of winning
- **This is the recommended default**

### Team Profile 3: Expert Team, Perfect Prep, Compute Access
→ **Go AMBITIOUS**
- Only if ALL conditions met:
  - ✅ 3-4 people with clear roles
  - ✅ 10+ hours prep work done
  - ✅ RASPA tested and working
  - ✅ Access to compute cluster (20+ cores)
  - ✅ At least one domain expert (MOFs/reticular chemistry)
- If ANY condition fails → Drop to HARD

### Team Profile 4: ML-Heavy Team, No Materials Background
→ **Go HARD, skip reticular chemistry**
- Focus on ML techniques (generative + AL)
- Use CDVAE without domain customization
- Strong on visualization and interaction
- Lean into "AI learns reality" narrative

---

## Recommended Strategy: Progressive Enhancement

### Start with BASELINE, add features if ahead of schedule:

**Hour 1-3: Core Baseline** (guaranteed working)
- Data + Property prediction + Synthesizability + Uncertainty
- Multi-objective scoring
- Simple Pareto frontier

**Hour 4: Checkpoint Decision**
- ✅ If on track → Add generation (move to HARD)
- ⚠️ If behind → Stay BASELINE, polish what you have

**Hour 5: Second Checkpoint**
- ✅ If generation working → Add regeneration loop
- ⚠️ If generation buggy → Remove it, stay BASELINE

**Hour 6: Final Features**
- ✅ If everything working → Consider real GCMC (move to AMBITIOUS)
- ⚠️ If anything broken → Debug and polish

**Hour 7: Presentation**
- Stop coding at 5:30pm (30 min before deadline)
- Polish dashboard, prepare demo, make slides

### This ensures:
- ✅ You always have a working demo
- ✅ You add features only if time permits
- ✅ You don't break working code chasing ambitious features

---

## Final Recommendation

### **Target: HARD, Fallback: BASELINE, Stretch: AMBITIOUS**

**Plan for HARD:**
- Full pipeline with generation
- High win probability if executed well
- Manageable risk with good prep

**Prepare BASELINE:**
- If generation fails, pivot immediately
- Still shows all key concepts
- Guaranteed working demo

**Consider AMBITIOUS only if:**
- Team of 3-4 experts
- 10+ hours prep work
- Compute cluster access confirmed
- Comfortable with high risk

**Most likely to win: Well-executed HARD version**

The HARD version hits the sweet spot:
- ✅ Complete story (generation + multi-objective + AL)
- ✅ Manageable risk (pre-trained models reduce unknowns)
- ✅ Novel contribution (unexplored intersection)
- ✅ Great visualization (3D Pareto, uncertainty maps)
- ✅ Addresses real problem (synthesizability gap)

**Don't be a hero - start BASELINE, enhance to HARD if time allows.**
