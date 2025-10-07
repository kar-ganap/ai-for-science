# Hackathon Project Feasibility Analysis (6-7 Hour Window)

## Evaluation Criteria
- ✅ **Viable**: Can complete in 6-7 hours with conservative GPU resources (1-2 GPUs)
- ⚠️ **Risky**: Tight timeline, depends on tools working smoothly
- ❌ **Not Viable**: Requires too much compute/setup time

## Battery Track Projects

### 1. Fine-tune CDVAE on Na-ion Cathodes → Screen with CHGNet
**Computational Viability:** ⚠️ **RISKY**
- Fine-tuning CDVAE: 4-8 GPU-hours (cuts it close)
- Data preparation: 1-2 hours (filter Materials Project for Na cathodes)
- CHGNet screening: Fast (seconds per structure, pre-trained available)
- **Bottleneck:** CDVAE setup can be finicky; if it doesn't work in first hour, you're stuck
- **Resource needs:** 1 GPU for training, CPU for screening

**Audience Appeal:** 🌟🌟🌟 (3/5)
- "AI generates new battery materials" is compelling
- BUT: Crystal structures are abstract without good visualization
- Needs: Interactive 3D viewer (py3Dmol, ASE visualization)

---

### 2. Active Learning Loop for Battery Materials
**Computational Viability:** ✅ **VIABLE**
- M3GNet/CHGNet: Pre-trained, ready to use (0 setup time)
- Uncertainty estimation: Ensemble (3-5 models) or use dropout (~1 GPU-hour to run)
- Active learning loop: Can simulate without actual DFT (use validation set as "oracle")
- **Timeline:**
  - Hour 1: Setup, download pre-trained models
  - Hours 2-3: Implement uncertainty scoring (ensemble disagreement or entropy)
  - Hours 4-5: Run active learning loop, collect metrics
  - Hours 6-7: Visualization dashboard
- **Resource needs:** 1 GPU (or even CPU with small dataset)

**Audience Appeal:** 🌟🌟🌟🌟 (4/5)
- "AI learns where it's uncertain" is very ML-native concept
- Can show learning curves, uncertainty maps
- **Pitch:** "Reducing expensive DFT calculations by 10x by teaching AI where to focus"
- Weakness: Less visually dramatic than generation

---

### 3. Interface/Dendrite Formation Explorer with MACE
**Computational Viability:** ⚠️ **RISKY**
- MACE pre-trained models: Available (MACE-MP-0, MACE-OFF)
- MD simulation: ~1-10 ns feasible in hours on GPU
- **Risk factors:**
  - MACE-MP-0 may not be trained on Li-metal interfaces (extrapolation risk)
  - Need to set up reasonable initial configurations
  - Dendrite formation may not happen in accessible timescales
- **Alternative:** Pre-compute trajectories beforehand, focus on analysis/visualization
- **Resource needs:** 1 GPU for MD

**Audience Appeal:** 🌟🌟🌟🌟🌟 (5/5)
- **Visualizing dendrite growth is extremely compelling!**
- Movies of atoms moving, dendrites penetrating electrolyte
- Clear failure mechanism → design better materials
- **Pitch:** "Seeing why batteries fail at the atomic level"
- **If you can pull this off, it wins for visual impact**

---

## MOF Track Projects

### 4. Conditional Diffusion → Generate DAC MOFs → GCMC Validation
**Computational Viability:** ⚠️ **RISKY**
- Diffusion model training: 4-8 GPU-hours (tight)
- Alternative: Use pre-trained CDVAE, fine-tune on MOF subset (faster)
- GCMC validation: 1-10 CPU-hours per MOF (can parallelize 10-20 MOFs on cluster)
- RASPA setup: 1-2 hours if unfamiliar
- **Timeline:**
  - Hours 1-2: Setup CDVAE/diffusion model, prepare CoRE MOF data
  - Hours 3-5: Fine-tune on CO₂-adsorbing MOFs
  - Hours 6-7: Generate candidates, run GCMC on top 10-20
- **Resource needs:** 1-2 GPUs for training, CPU cluster for GCMC

**Audience Appeal:** 🌟🌟🌟🌟 (4/5)
- "AI designs materials to capture CO₂ from air" is highly topical
- Can show generated structures + predicted performance
- Climate relevance resonates with broad audience
- Weakness: MOF structures less intuitive than "batteries"

---

### 5. Multi-Objective Optimization: CO₂ Uptake + Synthesizability
**Computational Viability:** ✅ **HIGHLY VIABLE**
- GNN for CO₂ uptake prediction: 1-2 GPU-hours on CoRE MOF database
- Synthesizability model: Train on SynMOF database (1 GPU-hour)
- Multi-objective optimization: Genetic algorithm or Bayesian opt (fast, CPU)
- **Timeline:**
  - Hour 1: Download CoRE MOF + SynMOF data
  - Hours 2-3: Train uptake GNN
  - Hour 4: Train synthesizability classifier
  - Hours 5-6: Implement multi-objective search (NSGA-II or similar)
  - Hour 7: Pareto frontier visualization, analyze trade-offs
- **Resource needs:** 1 GPU for training, CPU for optimization
- **This is the safest option computationally**

**Audience Appeal:** 🌟🌟🌟🌟🌟 (5/5)
- **Multi-objective optimization is very relatable** (everyone understands trade-offs)
- Pareto frontier visualization is clean and compelling
- **Pitch:** "Finding materials that are both high-performance AND actually makeable"
- Addresses real-world constraint (most AI designs can't be synthesized)
- Story arc: "AI generates fantasy materials → We add reality check"

---

### 6. Process-Aware Design (GNN + PSA Modeling)
**Computational Viability:** ❌ **NOT VIABLE**
- PSA (Pressure Swing Adsorption) modeling requires specialized knowledge
- Integration with molecular simulations is non-trivial
- Would need existing PSA simulator (not common in Python ecosystem)
- **Too ambitious for 6-7 hours unless you have prior experience**

**Audience Appeal:** 🌟🌟🌟 (3/5)
- Industry relevance high, but technical details lose non-experts

---

## Alternative Ideas (Even More Viable)

### 7. Interactive Materials Property Explorer
**Computational Viability:** ✅ **EASIEST**
- Use pre-trained M3GNet/CHGNet (no training needed!)
- Build Streamlit/Gradio interface
- User uploads structure → instant predictions (energy, stability, band gap)
- **Timeline:**
  - Hours 1-2: Setup models, test inference
  - Hours 3-5: Build web interface
  - Hours 6-7: Add features (structure editor, comparison, database search)
- **Resource needs:** 1 GPU (or even CPU)

**Audience Appeal:** 🌟🌟🌟🌟 (4/5)
- Interactive demos are crowd-pleasers
- "Try your own material!" engagement
- Limited novelty (just using existing models)

---

### 8. Inverse Design: Property → Structure Generation
**Computational Viability:** ✅ **VIABLE**
- Use pre-trained CDVAE or similar
- Conditional generation: "I want band gap = 2.0 eV" → generates structures
- Validate with M3GNet
- **Timeline:**
  - Hours 1-3: Setup conditional sampling from pre-trained model
  - Hours 4-5: Implement property targeting (classifier guidance)
  - Hours 6-7: Generate gallery of results, validate predictions
- **Resource needs:** 1 GPU

**Audience Appeal:** 🌟🌟🌟🌟🌟 (5/5)
- **Inverse design is the "holy grail" story**
- "Tell AI what you want, it designs it" is powerful narrative
- Works for non-technical audience

---

## Recommendations

### **Top 3 for Hackathon Success:**

#### 🥇 **Multi-Objective MOF Optimization (#5)**
**Why:**
- ✅ Computationally safest (all components proven, fast training)
- ✅ Addresses real problem (synthesizability gap)
- ✅ Clear visual output (Pareto frontier)
- ✅ Story resonates: "AI + reality constraints"
- **Best for:** Team with ML background, wants guaranteed results

#### 🥈 **Inverse Design for Batteries (#8)**
**Why:**
- ✅ Uses pre-trained models (low risk)
- ✅ "Designing materials on demand" is compelling
- ✅ Interactive element (user specifies properties)
- **Best for:** Team that wants to show flashy demo

#### 🥉 **Active Learning Loop (#2)**
**Why:**
- ✅ Very low computational risk
- ✅ Addresses real cost problem (DFT expenses)
- ✅ Shows ML methodology well
- ✅ Can generate publication-quality learning curves
- **Best for:** Team with strong ML/CS background, wants to highlight AI innovation

---

### **High-Risk, High-Reward:**

#### 🎰 **Dendrite Visualization (#3)**
**Pros:**
- 🌟🌟🌟🌟🌟 Visual impact (best of all options)
- Strong narrative (seeing failure mechanisms)
- Unique (no one else likely doing this)

**Cons:**
- ⚠️ MACE may not work out-of-the-box for interfaces
- ⚠️ Dendrite formation may require longer timescales than accessible
- ⚠️ Requires MD expertise to interpret results

**Recommendation:** **Only if you have a backup plan** (pre-computed trajectories) or team member with MD experience

---

## Final Verdict

**If you want to WIN on technical merit:** Multi-Objective MOF (#5)
**If you want to WIN on presentation:** Inverse Design (#8) or Dendrite Viz (#3)
**If you want to LEARN the most:** Active Learning (#2)

**Conservative choice:** #5 or #2
**Ambitious choice:** #3 or #8
**Don't attempt:** #1, #6 (too tight on time)

---

## Resource Requirements Summary

| Project | GPU-Hours | CPU-Hours | Risk | Demo Quality |
|---------|-----------|-----------|------|--------------|
| #1 CDVAE Fine-tune | 4-8 | 2-4 | ⚠️ High | 🌟🌟🌟 |
| #2 Active Learning | 1-2 | 2-4 | ✅ Low | 🌟🌟🌟🌟 |
| #3 Dendrite MD | 2-4 | 1-2 | ⚠️ Medium | 🌟🌟🌟🌟🌟 |
| #4 MOF Diffusion | 4-8 | 10-20 | ⚠️ Medium | 🌟🌟🌟🌟 |
| #5 Multi-Objective | 2-3 | 1-2 | ✅ Low | 🌟🌟🌟🌟🌟 |
| #6 Process-Aware | 2-4 | 10-20 | ❌ High | 🌟🌟🌟 |
| #7 Interactive Explorer | 0-1 | 1-2 | ✅ Lowest | 🌟🌟🌟🌟 |
| #8 Inverse Design | 1-2 | 2-4 | ✅ Low | 🌟🌟🌟🌟🌟 |

**With conservative GPU resources (1-2 A100s for 6-7 hours):**
- ✅ Feasible: #2, #5, #7, #8
- ⚠️ Risky but possible: #1, #3, #4
- ❌ Skip: #6
