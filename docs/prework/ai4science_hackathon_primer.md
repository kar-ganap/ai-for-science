# AI4Science Hackathon: Technical Primer

## Problem Space 1: Next-Generation Battery Materials Discovery

### a) General Problem Space

The battery materials discovery challenge encompasses several interconnected problems:

**Solid-State Electrolytes (SSEs):**
- Need for high ionic conductivity (>10⁻³ S/cm at room temperature) with negligible electronic conductivity
- Mechanical stability to suppress dendrite formation and penetration
- Wide electrochemical stability window (0-5V vs Li/Li⁺)
- Chemical/interfacial compatibility with electrode materials
- Low cost and environmental sustainability

**Sodium-Ion Batteries (SIBs):**
- Alternative to lithium-based systems using more abundant Na
- Hard carbon anodes (graphite doesn't work for Na intercalation)
- Cathode materials: layered oxides, polyanionic compounds, Prussian blue analogs
- Challenge: achieving energy density comparable to Li-ion while maintaining cost advantages

**Critical Challenges:**
1. **Ionic Transport:** Understanding and optimizing Li⁺/Na⁺ migration mechanisms through bulk materials
2. **Interfacial Phenomena:** Solid-electrolyte interphase (SEI) formation, dendrite nucleation/growth
3. **Mechanical-Electrochemical Coupling:** Volume changes during cycling, stress-induced degradation
4. **Materials Stability:** Thermodynamic stability, air/moisture sensitivity, phase transitions

### b) State of the Art

**Solid-State Electrolytes:**
- Sulfide-based: Li₁₀GeP₂S₁₂ (LGPS, σ ≈ 10⁻² S/cm), Li₆PS₅Cl (argyrodite)
- Oxide-based: Li₇La₃Zr₂O₁₂ (LLZO, garnet structure)
- Polymer-based: PEO with Li salts
- Recent ML discovery: Li₄.₅TiO₃.₂₅ identified via high-throughput AIMD screening

**Sodium-Ion Cathodes:**
- Ni-based layered oxides (O3-ZNMT, O3-NFM) exceeding LFP volumetric energy density by 24-37%
- Polyanionic compounds: NaFePO₄, Na₃V₂(PO₄)₃
- Prussian blue analogs: NaMnFe(CN)₆

**Performance Metrics:**
- Best SIB cathodes reach ~150 Wh/kg (comparable to LFP)
- Solid-state batteries targeting >500 Wh/kg (vs. 250-300 Wh/kg for current Li-ion)

### c) Simulation/Computational Approaches

**Density Functional Theory (DFT):**

*What it does:* Solves electronic structure problem by minimizing energy functional of electron density
- **Functionals:** GGA (PBE), GGA+U for transition metal oxides, hybrid functionals (HSE06) for band gaps
- **Output:** Formation energy, band structure, density of states, migration barriers
- **Cost:** ~10³-10⁴ CPU-hours for unit cell optimization; scales as O(N³) with electrons

*Applications:*
- Thermodynamic stability: Formation energy Eᶠ = E(compound) - Σμᵢnᵢ
- Phase stability: Convex hull analysis (distance above hull determines synthesizability)
- Voltage calculations: V = -ΔG/nF ≈ -ΔE/nF for intercalation reactions

**Nudged Elastic Band (NEB) Method:**

*What it does:* Finds minimum energy path (MEP) for ion migration
- Creates "elastic band" of intermediate images between initial/final states
- Optimizes perpendicular forces while maintaining equal spacing
- **Output:** Migration barrier Eₐ (activation energy for diffusion)
- **Cost:** ~10× single-point DFT calculation per image (typically 5-11 images)

*Key Results:*
- Excellent SSEs show barriers Eₐ < 0.3 eV
- Migration mechanisms: vacancy hopping, interstitialcy, knock-off, concerted migration
- Dataset: 619 literature-derived migration barriers for Li/Na/K systems

**Ab Initio Molecular Dynamics (AIMD):**

*What it does:* MD with forces from on-the-fly DFT calculations
- Timestep: 0.5-2 fs (femtoseconds)
- Temperature: 600-1200K for accelerated sampling
- Duration: 10-100 ps typical (10⁴-10⁵ steps)
- **Cost:** ~10⁵-10⁶ CPU-hours for meaningful statistics

*Applications:*
- Ionic conductivity: σ = (nq²/k_BT)D, where D from mean squared displacement
- Disordered/amorphous phases (SEI, liquid electrolytes)
- Finite temperature effects, collective diffusion mechanisms
- Validation/training data for ML force fields

**Classical Molecular Dynamics:**

*Force fields:* Parametric functions (Lennard-Jones, Buckingham, embedded atom method)
- Timestep: 0.5-2 fs
- Duration: ns-μs accessible
- System size: 10⁴-10⁶ atoms
- **Cost:** ~10²-10³ CPU-hours for μs trajectory

*Limitations:*
- Requires pre-existing parameters (often unavailable for new materials)
- No bond breaking/formation (no electrochemistry)
- Limited accuracy for polarization, charge transfer

### d) Where ML Fits In

**Property Prediction Models:**

*Graph Neural Networks (GNNs):*
- **CGCNN (Crystal Graph CNN):** Nodes = atoms, edges = bonds, convolutions aggregate neighbor information
- **MEGNet, SchNet:** Similar architectures with different aggregation schemes
- **Input:** Crystal structure (atom types, positions, lattice)
- **Output:** Formation energy, band gap, ionic conductivity

*Performance:*
- Formation energy: MAE ~25-50 meV/atom (cf. DFT ≈ 20-50 meV/atom to experiment)
- MBVGNN for Na-ion cathodes: 43-48% improvement over baseline GNNs
- Identified 194 high-energy-density ternary Na cathode candidates

*Training Data:*
- Materials Project: ~150K compounds with DFT energies
- OQMD, AFLOW: Similar scale databases
- Challenge: Data imbalance (many oxides, few sulfides/fluorides)

**Machine-Learned Force Fields (MLFFs):**

These are the most transformative ML application for battery materials.

*Architecture Comparison:*

| Model | Training Data | Key Innovation | DFT Accuracy | Speed-up |
|-------|--------------|----------------|--------------|----------|
| **M3GNet** | Materials Project (>1M structures) | 3-body interactions, universal (full periodic table) | Forces: ~100 meV/Å | ~10⁴-10⁶× |
| **CHGNet** | Materials Project (1.5M structures) | Charge-aware, predicts magnetic moments | Energy: ~30 meV/atom | ~10⁴-10⁶× |
| **MACE** | MPtrj (Materials Project trajectories) | Higher-order equivariant message passing | Energy: ~20 meV/atom | ~10⁴-10⁶× |

*Technical Details:*

**Equivariance:** Output transforms correctly under rotations/translations
- E(R{r}) = E({r}), F(R{r}) = RF({r})
- Implemented via spherical harmonics, tensor products
- Preserves physical symmetries → better generalization

**Message Passing:**
1. Embed atom types: h⁽⁰⁾ᵢ = embedding(Zᵢ)
2. For each layer: hᵢ⁽ˡ⁺¹⁾ = UPDATE(hᵢ⁽ˡ⁾, Σⱼ MESSAGE(hᵢ⁽ˡ⁾, hⱼ⁽ˡ⁾, rᵢⱼ))
3. Energy: E = Σᵢ READOUT(hᵢ⁽ᴸ⁾)
4. Forces: F = -∇E (automatic differentiation)

*Applications:*

**CHGNet for Battery Cathodes:**
- Simulated LiMnO₂ phase transformations with Mn migration + charge transfer
- Revealed coupling between Jahn-Teller distortions and Li diffusion in LiNiO₂
- Large-scale (>1000 atoms) simulations at ns timescales

**M3GNet for Discovery:**
- Screened 31 million hypothetical structures in hours (would take decades with DFT)
- Identified candidates with 70% less Li than conventional batteries
- Ionic diffusivity calculations via MD at 300K (unfeasible with AIMD)

**MACE for Interfaces:**
- Accurate SEI formation dynamics (organic molecules + Li metal)
- Dendrite nucleation/penetration in solid electrolytes
- Achieves chemical accuracy (~1 kcal/mol = 43 meV) for reaction barriers

*Challenges:*
- **Extrapolation:** Performance degrades for chemistries/structures far from training data
- **Active Learning:** Iteratively add DFT calculations for uncertain regions
- **Charge Transfer:** CHGNet addresses this; remains challenging for M3GNet/MACE

**High-Throughput Screening Workflows:**

*Typical Pipeline:*
1. **Enumeration:** Generate candidate structures (substitutions, decorations, prototype matching)
2. **Pre-screening:** Heuristics (charge balance, size mismatch, electronegativity)
3. **GNN Filtering:** Predict formation energy, stability (~1 GPU-second/structure)
4. **DFT Relaxation:** Refine top candidates (~1000 CPU-hours/structure)
5. **Validation:** NEB for migration barriers, AIMD for conductivity (~10⁴ CPU-hours)
6. **Experimental Synthesis:** Top 5-10 candidates

*AiiDA Framework:*
- Workflow management for automated DFT calculations
- Provenance tracking (reproducibility)
- Integration with Materials Project, NOMAD

**Active Learning:**

*Concept:* Iteratively improve models by strategically selecting new DFT calculations

*Algorithm:*
1. Train initial MLFF on small dataset
2. Run MD simulations
3. Identify high-uncertainty configurations (ensemble disagreement, gradient norms)
4. Compute DFT for uncertain structures
5. Retrain → Repeat

*Benefits:*
- Reduces DFT calculations by 10-100× vs. random sampling
- Improves extrapolation to new chemistries
- Recent work: ≤1000 DFT calculations sufficient for battery-specific potentials

### e) Where Generative AI Fits In

**Diffusion Models for Crystal Structures:**

*DiffCSP (Diffusion for Crystal Structure Prediction):*

*How it works:*
1. **Forward Process:** Gradually add Gaussian noise to fractional coordinates and lattice parameters
   - xₜ = √(αₜ)x₀ + √(1-αₜ)ε, ε ~ N(0,I)
   - Over T steps (typically 1000), structure becomes random
2. **Reverse Process:** Train neural network ε_θ to predict noise
   - Learn p(xₜ₋₁|xₜ) to denoise step-by-step
3. **Generation:** Start from random noise, iteratively denoise
4. **Symmetry Preservation:** Operate in fractional coordinates, apply space group constraints

*Performance:*
- Validity: 80-90% generated structures are charge-balanced, physically reasonable
- Property Targeting: Conditional generation (target band gap, formation energy)

*CDVAE (Crystal Diffusion VAE):*
- Combines VAE latent space with diffusion
- E(3)-equivariant graph networks preserve rotation/translation symmetry
- State-of-the-art validity and diversity metrics

**MatterGen (Recent Breakthrough):**

*Capabilities:*
- Generates stable materials across full periodic table
- Conditional on: chemistry (elements), symmetry (space group), properties (band gap, magnetic/mechanical)
- **Stability:** 2× more likely to be stable than previous models
- **Novelty:** Generated structures not in training set

*Architecture:*
- Denoising diffusion probabilistic model
- Hierarchical: First generate space group, then composition, then atomic positions
- Training: Materials Project + proprietary datasets

*Applications for Batteries:*
- Generate novel Li/Na solid electrolytes with target ionic conductivity
- Design cathode materials with specific voltage windows
- Explore chemistries not yet synthesized (e.g., rare-earth-free alternatives)

**GFlowNets (Generative Flow Networks):**

*Fundamental Difference:*
- **Diffusion:** All-at-once generation (sample from learned distribution)
- **GFlowNets:** Sequential decision-making (build structure step-by-step)

*Algorithm:*
1. Define actions: select space group → add element → place atom → set lattice parameter
2. Train policy to maximize: P(structure) ∝ exp(Reward/T)
3. Reward = property of interest (e.g., predicted ionic conductivity)
4. Explore diverse high-reward structures

*Crystal-GFN:*
- Generates diverse candidates for target properties
- Better mode coverage than VAEs (which collapse to mean)
- Computational cost: Higher than diffusion (sequential generation)

*SHAFT (Symmetry-Aware Hierarchical GFlowNet):*
- Decomposes search: space group → lattice → atoms
- Exploits symmetry to reduce search space exponentially
- Outperforms vanilla GFlowNets and CDVAE

*When to Use GFlowNets:*
- Need diversity (multiple distinct solutions)
- Have expensive reward function (minimize evaluations)
- Sequential constraints (e.g., synthesis planning)

**Integrated Generative Workflow for Batteries:**

```
[Generative Model] → [MLFF Screening] → [DFT Validation] → [Experimental Synthesis]
     (10⁶ candidates)      (10³ relax)      (10² detailed)       (5-10 samples)
     ~GPU-hours           ~CPU-hours       ~10³ CPU-hr/each      ~months
```

*Example:*
1. **MatterGen:** Generate 1M Na-containing structures with target formation energy < -1 eV/atom
2. **M3GNet:** Relax structures, filter for dynamic stability (no imaginary phonons)
3. **GNN Predictor:** Estimate migration barriers (trained on NEB dataset)
4. **CHGNet MD:** Calculate diffusivity for top 100 at 300K, 600K
5. **DFT+NEB:** Validate top 20 with accurate barriers
6. **Experimental:** Synthesize top 5

*Current Limitations:*
- **Synthesizability:** Generative models don't guarantee synthetic accessibility
- **Composition Bias:** Models reflect training data (mostly oxides)
- **Property Accuracy:** Predicted properties from surrogate models have errors
- **Validation Bottleneck:** Still need extensive DFT/experimental verification

---

## Problem Space 2: Carbon Capture Materials (MOFs)

### a) General Problem Space

**Metal-Organic Frameworks (MOFs):**
- Crystalline porous materials: metal nodes + organic linkers
- Pore sizes: 0.5-5 nm, surface areas: 1000-7000 m²/g
- Tunable chemistry: >100,000 synthesized, >500,000 hypothetical in databases

**Carbon Capture Applications:**

*Post-Combustion Capture (Flue Gas):*
- CO₂ concentration: 10-15%
- N₂ as main impurity (~75%)
- Temperature: 40-60°C
- Pressure: ~1 bar
- **Target:** High CO₂/N₂ selectivity (>100), working capacity >3 mmol/g

*Direct Air Capture (DAC):*
- CO₂ concentration: 400 ppm (0.04%)
- Temperature/humidity variations
- Pressure: 1 bar
- **Target:** Extremely high selectivity, CO₂ uptake at ultra-low partial pressure, moisture stability

*Pre-Combustion Capture:*
- CO₂/H₂ separation in gasification/reforming
- Higher pressures (15-40 bar)
- Higher CO₂ concentrations (15-40%)

**Key Performance Metrics:**

1. **CO₂ Uptake:** mmol/g at specific P, T (adsorption capacity)
2. **Selectivity:** α_CO2/N2 = (x_CO2/y_CO2)/(x_N2/y_N2), where x = adsorbed, y = gas phase
3. **Working Capacity:** Δn = n(P_ads, T_ads) - n(P_des, T_des)
4. **Regeneration Energy:** Q_reg = Q_sensible + Q_desorption
5. **Stability:** Cycling performance, moisture tolerance, thermal/chemical stability

**Challenges:**

1. **Moisture Competition:** H₂O adsorbs strongly, blocks CO₂ sites
2. **Cost:** MOF synthesis expensive; scalability unclear
3. **Mechanical Stability:** Frameworks can collapse under pressure/vacuum cycling
4. **Kinetics:** Diffusion limitations in narrow pores
5. **Material-Process Gap:** Best material in single-component tests ≠ best in real process

### b) State of the Art

**Top-Performing MOFs:**

*Post-Combustion Capture:*
- **Mg-MOF-74 (CPO-27-Mg):**
  - Open metal sites (coordinatively unsaturated Mg²⁺)
  - CO₂ uptake: 8.9 mmol/g at 0.15 bar, 298K
  - Selectivity: ~200 (CO₂/N₂)
  - Issue: Moisture sensitive, high regeneration energy

- **SIFSIX-3-Ni:**
  - Inorganic pillars with strong electrostatic CO₂ binding
  - Record selectivity: >1800 (CO₂/N₂) at dilute conditions
  - Uptake: 1.6 mmol/g at 0.04 bar (DAC-relevant)
  - Issue: Very narrow pores → diffusion limitations

- **MOF-177:**
  - High surface area (4500 m²/g)
  - CO₂ uptake: 33 mmol/g at 35 bar, 298K
  - Low selectivity without functionalization

*Functionalized MOFs:*
- **Amine-Functionalized:** -NH₂ groups enhance CO₂ affinity
  - Example: NH₂-MIL-53: 10% uptake increase
- **Ionic Liquids:** Impregnation with [BMIM][BF₄] improves selectivity
- **Defect Engineering:** Controlled defects create additional CO₂ sites

**Recent Computational Discoveries (2024):**

- **CoRE MOF 2019 Database:** 12,020 structures screened
- **ARC-MOF Database:** 279,632 MOFs with DFT-derived charges
- **openDAC23:** 8,000 MOFs with H₂O/CO₂ co-adsorption data
- AI-designed MOFs with 40% faster charge assignment than DFT

### c) Simulation/Computational Approaches

**Grand Canonical Monte Carlo (GCMC):**

*What it does:* Simulates adsorption at fixed μ (chemical potential), V (volume), T (temperature)

*Monte Carlo Moves:*
1. **Insertion:** Add molecule at random position (probability ∝ exp(-U/k_BT))
2. **Deletion:** Remove random molecule
3. **Translation:** Move molecule
4. **Rotation:** Rotate molecule
5. **Acceptance:** Metropolis criterion with μVT ensemble weights

*Detailed Algorithm:*
```
For pressure P:
  μ = μ_ideal(P,T) = k_BT ln(ΛP/k_BT)  # Λ = thermal de Broglie wavelength
  For MC_steps:
    Choose move type (insert/delete/translate/rotate)
    Calculate ΔU (energy change)
    Accept with probability: min(1, exp(-(ΔU - μΔN)/k_BT))
  <N> = time average → adsorption isotherm point n(P,T)
```

*Output:*
- **Adsorption Isotherm:** n(P) at fixed T (typically Langmuir or BET shape)
- **Selectivity:** From mixture GCMC simulations
- **Heat of Adsorption:** ΔH_ads = <U_ads> - <U_gas> (related to desorption energy)

*Typical Parameters:*
- Equilibration: 10⁵-10⁶ steps
- Production: 10⁶-10⁷ steps
- Computational Cost: ~1-10 CPU-hours per pressure point
- Software: RASPA, RASPA2 (open-source standard)

**Force Fields for GCMC:**

*Generic Force Fields:*
- **UFF (Universal Force Field):** Rapid but low accuracy
- **DREIDING:** Better for organic molecules
- **TraPPE (Transferable Potentials for Phase Equilibria):** For CO₂, CH₄, N₂

*CO₂ Model:*
- 3-site model: O=C=O
- Lennard-Jones + electrostatic: U = 4ε[(σ/r)¹² - (σ/r)⁶] + q₁q₂/r
- Partial charges: C = +0.7e, O = -0.35e (typical)

*MOF-Specific:*
- **Framework atoms:** Fixed positions (rigid framework approximation)
- **Charges:** Critical for electrostatic interactions
  - DFT-derived: DDEC (Density Derived Electrostatic and Chemical)
  - Fast ML: 40% faster than DFT, similar accuracy
- **Flexibility:** Advanced simulations include lattice vibrations (rare, expensive)

**Density Functional Theory for MOFs:**

*Challenges:*
- Large unit cells: 100-1000 atoms typical
- Dispersion interactions: van der Waals crucial for CO₂ adsorption
- Cost: Single-point energy ~10³-10⁴ CPU-hours for large MOF

*Applications:*

1. **Charge Assignment:**
   - DDEC method: Partition electron density to atoms
   - Used in GCMC for electrostatic potential
   - Recent ML models predict charges without full DFT (40% speed-up)

2. **Binding Energy:**
   - Place CO₂ at various sites, optimize geometry
   - E_bind = E(MOF+CO₂) - E(MOF) - E(CO₂)
   - Includes dispersion: DFT-D3 correction essential
   - Typical values: -20 to -60 kJ/mol for physisorption

3. **Electronic Structure:**
   - Band gap, conductivity (relevant for electrochemical CO₂ reduction)
   - Stability assessment (HOMO-LUMO gaps)

*Practical Workflow:*
- Full DFT for small representative set (~100 MOFs)
- Train ML models on DFT data
- Predict properties for 10⁵-10⁶ hypothetical MOFs

**Molecular Dynamics for MOFs:**

*Classical MD:*
- Study diffusion: D from mean squared displacement
- Permeability: P = D × S (diffusivity × solubility)
- Rare: most MOF screening assumes equilibrium adsorption

*AIMD (Rare):*
- Framework flexibility, bond breaking (e.g., degradation mechanisms)
- Water-induced structural changes
- Extremely expensive: limited to small MOFs

### d) Where ML Fits In

**Adsorption Property Prediction:**

*Graph Neural Networks:*

**Architecture:**
- Nodes: Atoms (metal centers, organic linkers)
- Edges: Bonds (or spatial proximity cutoff)
- Message passing: Aggregate neighbor information
- Readout: Global pooling → CO₂ uptake, selectivity

**Training Data:**
- CoRE MOF Database: 12K structures with GCMC-computed uptakes
- NIST/ARPA-E databases: Experimental isotherms (smaller, ~10³ samples)

**Performance:**
- MAE for CO₂ uptake: ~1-2 mmol/g (GCMC baseline: ~0.1 mmol/g)
- Selectivity prediction: R² ~0.85-0.90
- Speed: ~10⁻³ seconds per prediction (vs. hours for GCMC)

*Feature Engineering Approaches (Traditional ML):*

**Descriptors:**
1. **Geometric:**
   - Pore size distribution (PSD): from Monte Carlo integration
   - Surface area: Connolly/accessible surface (probe molecule rolling)
   - Void fraction: Vᵥₒᵢ𝒹/Vₜₒₜₐₗ

2. **Chemical:**
   - Functional group counts (-OH, -NH₂, -COOH)
   - Metal node identity (Mg, Zn, Cu...)
   - Linker chemistry (aromatic, aliphatic)

3. **Electronic:**
   - Partial atomic charges (max, min, variance)
   - Heat of adsorption (from initial DFT calculations)

**Models:**
- Random Forest, Gradient Boosting (XGBoost)
- Feature importance: Often pore size + heat of adsorption dominate
- Performance: Comparable to GNNs for smaller datasets

**Multi-Scale Modeling (2024 Advance):**

*Workflow:*
1. **Atomic Scale:** GNN predicts uptake, selectivity
2. **Process Scale:** Integrate into pressure swing adsorption (PSA) model
   - Breakthrough curves, bed sizing, energy consumption
3. **Optimization:** Multi-objective (maximize CO₂ purity, minimize energy)

*Key Insight:*
- Best material at atomic scale ≠ best at process scale
- Example: High selectivity but low capacity → large bed volume → high cost

**Fast Charge Prediction (2024):**

*Problem:* DDEC charges require expensive DFT calculations

*Solution:*
- Train ML models (GNNs) on DFT-DDEC database (ARC-MOF)
- Input: MOF structure (atom positions)
- Output: Partial atomic charges
- Performance: 40% faster than DFT, <5% error in predicted uptakes

*Impact:* Enables rapid GCMC screening without DFT bottleneck

### e) Where Generative AI Fits In

**MOF Generation Landscape:**

The vast chemical space (~10²⁰ possible MOFs from common building blocks) necessitates generative approaches.

**Diffusion Models for MOFs:**

*Framework-Specific Challenges:*
- **Symmetry:** MOFs often have high symmetry (space groups)
- **Bonding Rules:** Metal-linker coordination chemistry constraints
- **Porosity:** Must generate open frameworks (not dense structures)

*Generative Diffusion Workflow:*

1. **Denoising Process:**
   - Forward: Structure → noise (gradual corruption)
   - Reverse: Noise → structure (learned denoising)
   - Timesteps: 1000 typical

2. **Conditional Generation:**
   - Condition on: CO₂ uptake, selectivity, pore size
   - Guidance: Classifier-free guidance scales property strength
   - Example: "Generate MOF with CO₂ uptake > 10 mmol/g at 0.1 bar"

3. **Validation:**
   - GCMC simulation on generated structures
   - DFT geometry optimization (check stability)
   - Success rate: ~40-60% produce stable, porous frameworks

*Recent Work (Communications Chemistry 2023):*
- Generated 120 novel MOFs for CO₂ capture
- 87.5% were chemically valid
- Top candidates showed 15% higher working capacity than existing MOFs

**Reticular Chemistry + ML:**

*SynMOF Database:*
- Automatic extraction of synthesis conditions from 10K+ papers
- ML models predict: solvent, temperature, time, modulator
- Accuracy: >90% for top-3 solvent predictions

*Workflow:*
1. Design MOF with generative model (optimize CO₂ uptake)
2. Predict synthesizability with ML (feasibility score)
3. Predict synthesis conditions (experimental recipe)
4. Experimental validation

*Challenge:* Still a gap between computational screening and lab synthesis
- Only ~10-20% of predicted MOFs successfully synthesized
- Factors: Kinetic stability, competing phases, synthesis expertise

**GFlowNets for MOF Design:**

*Advantages over Diffusion:*
- **Diversity:** Explore multiple distinct solutions (not just mode of distribution)
- **Constraints:** Easily incorporate synthetic feasibility as rewards
- **Interpretability:** Sequential decisions are human-understandable

*MOF-GFlowNet Algorithm:*
1. **State Space:** Partial MOF structures
2. **Actions:**
   - Choose metal node (Mg²⁺, Zn²⁺, Cu²⁺, ...)
   - Choose organic linker (BDC, BTB, BTC, ...)
   - Define topology (pcu, dia, sod, ...)
3. **Reward:**
   - R = f(predicted CO₂ uptake, selectivity, stability)
   - From GNN or GCMC simulations
4. **Training:** Policy maximizes log P(MOF) ∝ R
5. **Generation:** Sample diverse high-reward MOFs

*Performance:*
- Generates 100× more diverse structures than VAEs
- Finds multiple distinct high-performers (different topologies)

**ChatMOF (LLM-Based, 2024):**

*Capabilities:*
- Natural language queries: "Find MOF for CO₂ capture at low pressure"
- Predicts structures from text descriptions
- Generates synthesis procedures

*Architecture:*
- Fine-tuned LLaMA on MOF literature + databases
- Embedding layer converts structures to tokens
- Accuracy: 95.7% prediction, 87.5% generation

*Limitations:*
- Opaque decision-making (black box)
- Limited to chemistries in training data
- Property predictions less accurate than specialized GNNs

**Integrated Generative Workflow:**

```
[Generative Model] → [GCMC Screening] → [Synthesizability ML] → [DFT Validation] → [Experimental Synthesis]
    (10⁴ MOFs)          (10³ promising)      (10² feasible)         (20 stable)          (3-5 samples)
    ~GPU-hours          ~10³ CPU-hours       ~seconds              ~10⁴ CPU-hr          ~months
```

*Current Best Practice (2024):*

1. **Conditional Diffusion/GFlowNet:** Generate 10K MOFs with target CO₂ uptake > 8 mmol/g
2. **Fast ML Charge Prediction:** Assign charges without DFT (minutes per MOF)
3. **GCMC Screening:** Validate uptake, selectivity, working capacity (hours per MOF)
4. **Synthesizability Filter:** ML model predicts feasibility (>0.7 threshold)
5. **Multi-Scale Optimization:** Process-level PSA simulation, rank by $/ton CO₂
6. **DFT Validation:** Top 20 candidates, check thermodynamic stability
7. **Experimental:** Top 3-5, full synthesis + cycling tests

*Success Rate:*
- ~1-3 successfully synthesized per 10K generated
- ~1 outperforms state-of-the-art in real-world testing

---

## Computational Resources Estimate for Hackathon

**Feasible in 1-Day Hackathon:**

### Battery Materials:
✅ **GNN Property Prediction:** Train on Materials Project subset (~10K samples, 1-2 GPU-hours)
✅ **MLFF Inference:** Use pre-trained CHGNet/M3GNet for structure relaxation (CPU/GPU, seconds per structure)
✅ **High-Throughput Screening:** MLFF + GNN pipeline for 1000s of candidates
✅ **Generative Model Fine-Tuning:** Fine-tune DiffCSP/CDVAE on battery-specific subset (4-8 GPU-hours)
❌ **DFT Calculations:** Too slow (unless NVIDIA provides pre-computed database)
❌ **AIMD from Scratch:** Too expensive
❌ **Training Universal MLFF:** Requires weeks on 100s of GPUs

### MOF Carbon Capture:
✅ **GCMC Simulations:** 10-100 MOFs with RASPA (accessible on CPU cluster)
✅ **GNN Adsorption Prediction:** Train on CoRE database (1-2 GPU-hours)
✅ **Generative Model:** Fine-tune diffusion model for CO₂-optimized MOFs (4-8 GPU-hours)
✅ **ML Synthesizability:** Train on SynMOF database (1 GPU-hour)
❌ **DFT Charge Assignment:** Too slow (use pre-trained ML or UFF)
❌ **Process-Level Optimization:** Possible but time-consuming (low priority)

**Recommended Hackathon Strategy:**

1. **Pre-Compute:** Download pre-trained models (M3GNet, CHGNet, MACE, DiffCSP)
2. **Focus:**
   - For batteries: MLFF-based screening + targeted generation
   - For MOFs: GCMC + generative design + synthesizability prediction
3. **Validation:** Use existing databases (Materials Project, CoRE MOF) for benchmarking
4. **Demo:** Interactive tool for property-conditioned generation + rapid screening

---

## Key References for Deep Dive

**Battery Materials:**
- M3GNet: Chen & Ong, Nature Computational Materials (2022)
- CHGNet: Deng et al., Nature Machine Intelligence (2023)
- MACE: Batatia et al., NeurIPS (2022)
- Battery ML Review: Nature Computational Materials (2022)

**MOF Carbon Capture:**
- GCMC Tutorial: Smit & Maesen, Chemical Reviews (2008)
- ML for MOFs: Jablonka et al., Chemical Reviews (2020)
- Generative MOFs: Communications Chemistry (2023)
- CoRE MOF Database: Chung et al., Chemistry of Materials (2019)

**Generative Models:**
- DiffCSP: Jiao et al., NeurIPS (2023)
- MatterGen: Merchant et al., arXiv (2024)
- GFlowNets: Bengio et al., arXiv (2021)
- Crystal-GFN: Mila, ICML (2023)

---

## Appendix: Acronyms

- **AIMD:** Ab Initio Molecular Dynamics
- **CDVAE:** Crystal Diffusion Variational Autoencoder
- **CGCNN:** Crystal Graph Convolutional Neural Network
- **CHGNet:** Crystal Hamiltonian Graph Neural Network
- **DAC:** Direct Air Capture
- **DDEC:** Density Derived Electrostatic and Chemical (charge method)
- **DFT:** Density Functional Theory
- **GCMC:** Grand Canonical Monte Carlo
- **GNN:** Graph Neural Network
- **LGPS:** Li₁₀GeP₂S₁₂
- **LLZO:** Li₇La₃Zr₂O₁₂
- **MACE:** Machine Learning Interatomic Potential (ACE basis)
- **M3GNet:** Materials 3-body Graph Network
- **MLFF:** Machine-Learned Force Field
- **MOF:** Metal-Organic Framework
- **NEB:** Nudged Elastic Band
- **SEI:** Solid Electrolyte Interphase
- **SIB:** Sodium-Ion Battery
- **SSE:** Solid-State Electrolyte
