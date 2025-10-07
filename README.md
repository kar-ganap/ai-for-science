# AI for Science Hackathon: MOF Discovery Project

**Active Inverse Design for Synthesizable CO₂-Capturing MOFs**

## Overview

This project combines three advanced ML techniques to address the synthesizability gap in computational materials discovery:

1. **Multi-Objective Optimization**: Balance performance, synthesizability, and model confidence
2. **Active Learning**: Intelligently select which materials to validate, reducing expensive computations
3. **Inverse Design**: Generate novel MOF structures with target properties

## Problem Statement

Current AI approaches can design millions of MOFs (Metal-Organic Frameworks) for carbon capture, but **~90% cannot be synthesized in the lab**. This project builds a system that:
- Generates/screens MOFs with target CO₂ capture performance
- Evaluates trade-offs between performance and synthesizability
- Knows when it's uncertain and requests validation
- Learns from feedback to improve predictions

## Project Structure

```
ai-for-science/
├── README.md                          # This file
├── docs/
│   └── prework/                       # Pre-hackathon research and planning
│       ├── ai4science_hackathon_primer.md           # Technical primer on batteries & MOFs
│       ├── hackathon_project_analysis.md            # Feasibility analysis of project ideas
│       ├── mof_project_tiers.md                     # Baseline/Hard/Ambitious formulations
│       ├── oracle_explanation.md                    # Understanding active learning oracles
│       ├── solo_implementation_guide.md             # Hour-by-hour implementation plan
│       ├── technique_relevance_analysis.md          # Why MOFs > batteries for this approach
│       └── unified_project_concept.md               # Original integrated concept
├── src/                               # Source code (to be created)
│   ├── data/                          # Data loading and preprocessing
│   ├── models/                        # ML models (property prediction, synthesizability)
│   ├── optimization/                  # Multi-objective optimization & Pareto frontier
│   ├── active_learning/               # Active learning loop
│   └── visualization/                 # Plotting and dashboard
├── notebooks/                         # Jupyter notebooks for exploration (to be created)
├── data/                              # Data files (to be created)
│   ├── raw/                           # Original datasets (CoRE MOF, etc.)
│   └── processed/                     # Processed features and labels
├── tests/                             # Unit tests (to be created)
├── app.py                             # Streamlit dashboard (to be created)
└── requirements.txt                   # Python dependencies (to be created)
```

## Key Documents

### Pre-Hackathon Planning

1. **[AI4Science Hackathon Primer](docs/prework/ai4science_hackathon_primer.md)**
   - Technical deep-dive on battery materials and MOF carbon capture
   - Computational methods: DFT, GCMC, ML force fields
   - State-of-the-art approaches and where ML fits in

2. **[Solo Implementation Guide](docs/prework/solo_implementation_guide.md)** ⭐ **START HERE**
   - Hour-by-hour implementation plan for hackathon day
   - Code templates and examples
   - Risk mitigation strategies
   - Pre-hackathon testing scripts

3. **[MOF Project Tiers](docs/prework/mof_project_tiers.md)**
   - Baseline: Multi-objective screening (guaranteed working)
   - Hard: Full pipeline with generation (recommended)
   - Ambitious: Real GCMC validation (high-risk)

4. **[Technique Relevance Analysis](docs/prework/technique_relevance_analysis.md)**
   - Why MOFs are better than batteries for this approach
   - How multi-objective + active learning synergize

5. **[Oracle Explanation](docs/prework/oracle_explanation.md)**
   - Understanding the "oracle" concept in active learning
   - Simulated vs. real validation approaches

## Target Implementation: HARD Version

**Goal**: Complete generative pipeline with multi-objective optimization and active learning

**Timeline**: 6-7 hours
- Hour 1: Foundation (data loading, basic models)
- Hour 2: Multi-objective scoring
- Hour 3: Visualization (3D Pareto frontier)
- Hour 4: Active learning loop ← **BASELINE CHECKPOINT**
- Hour 5: Add generation (CDVAE or simple variants)
- Hour 6-7: Interactive dashboard + presentation

**Key Technologies**:
- PyTorch + PyTorch Geometric
- MatGL (pre-trained materials models)
- Scikit-learn (ensemble methods)
- Streamlit (interactive dashboard)
- Plotly (3D visualization)

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
cd ai-for-science

# Create virtual environment
conda create -n mof-discovery python=3.10
conda activate mof-discovery

# Install dependencies (to be created)
pip install -r requirements.txt

# Download CoRE MOF database
python scripts/download_data.py
```

### Testing Setup

```bash
# Run setup tests
python tests/test_setup.py

# Should see:
# ✅ All imports successful!
# ✅ CoRE MOF: 12000 structures
# ✅ MatGL model loaded
```

## Quick Start

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Run Active Learning Pipeline
```bash
python src/run_active_learning.py --iterations 5 --samples-per-iter 50
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```

## Key Results (Expected)

- **MOFs Evaluated**: 12,000+ (CoRE database)
- **Novel MOFs Generated**: 500-1000
- **Pareto-Optimal Candidates**: 20-50
- **Uncertainty Reduction**: 40-60% over 5 AL iterations
- **Sample Efficiency**: 2-3× better than random sampling

## Hackathon Strategy

### Progressive Enhancement Approach
1. **Build BASELINE first** (Hours 1-4): Multi-objective screening with AL
2. **Add generation if on track** (Hour 5): CDVAE or simple variants
3. **Polish visualization** (Hours 6-7): Dashboard + presentation

### Fallback Plans
- If generation breaks → Stick with screening (still complete story)
- If time runs short → Pre-generated figures as backup
- If demo crashes → Static HTML exports ready

### Success Criteria
- ✅ Working demo (no crashes during presentation)
- ✅ Complete pipeline (generation → scoring → learning)
- ✅ Quantitative results (Pareto frontier, uncertainty metrics)
- ✅ Clear narrative (addresses synthesizability gap)

## Competitive Advantages

### Technical Novelty
- **First integration** of multi-objective + active learning for MOF synthesizability
- **Uncertainty as Pareto objective**: Novel formulation
- **Closed-loop discovery**: Generate → Validate → Learn → Regenerate

### Audience Appeal
- **For ML researchers**: Novel AL application, ensemble uncertainty
- **For materials scientists**: Addresses THE bottleneck (synthesizability gap)
- **For VCs/industry**: Deployment-focused (reduces lab failures)
- **For general audience**: "AI that knows when to ask for help"

### Presentation Angle
> "As an ML practitioner tackling materials science, I focused on the **learning dynamics** rather than domain complexity. The result: a general framework for uncertainty-aware discovery that generalizes beyond MOFs to any expensive validation scenario."

## Resources

### Datasets
- [CoRE MOF Database](https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs): 12,000+ MOFs with properties
- [Materials Project](https://materialsproject.org/): 150,000+ inorganic materials
- SynMOF: Synthesis conditions from literature

### Pre-trained Models
- MatGL / M3GNet: Universal materials property predictor
- CDVAE: Crystal structure generative model

### References
- M3GNet: Chen & Ong, Nature Computational Materials (2022)
- CHGNet: Deng et al., Nature Machine Intelligence (2023)
- DiffCSP: Jiao et al., NeurIPS (2023)
- MOF Synthesizability: Multiple recent papers (2024)

## License

MIT License (or specify your preference)

## Contact

Kartik Ganapathi (add your contact info if desired)

## Acknowledgments

- AI4Science Hackathon organizers
- AGI House, Khosla Ventures, NVIDIA (sponsors)
- Prof. Gábor Csányi (Cambridge)
- Prof. Shyue Ping Ong (UC San Diego)

---

**Last Updated**: October 7, 2025

**Status**: Pre-hackathon preparation phase

**Target**: HARD implementation (full pipeline with generation)
