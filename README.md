# Economic Active Learning for MOF Discovery

**Active Generative Discovery: Portfolio-Constrained VAE-Guided Materials Discovery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates a novel approach to materials discovery that combines **economic active learning** with **conditional generative modeling** to accelerate the discovery of high-performance CO₂-capturing Metal-Organic Frameworks (MOFs).

### Key Innovation

Traditional active learning focuses on either exploration (reduce uncertainty) or exploitation (find best materials). We show that:

1. **Exploration is 18.6× more effective at learning** than exploitation (9.3% vs 0.5% uncertainty reduction)
2. **Generative discovery enables +26.6% breakthrough** improvements over screening real materials alone
3. **Budget constraints matter**: Dual-cost optimization (validation + synthesis) with greedy knapsack selection
4. **Portfolio management**: 70-85% generated MOFs balanced with 15-30% real MOFs prevents overfitting

## Key Results

### Figure 1: Economic Active Learning (ML Ablation)

| Strategy | Uncertainty Reduction | MOFs Validated | Total Cost | Learning Efficiency |
|----------|----------------------|----------------|------------|---------------------|
| **Exploration** ✅ | **+9.3%** | 315 | $246.71 | **0.0377%/$** |
| Exploitation | +0.5% | 120 | $243.53 | 0.0021%/$ |
| Random | -1.5% ⚠️ | 315 | $247.00 | -0.0061%/$ |
| Expert Heuristic | N/A | 20 | $42.91 | N/A |

**Finding**: Exploration-based acquisition is **18.6× better** at learning than exploitation under tight budget constraints ($50/iteration).

### Figure 2: Active Generative Discovery

| Method | Iteration 1 | Iteration 2 | Iteration 3 | Improvement |
|--------|-------------|-------------|-------------|-------------|
| **AGD (Real + Generated)** | 9.03 (R) | 10.43 (G) | **11.15 (G)** | **+26.6%** |
| Baseline (Real only) | 8.75 | 8.75 | 8.75 | 0% (stuck) |

**Finding**: Conditional VAE generation enables breakthrough discoveries. Best MOF (11.15 mol/kg CO₂) is generated with Ti + terephthalic acid linker.

**Additional Metrics**:
- **Compositional Diversity**: 100% unique (47/47 generated MOFs)
- **Chemical Coverage**: 19/20 metal-linker combinations explored (95%)
- **Portfolio Balance**: 72.7% generated MOFs (within 70-85% constraint)

## Project Structure

```
ai-for-science/
├── README.md                          # This file
├── streamlit_app.py                   # Interactive dashboard (http://localhost:8501)
├── pyproject.toml                     # Project configuration (uv)
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── active_learning/               # Economic AL framework
│   │   ├── economic_learner.py        # Main AL loop with budget constraints
│   │   └── acquisition.py             # Exploration vs exploitation strategies
│   ├── models/                        # Surrogate models
│   │   └── gp_ensemble.py             # Gaussian Process ensemble (epistemic uncertainty)
│   ├── generation/                    # Generative models
│   │   └── conditional_vae.py         # Conditional VAE for goal-directed MOF generation
│   ├── cost/                          # Cost estimation
│   │   └── estimator.py               # Dual-cost model (validation + synthesis)
│   ├── validation/                    # Pseudo-validation
│   │   └── pseudo_validation.py       # Oracle for ground truth CO₂ uptake
│   ├── integration/                   # End-to-end pipelines
│   │   └── active_generative_discovery.py  # Full AGD pipeline
│   └── visualization/                 # Publication figures
│       ├── figure1_ml_ablation.py     # 4-panel Economic AL comparison
│       └── figure2_active_generative_discovery.py  # 4-panel AGD results
│
├── scripts/                           # Executable scripts
│   ├── demos/                         # Demo scripts
│   │   └── demo_active_generative_discovery.py  # 3-iteration AGD demo
│   ├── training/                      # Model training
│   ├── analysis/                      # Results analysis
│   ├── data_generation/               # Data generation
│   └── data_preprocessing/            # Data preprocessing
│
├── tests/                             # Test scripts
│   ├── test_economic_al_crafted.py    # Test exploration strategy
│   ├── test_economic_al_expected_value.py  # Test exploitation strategy
│   └── exploratory/                   # Exploratory tests
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System architecture deep-dive
│   ├── HACKATHON_BRIEF.md             # 1-page project summary
│   ├── technical/                     # Technical reports
│   │   ├── PIPELINE_TEST_REPORT.md    # Full pipeline testing results
│   │   └── REORGANIZATION_SUMMARY.md  # File organization changes
│   ├── guides/                        # User guides
│   ├── hackathon/                     # Hackathon materials
│   └── archive/                       # Archived session summaries
│
├── data/                              # Data files (gitignored)
│   └── processed/
│       └── crafted_mofs_co2_with_costs.csv  # CRAFTED dataset (687 MOFs)
│
├── results/                           # Generated results (gitignored)
│   ├── figures/                       # Publication-quality figures
│   │   ├── figure1_ml_ablation.png    # Main Figure 1
│   │   └── figure2_active_generative_discovery.png  # Main Figure 2
│   └── active_generative_discovery_demo/
│       └── demo_results.json          # AGD demo results (3 iterations)
│
└── logs/                              # Log files (gitignored)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-for-science

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Run the Demo

**1. Launch Interactive Dashboard**
```bash
uv run streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**2. Run Economic AL Tests**
```bash
# Test exploration strategy (5 iterations @ $50/iter)
uv run python tests/test_economic_al_crafted.py

# Test exploitation strategy (5 iterations @ $50/iter)
uv run python tests/test_economic_al_expected_value.py
```

**3. Run Active Generative Discovery Demo**
```bash
# Full AGD pipeline (3 iterations @ $500/iter, ~10 minutes)
uv run python scripts/demos/demo_active_generative_discovery.py
```

**4. Generate Publication Figures**
```bash
# Figure 1: Economic AL comparison (4 panels)
uv run python src/visualization/figure1_ml_ablation.py

# Figure 2: AGD results (4 panels)
uv run python src/visualization/figure2_active_generative_discovery.py
```

## System Architecture

### Economic Active Learning

**Innovation**: Budget-constrained acquisition with dual-cost optimization (validation + synthesis).

**Acquisition Strategies**:
- **Exploration**: `uncertainty / cost` - Maximize learning efficiency
- **Exploitation**: `value × uncertainty / cost` - Maximize expected value

**Budget Compliance**: Greedy knapsack algorithm ensures 100% compliance with per-iteration budget constraints.

### Conditional VAE

**Innovation**: Goal-directed MOF generation conditioned on target CO₂ uptake.

**Key Features**:
- **Adaptive targeting**: Target CO₂ increases based on validated discoveries (7.1 → 8.7 → 10.2 mol/kg)
- **Diversity enforcement**: 100% unique compositions across all iterations
- **Physical constraints**: Realistic cell dimensions and volumes

**Architecture**:
- Encoder: [metal | geometry] → latent space (4D)
- Decoder: [latent | metal | target_co2] → geometry features
- Loss: Reconstruction (MSE) + KL divergence

### GP Ensemble

**Innovation**: True epistemic uncertainty from Bayesian covariance matrix.

**Why GP over Random Forest?**
- GP provides true epistemic uncertainty (not just ensemble variance)
- Principled uncertainty quantification for exploration
- Excels in small data regime (100-400 training samples)

### Portfolio Constraints

**Innovation**: Risk management through balanced real/generated portfolios.

**Constraint**: 70-85% generated MOFs, 15-30% real MOFs
- Prevents VAE overfitting
- Maintains ground truth anchor
- Enables breakthrough discoveries while managing risk

## Key Technologies

- **Surrogate Model**: scikit-learn Gaussian Process Regressor (Matérn kernel)
- **Generative Model**: PyTorch Conditional VAE
- **Optimization**: Greedy knapsack (budget-constrained selection)
- **Dashboard**: Streamlit with Plotly visualizations
- **Package Manager**: uv (fast, modern Python package manager)

## Scientific Methodology

### Dataset

**CRAFTED MOF Database**: 687 MOFs with experimental CO₂ uptake labels
- Cell dimensions (a, b, c)
- Cell volume
- Metal type
- Organic linker
- CO₂ uptake (mol/kg @ 298K, 1 bar)

### Cost Model

**Validation Cost**: $0.01 - $0.10/sample (characterization)
**Synthesis Cost**: Metal-dependent base cost × volume scaling × complexity factor
- Precious metals (Au, Pt, Pd): High cost
- Transition metals (Cu, Zn, Fe): Low cost
- Volume scaling: log(volume) factor

**Average Cost**: $0.78/MOF (exploration) vs $2.03/MOF (exploitation)

### Experimental Design

**Figure 1 (Economic AL)**:
- 5 iterations @ $50/iteration
- Initial training: 100 MOFs (random)
- Candidate pool: 587 MOFs
- Metrics: Uncertainty reduction, budget compliance, sample efficiency

**Figure 2 (AGD)**:
- 3 iterations @ $500/iteration
- Initial training: 100 MOFs (random)
- Generation: 14-18 MOFs/iteration (target: adaptive)
- Portfolio constraint: 70-85% generated
- Metrics: Best discovery, compositional diversity, coverage

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Complete system architecture with 7 levels of detail
- **[HACKATHON_BRIEF.md](docs/HACKATHON_BRIEF.md)**: 1-page project summary for presentations
- **[PIPELINE_TEST_REPORT.md](docs/technical/PIPELINE_TEST_REPORT.md)**: Comprehensive testing results (15 minutes, all tests passed)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ganapathi2025economic_al_mof,
  author = {Ganapathi, Kartik},
  title = {Economic Active Learning for MOF Discovery},
  year = {2025},
  url = {https://github.com/yourusername/ai-for-science}
}
```

## References

### Dataset
- **CRAFTED**: Boyd, P. G., et al. "Data-driven design of metal–organic frameworks for wet flue gas CO₂ capture." Nature 576.7786 (2019): 253-256.

### Active Learning
- Lookman, T., et al. "Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design." npj Computational Materials 5.1 (2019): 21.

### Generative Models
- Kingma, D. P., & Welling, M. "Auto-encoding variational bayes." ICLR (2014).
- Xie, T., et al. "Crystal diffusion variational autoencoder for periodic material generation." ICLR (2022).

## License

MIT License

## Contact

Kartik Ganapathi - [Your Email/GitHub]

## Acknowledgments

- AI4Science Hackathon (AGI House, Khosla Ventures, NVIDIA)
- CRAFTED MOF Database (Prof. Tom Daff, University of Edinburgh)
- OpenAI GPT-4 / Anthropic Claude (development assistance)

---

**Last Updated**: October 11, 2025

**Status**: ✅ Complete pipeline with validated results

**Key Finding**: Exploration-based economic active learning achieves 18.6× better learning efficiency than exploitation, and conditional generation enables +26.6% breakthrough discoveries.
