# System Architecture: Active Generative Discovery for MOF Discovery

This document provides a progressive unpacking of the system architecture, from high-level overview to detailed component designs.

---

## Level 1: System Overview

**The Big Picture**: Three interconnected systems working together for cost-effective MOF discovery.

```mermaid
graph TB
    subgraph "Input Layer"
        DATA[("CRAFTED Dataset<br/>687 MOFs with<br/>CO₂ uptake labels")]
        BUDGET["Budget Constraints<br/>$50-500/iteration"]
    end

    subgraph "Core System"
        AL["Economic Active Learning<br/>(Budget-Constrained ML)"]
        VAE["Conditional VAE<br/>(Generative Model)"]
        GP["Gaussian Process Ensemble<br/>(Surrogate Model)"]
    end

    subgraph "Output Layer"
        DISC["High-Performance<br/>MOF Discoveries"]
        METRICS["Learning Metrics<br/>(Uncertainty Reduction)"]
    end

    DATA --> AL
    BUDGET --> AL
    AL <--> GP
    AL <--> VAE
    VAE --> AL
    GP --> METRICS
    AL --> DISC

    style AL fill:#06A77D,stroke:#333,stroke-width:3px,color:#fff
    style VAE fill:#FF8C00,stroke:#333,stroke-width:3px,color:#fff
    style GP fill:#4169E1,stroke:#333,stroke-width:3px,color:#fff
```

**Key Innovation**: Tight coupling of Active Learning with Generative Discovery enables breaking through baseline performance plateaus.

---

## Level 2: Economic Active Learning Deep Dive

**The AL Loop**: Budget-constrained selection with dual-cost optimization.

```mermaid
graph TB
    subgraph "Initialization"
        INIT_DATA["Initial Training Set<br/>100 random MOFs"]
        POOL["Candidate Pool<br/>587 real MOFs"]
    end

    subgraph "Active Learning Iteration Loop"
        TRAIN["Train GP Ensemble<br/>(5 regressors)"]
        PREDICT["Predict on Pool<br/>(μ, σ²)"]

        subgraph "Acquisition Function"
            COST_EST["Cost Estimator<br/>(validation + synthesis)"]
            ACQ["Compute Scores<br/>• Exploration: σ/cost<br/>• Exploitation: μ×σ/cost"]
        end

        SELECT["Budget-Constrained<br/>Selection<br/>(Greedy Knapsack)"]
        VALIDATE["Validate Selected<br/>MOFs<br/>(Ground Truth)"]
        UPDATE["Update Training Set"]
    end

    subgraph "Termination"
        CONVERGE{"Budget<br/>Exhausted?"}
        RESULTS["Final Model<br/>+ Best MOFs"]
    end

    INIT_DATA --> TRAIN
    POOL --> PREDICT
    TRAIN --> PREDICT
    PREDICT --> COST_EST
    COST_EST --> ACQ
    ACQ --> SELECT
    SELECT --> VALIDATE
    VALIDATE --> UPDATE
    UPDATE --> TRAIN

    SELECT --> CONVERGE
    CONVERGE -->|No| TRAIN
    CONVERGE -->|Yes| RESULTS

    style TRAIN fill:#4169E1,stroke:#333,stroke-width:2px,color:#fff
    style ACQ fill:#06A77D,stroke:#333,stroke-width:2px,color:#fff
    style SELECT fill:#9B59B6,stroke:#333,stroke-width:2px,color:#fff
```

**Budget Compliance**: Greedy knapsack ensures 100% compliance with per-iteration budget constraints.

---

## Level 3: Active Generative Discovery (AGD)

**VAE Integration**: Conditional generation of novel MOFs within the AL loop.

```mermaid
graph TB
    subgraph "Active Learning Components"
        POOL_REAL["Real MOF Pool<br/>(CRAFTED data)"]
        AL_SELECT["AL Selector<br/>(Exploration strategy)"]
    end

    subgraph "Generative Pipeline"
        VAE_TRAIN["Train Conditional VAE<br/>on validated data"]
        VAE_GEN["Generate Novel MOFs<br/>Target: adaptive CO₂"]
        DEDUP["Deduplication<br/>(100% unique)"]
    end

    subgraph "Portfolio Management"
        MERGE["Merge Candidates<br/>(Real + Generated)"]
        CONSTRAINT["Portfolio Constraint<br/>70-85% generated"]
        FINAL_SELECT["Final Selection<br/>(Mixed portfolio)"]
    end

    subgraph "Validation & Feedback"
        VAL["Validate Portfolio"]
        UPDATE_VAE["Update VAE Target<br/>(7.1 → 8.7 → 10.2)"]
        UPDATE_AL["Update AL Model"]
    end

    POOL_REAL --> MERGE
    VAE_TRAIN --> VAE_GEN
    VAE_GEN --> DEDUP
    DEDUP --> MERGE
    MERGE --> CONSTRAINT
    CONSTRAINT --> FINAL_SELECT
    FINAL_SELECT --> VAL
    VAL --> UPDATE_VAE
    VAL --> UPDATE_AL
    UPDATE_VAE --> VAE_GEN
    UPDATE_AL --> AL_SELECT
    AL_SELECT --> POOL_REAL

    style VAE_GEN fill:#FF8C00,stroke:#333,stroke-width:3px,color:#fff
    style CONSTRAINT fill:#9B59B6,stroke:#333,stroke-width:2px,color:#fff
    style VAL fill:#06A77D,stroke:#333,stroke-width:2px,color:#fff
```

**Adaptive Targeting**: VAE target CO₂ uptake increases based on validated discoveries (7.1 → 8.7 → 10.2 mol/kg).

**Key Result**: +26.6% discovery improvement vs baseline (real MOFs only).

---

## Level 4: Surrogate Model & Data Pipeline

**GP Ensemble**: True epistemic uncertainty for informed exploration.

```mermaid
graph TB
    subgraph "Feature Engineering"
        RAW["Raw MOF Data<br/>• Composition<br/>• Structure<br/>• Cost"]
        EXTRACT["Feature Extraction<br/>• cell_a, cell_b, cell_c<br/>• volume<br/>• metal"]
        SCALE["StandardScaler<br/>(numerical stability)"]
    end

    subgraph "GP Ensemble (5 Models)"
        GP1["GP 1<br/>Matern kernel"]
        GP2["GP 2<br/>Matern kernel"]
        GP3["GP 3<br/>Matern kernel"]
        GP4["GP 4<br/>Matern kernel"]
        GP5["GP 5<br/>Matern kernel"]
    end

    subgraph "Uncertainty Quantification"
        COV["Covariance Matrix<br/>(epistemic uncertainty)"]
        PRED_MEAN["Mean Prediction<br/>μ(x)"]
        PRED_STD["Std Deviation<br/>σ(x)"]
    end

    subgraph "Output"
        UNC["Pool Uncertainty<br/>(for AL)"]
        PERF["Performance Prediction<br/>(for exploitation)"]
    end

    RAW --> EXTRACT
    EXTRACT --> SCALE
    SCALE --> GP1
    SCALE --> GP2
    SCALE --> GP3
    SCALE --> GP4
    SCALE --> GP5

    GP1 --> COV
    GP2 --> COV
    GP3 --> COV
    GP4 --> COV
    GP5 --> COV

    COV --> PRED_MEAN
    COV --> PRED_STD
    PRED_MEAN --> PERF
    PRED_STD --> UNC

    style COV fill:#4169E1,stroke:#333,stroke-width:2px,color:#fff
    style PRED_STD fill:#06A77D,stroke:#333,stroke-width:2px,color:#fff
```

**Why GP > RF**:
- GP provides true Bayesian epistemic uncertainty from covariance matrix
- RF only provides ensemble variance (less accurate for exploration)
- GP enables principled uncertainty-based exploration

---

## Level 5: Cost Estimation System

**Dual-Cost Model**: Validation + synthesis costs for realistic budgeting.

```mermaid
graph TB
    subgraph "Input Features"
        MOF["MOF Properties<br/>• Volume<br/>• Metal<br/>• Complexity"]
    end

    subgraph "Cost Components"
        VAL_COST["Validation Cost<br/>$0.01 - $0.10<br/>(characterization)"]
        SYNTH_BASE["Base Synthesis Cost<br/>Metal-dependent"]
        SYNTH_SCALE["Volume Scaling<br/>log(volume) factor"]
        SYNTH_COMPLEX["Complexity Factor<br/>geometry-dependent"]
    end

    subgraph "Cost Computation"
        SYNTH_TOTAL["Total Synthesis Cost<br/>= base × scale × complexity"]
        TOTAL["Total Cost<br/>= validation + synthesis"]
    end

    subgraph "Budget Management"
        ITER_BUDGET["Per-Iteration Budget<br/>$50 (Fig1) or $500 (Fig2)"]
        KNAPSACK["Greedy Knapsack<br/>Maximize value/cost<br/>subject to budget"]
        SELECT["Selected MOFs<br/>(100% compliant)"]
    end

    MOF --> VAL_COST
    MOF --> SYNTH_BASE
    MOF --> SYNTH_SCALE
    MOF --> SYNTH_COMPLEX

    SYNTH_BASE --> SYNTH_TOTAL
    SYNTH_SCALE --> SYNTH_TOTAL
    SYNTH_COMPLEX --> SYNTH_TOTAL

    VAL_COST --> TOTAL
    SYNTH_TOTAL --> TOTAL

    TOTAL --> KNAPSACK
    ITER_BUDGET --> KNAPSACK
    KNAPSACK --> SELECT

    style TOTAL fill:#FFD93D,stroke:#333,stroke-width:2px,color:#333
    style KNAPSACK fill:#9B59B6,stroke:#333,stroke-width:2px,color:#fff
```

**Cost Ranges**:
- Validation: $0.01 - $0.10/sample (characterization)
- Synthesis: $0.10 - $3.00/g (metal-dependent, volume-scaled)
- Average: $0.78/MOF (exploration) vs $2.03/MOF (exploitation)

---

## Level 6: Conditional VAE Architecture

**Generative Model**: Learns latent representations conditioned on target CO₂ uptake.

```mermaid
graph TB
    subgraph "Input"
        METAL["Metal Type<br/>(one-hot encoded)"]
        TARGET["Target CO₂<br/>(scalar)"]
        GEO["Geometry Features<br/>(cell_a, b, c, volume)"]
    end

    subgraph "Encoder"
        ENC_INPUT["Concatenate<br/>[metal | geo]"]
        ENC_H1["Dense Layer 1<br/>(64 units, ReLU)"]
        ENC_H2["Dense Layer 2<br/>(32 units, ReLU)"]
        LATENT["Latent Space<br/>μ, log(σ²)<br/>(4 dims)"]
    end

    subgraph "Latent Sampling"
        Z["Sample z ~ N(μ, σ²)<br/>(reparameterization)"]
    end

    subgraph "Decoder"
        DEC_INPUT["Concatenate<br/>[z | metal | target]"]
        DEC_H1["Dense Layer 1<br/>(32 units, ReLU)"]
        DEC_H2["Dense Layer 2<br/>(64 units, ReLU)"]
        OUTPUT["Output<br/>(cell_a, b, c, volume)"]
    end

    subgraph "Loss Function"
        RECON["Reconstruction Loss<br/>MSE(geo, geo')"]
        KL["KL Divergence<br/>D_KL(q(z|x)||p(z))"]
        TOTAL_LOSS["Total Loss<br/>= MSE + β×KL"]
    end

    METAL --> ENC_INPUT
    GEO --> ENC_INPUT
    ENC_INPUT --> ENC_H1
    ENC_H1 --> ENC_H2
    ENC_H2 --> LATENT
    LATENT --> Z
    Z --> DEC_INPUT
    METAL --> DEC_INPUT
    TARGET --> DEC_INPUT
    DEC_INPUT --> DEC_H1
    DEC_H1 --> DEC_H2
    DEC_H2 --> OUTPUT

    OUTPUT --> RECON
    LATENT --> KL
    RECON --> TOTAL_LOSS
    KL --> TOTAL_LOSS

    style LATENT fill:#FF8C00,stroke:#333,stroke-width:3px,color:#fff
    style Z fill:#FFD93D,stroke:#333,stroke-width:2px,color:#333
    style TOTAL_LOSS fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
```

**Generation Process**:
1. Sample z from N(0, 1)
2. Concatenate [z | metal | target_co2]
3. Decode to geometry features
4. Apply physical constraints (positive values, realistic ranges)
5. Deduplicate against existing MOFs

**Diversity**: 100% unique compositions across all iterations (no duplicates).

---

## Level 7: Complete Data Flow

**End-to-End Pipeline**: From raw data to discovered MOFs.

```mermaid
graph TB
    subgraph "Data Sources"
        CRAFTED[("CRAFTED Database<br/>687 experimental MOFs")]
        COSTS["Estimated Costs<br/>(metal, volume-based)"]
    end

    subgraph "Preprocessing"
        SPLIT["Train/Pool Split<br/>100 train / 587 pool"]
        FEATURES["Feature Engineering<br/>Geometric + Cost"]
    end

    subgraph "Iteration 1"
        GP1["Train GP<br/>(100 MOFs)"]
        ACQ1["Acquisition<br/>(exploration)"]
        SEL1["Select<br/>(budget: $500)"]
        VAL1["Validate<br/>(~12 MOFs)"]
    end

    subgraph "Iteration 2+"
        VAE_TRAIN["Train VAE<br/>(validated data)"]
        VAE_GEN["Generate MOFs<br/>(target: 8.7)"]
        MERGE["Merge Real+Gen"]
        GP2["Retrain GP<br/>(112 MOFs)"]
        ACQ2["Acquisition"]
        SEL2["Select<br/>(70-85% gen)"]
        VAL2["Validate"]
    end

    subgraph "Results"
        BEST["Best MOF Found<br/>11.07 mol/kg"]
        MODEL["Final GP Model<br/>9.3% unc reduction"]
        NOVEL["Novel Structures<br/>51 unique generated"]
    end

    CRAFTED --> SPLIT
    COSTS --> FEATURES
    SPLIT --> FEATURES
    FEATURES --> GP1
    GP1 --> ACQ1
    ACQ1 --> SEL1
    SEL1 --> VAL1
    VAL1 --> VAE_TRAIN
    VAE_TRAIN --> VAE_GEN
    VAE_GEN --> MERGE
    VAL1 --> MERGE
    MERGE --> GP2
    GP2 --> ACQ2
    ACQ2 --> SEL2
    SEL2 --> VAL2
    VAL2 --> BEST
    VAL2 --> MODEL
    VAE_GEN --> NOVEL

    style VAE_TRAIN fill:#FF8C00,stroke:#333,stroke-width:3px,color:#fff
    style BEST fill:#06A77D,stroke:#333,stroke-width:3px,color:#fff
    style MODEL fill:#4169E1,stroke:#333,stroke-width:2px,color:#fff
```

---

## Key Design Decisions

### 1. Why Gaussian Processes?
- **True epistemic uncertainty**: GP covariance provides Bayesian uncertainty estimates
- **Small data regime**: GP excels with limited training data (100-400 samples)
- **Principled exploration**: Uncertainty guides selection, not just ensemble variance

### 2. Why Conditional VAE?
- **Targeted generation**: Condition on desired CO₂ uptake for goal-directed generation
- **Adaptive targeting**: Increase target as better MOFs discovered (7.1 → 10.2 mol/kg)
- **Diversity enforcement**: Latent space sampling ensures 100% unique structures

### 3. Why Portfolio Constraints?
- **Risk management**: 15-30% real MOFs provide ground truth anchor
- **Exploration balance**: 70-85% generated MOFs enable discovery
- **Prevents overfitting**: VAE can't dominate selection entirely

### 4. Why Dual-Cost Model?
- **Realistic budgeting**: Separates validation (cheap) from synthesis (expensive)
- **Metal-aware**: Different metals have vastly different costs
- **Volume-scaled**: Larger MOFs more expensive to synthesize

### 5. Why Exploration > Exploitation?
- **Learning efficiency**: 9.3% uncertainty reduction vs 0.5% (18.6× better)
- **Sample efficiency**: Validates more MOFs (315 vs 120) in same budget
- **Prevents local optima**: Broad sampling vs greedy exploitation

---

## Performance Summary

### Figure 1: Economic AL Ablation
| Method | Uncertainty Reduction | MOFs Validated | Cost | Efficiency |
|--------|----------------------|----------------|------|-----------|
| **Exploration** | +9.3% ✅ | 315 | $246.71 | 0.0377%/$ |
| **Exploitation** | +0.5% | 120 | $243.53 | 0.0021%/$ |
| **Random** | -1.5% ⚠️ | 315 | $247.00 | -0.0061%/$ |
| **Expert** | N/A | 20 | $42.91 | N/A |

### Figure 2: Active Generative Discovery
| Method | Iter 1 | Iter 2 | Iter 3 | Improvement |
|--------|--------|--------|--------|-------------|
| **AGD (Real+Gen)** | 9.03 (R) | 10.43 (G) | 11.07 (G) | **+26.6%** |
| **Baseline (Real)** | 8.75 | 8.75 | 8.75 | 0% (stuck) |

**Key Results**:
- 18.6× better learning with exploration vs exploitation
- +26.6% discovery improvement with generative discovery
- 100% budget compliance across all experiments
- 100% compositional diversity (zero duplicate structures)

---

## Implementation Files

### Core Components
- `src/active_learning.py` - Economic AL framework
- `src/generation/conditional_vae.py` - Conditional VAE model
- `src/cost/estimator.py` - Dual-cost estimation
- `src/models/gp_ensemble.py` - GP ensemble wrapper

### Pipelines
- `run_active_generative_discovery.py` - Main AGD pipeline
- `run_economic_al.py` - Economic AL experiments
- `generate_baseline_for_figure2.py` - Fair baseline comparison

### Visualization
- `src/visualization/figure1_ml_ablation.py` - 4-way comparison
- `src/visualization/figure2_active_generative_discovery.py` - AGD results

---

## References

**Dataset**: CRAFTED - 687 experimental MOFs with CO₂ uptake labels

**Models**:
- Gaussian Process (scikit-learn GaussianProcessRegressor)
- Conditional VAE (PyTorch implementation)

**Optimization**: Greedy knapsack for budget-constrained selection

**Baselines**: Random, Expert (heuristic), AL Exploration, AL Exploitation
