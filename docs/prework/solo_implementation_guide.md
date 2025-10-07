# Solo Implementation Guide: HARD Version (ML-Focused)

## Your Advantages & Strategy

### Strengths (Play to These):
✅ **ML Background:** Focus on the learning/optimization components
✅ **Solo:** No coordination overhead, faster decisions
✅ **Python/PyTorch:** Core skills for all components

### Weaknesses (Minimize These):
⚠️ **Limited MOF knowledge:** Use pre-built tools, don't customize chemistry
⚠️ **No computational chemistry:** Avoid DFT/force fields, use pre-trained models only
⚠️ **Time-constrained:** Aggressive fallback strategy

### Core Strategy:
1. **Treat MOFs as abstract objects** (don't worry about chemistry details)
2. **Use pre-trained models exclusively** (no training from scratch)
3. **Emphasize the ML pipeline** (your strength)
4. **Polish visualization** (makes up for depth with clarity)
5. **Have working code at every checkpoint** (never break what works)

---

## Pre-Hackathon Preparation (CRITICAL - 6-8 hours)

### Week Before: Software Environment

**Day 1: Core Setup (2 hours)**

```bash
# Create conda environment
conda create -n mof-discovery python=3.10
conda activate mof-discovery

# Core ML libraries
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pymatgen  # Materials structure handling
pip install matgl  # Pre-trained materials models

# Data science
pip install numpy pandas scikit-learn
pip install matplotlib seaborn plotly

# Visualization
pip install streamlit
pip install py3Dmol  # MOF structure visualization
pip install ase  # Atomic structure manipulation

# Utilities
pip install tqdm
pip install joblib
```

**Test Script:**
```python
# test_setup.py
import torch
import torch_geometric
from matgl.ext.pymatgen import Structure2Graph
import streamlit as st
print("✅ All imports successful!")
```

**Day 2-3: Download & Test Datasets (3 hours)**

```python
# download_data.py
import requests
import pandas as pd
from pathlib import Path

# 1. CoRE MOF Database
# Source: https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs
# Download CSV with properties
core_mof_url = "https://raw.githubusercontent.com/gregchung/gregchung.github.io/master/CoRE-MOFs/core-mof-v2.0-ddec.csv"
core_mofs = pd.read_csv(core_mof_url)
core_mofs.to_csv("data/core_mofs.csv")
print(f"✅ CoRE MOF: {len(core_mofs)} structures")

# 2. Pre-computed GCMC results (use CoRE MOF paper data)
# These structures have CO2 uptake at various pressures
# You'll use columns like: 'CO2_0.15bar_298K' (flue gas)

# 3. SynMOF-like data (synthesizability)
# For hackathon: Create proxy from CoRE MOF metadata
# MOFs with experimental data = synthesizable (1)
# MOFs without = uncertain (0.5)
core_mofs['synthesizable'] = core_mofs['reference'].notna().astype(float)
core_mofs['synthesizable'] = core_mofs['synthesizable'].fillna(0.5)

print(f"✅ Synthesizability labels created")
```

**Day 4: Test Pre-trained Models (2-3 hours)**

**Option A: Use MatGL (Recommended for you)**
```python
# test_matgl.py
import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
import pymatgen.core as pmg

# Load pre-trained M3GNet
model = matgl.load_model("M3GNet-MP-2021.2.8-PES")  # Potential Energy Surface

# Test on a simple structure
structure = pmg.Structure(
    lattice=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    species=["Zn", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
)

# Convert to graph and predict
graph_converter = Structure2Graph(element_types=get_element_list(), cutoff=5.0)
graph = graph_converter.get_graph(structure)

# This will predict energy (you'll use as proxy for stability)
energy = model.predict_structure(structure)
print(f"✅ M3GNet working! Energy: {energy}")
```

**Option B: Simple GNN from Scratch (If MatGL fails)**
```python
# simple_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleMOFPredictor(nn.Module):
    def __init__(self, num_features=10, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # Predict CO2 uptake

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# You'll train this on CoRE MOF data if MatGL is too slow
```

**Day 5: Test Generative Model (CRITICAL)**

**Option A: CDVAE (if you can get it working)**
```bash
# Clone CDVAE repo
git clone https://github.com/txie-93/cdvae.git
cd cdvae
pip install -e .

# Download pre-trained checkpoint
# (check their repo for latest)
```

**Option B: Simpler Fallback - Molecule Generation**
```python
# If CDVAE is too complex, use a simpler approach:
# Generate MOFs by mutating existing ones from CoRE database

import numpy as np
from pymatgen.core import Structure

def mutate_mof(original_structure, mutation_strength=0.1):
    """Simple genetic algorithm style mutation"""
    new_structure = original_structure.copy()

    # Perturb atomic positions slightly
    coords = new_structure.cart_coords
    noise = np.random.randn(*coords.shape) * mutation_strength
    new_coords = coords + noise

    # Create new structure
    mutated = Structure(
        lattice=new_structure.lattice,
        species=new_structure.species,
        coords=new_coords,
        coords_are_cartesian=True
    )
    return mutated

# This is less sophisticated but WILL work
# You can frame it as "genetic algorithm guided by ML"
```

**Day 6: Create Test Pipeline (1 hour)**

```python
# test_pipeline.py
"""
End-to-end test of all components
"""

# 1. Load data
mofs = pd.read_csv("data/core_mofs.csv")
print(f"✅ Loaded {len(mofs)} MOFs")

# 2. Load or create structures
# (store as CIF files or pymatgen Structure objects)

# 3. Test property prediction
# predictions = model.predict(structures)
print("✅ Property prediction working")

# 4. Test synthesizability model
# synth_scores = synth_model.predict(structures)
print("✅ Synthesizability prediction working")

# 5. Test Pareto frontier calculation
from scipy.spatial import ConvexHull
# pareto_front = compute_pareto(predictions, synth_scores, confidence)
print("✅ Pareto frontier working")

# 6. Test visualization
import plotly.graph_objects as go
# fig = go.Figure(data=[go.Scatter3d(...)])
print("✅ Visualization working")

print("\n🎉 All components ready for hackathon!")
```

---

## Hackathon Day: Hour-by-Hour Plan

### Pre-Hackathon Morning (if possible)
- ☕ Coffee
- Review your test scripts
- Make sure all imports work
- Have CoRE MOF data loaded

---

### Hour 1: Foundation (10:00 AM - 11:00 AM)

**Goal: Load data, basic property prediction**

**Code Block 1: Data Loading**
```python
# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load CoRE MOF database
mofs_df = pd.read_csv("data/core_mofs.csv")

# Key columns:
# - 'name': MOF identifier
# - 'CO2_0.15bar_298K': CO2 uptake (mmol/g) - YOUR TARGET
# - 'LCD' (Largest Cavity Diameter): geometric property
# - 'PLD' (Pore Limiting Diameter): geometric property
# - 'ASA_m2/g': Accessible surface area
# - 'metal': Metal type (Zn, Cu, etc.)

# Create features for simple model
features = ['LCD', 'PLD', 'ASA_m2/g', 'Density']
X = mofs_df[features].fillna(mofs_df[features].median())
y = mofs_df['CO2_0.15bar_298K'].fillna(0)  # Target: CO2 uptake

# Split for active learning
X_train, X_pool, y_train, y_pool = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_pool is your "oracle pool"

print(f"✅ Train: {len(X_train)}, Oracle Pool: {len(X_pool)}")
```

**Code Block 2: Quick Property Predictor**
```python
from sklearn.ensemble import RandomForestRegressor

# Build ensemble for uncertainty
n_estimators = 5
models = []

for i in range(n_estimators):
    model = RandomForestRegressor(n_estimators=100, random_state=i)
    model.fit(X_train, y_train)
    models.append(model)

print(f"✅ Trained ensemble of {n_estimators} models")

# Predict with uncertainty
def predict_with_uncertainty(X):
    predictions = np.array([m.predict(X) for m in models])
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    return mean_pred, std_pred

# Test
mean, std = predict_with_uncertainty(X_pool)
print(f"✅ Predictions: mean={mean.mean():.2f}, std={std.mean():.2f}")
```

**Checkpoint 1 (11:00 AM):**
- ✅ Data loaded
- ✅ Basic predictor working
- ✅ Uncertainty estimation working

---

### Hour 2: Synthesizability & Multi-Objective (11:00 AM - 12:00 PM)

**Code Block 3: Synthesizability Model**
```python
from sklearn.ensemble import GradientBoostingClassifier

# Create synthesizability labels
# Heuristic: MOFs with experimental references = synthesizable
mofs_df['synthesizable'] = (
    mofs_df['reference'].notna() &
    (mofs_df['metal'].isin(['Zn', 'Cu', 'Mg', 'Co']))  # Common metals
).astype(int)

# Features for synthesizability
synth_features = ['metal', 'LCD', 'Density', 'topology']
# Encode categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_synth = mofs_df.copy()
X_synth['metal_encoded'] = le.fit_transform(X_synth['metal'].fillna('Unknown'))
X_synth['topology_encoded'] = le.fit_transform(X_synth['topology'].fillna('Unknown'))

X_synth_features = X_synth[['metal_encoded', 'topology_encoded', 'LCD', 'Density']].fillna(0)
y_synth = X_synth['synthesizable']

# Train
synth_model = GradientBoostingClassifier(n_estimators=100)
synth_model.fit(X_synth_features, y_synth)

print(f"✅ Synthesizability model trained")
print(f"   Accuracy: {synth_model.score(X_synth_features, y_synth):.2f}")
```

**Code Block 4: Multi-Objective Scoring**
```python
def compute_pareto_frontier(objectives):
    """
    objectives: numpy array (n_samples, n_objectives)
    All objectives assumed to be MAXIMIZED
    Returns: indices of Pareto-optimal points
    """
    is_pareto = np.ones(len(objectives), dtype=bool)

    for i, obj in enumerate(objectives):
        if is_pareto[i]:
            # Check if any other point dominates this one
            is_dominated = np.any(
                np.all(objectives[is_pareto] >= obj, axis=1) &
                np.any(objectives[is_pareto] > obj, axis=1)
            )
            is_pareto[i] = not is_dominated

    return np.where(is_pareto)[0]

# Score all MOFs in pool
performance, uncertainty = predict_with_uncertainty(X_pool)
synthesizability = synth_model.predict_proba(X_synth_features.iloc[X_pool.index])[:, 1]
confidence = 1 / (1 + uncertainty)

# Stack objectives (all to maximize)
objectives = np.column_stack([performance, synthesizability, confidence])

# Compute Pareto frontier
pareto_indices = compute_pareto_frontier(objectives)
print(f"✅ Pareto frontier: {len(pareto_indices)} MOFs")
```

**Checkpoint 2 (12:00 PM - Lunch):**
- ✅ Synthesizability model working
- ✅ Multi-objective scoring working
- ✅ Pareto frontier computed

---

### Hour 3: Visualization (1:00 PM - 2:00 PM)

**Code Block 5: 3D Pareto Frontier**
```python
import plotly.graph_objects as go

def plot_pareto_3d(objectives, pareto_indices, iteration=0):
    """Interactive 3D scatter plot"""
    fig = go.Figure()

    # All points (gray)
    fig.add_trace(go.Scatter3d(
        x=objectives[:, 0],  # Performance
        y=objectives[:, 1],  # Synthesizability
        z=objectives[:, 2],  # Confidence
        mode='markers',
        marker=dict(size=3, color='lightgray', opacity=0.5),
        name='All MOFs',
        hovertemplate='Performance: %{x:.2f}<br>Synth: %{y:.2f}<br>Conf: %{z:.2f}'
    ))

    # Pareto frontier (red)
    fig.add_trace(go.Scatter3d(
        x=objectives[pareto_indices, 0],
        y=objectives[pareto_indices, 1],
        z=objectives[pareto_indices, 2],
        mode='markers',
        marker=dict(size=6, color='red', symbol='diamond'),
        name='Pareto Frontier',
        hovertemplate='Performance: %{x:.2f}<br>Synth: %{y:.2f}<br>Conf: %{z:.2f}'
    ))

    fig.update_layout(
        title=f'Multi-Objective Pareto Frontier (Iteration {iteration})',
        scene=dict(
            xaxis_title='CO₂ Uptake (mmol/g)',
            yaxis_title='Synthesizability',
            zaxis_title='Confidence'
        ),
        width=800,
        height=600
    )

    return fig

# Test
fig = plot_pareto_3d(objectives, pareto_indices)
fig.write_html("pareto_frontier.html")
print("✅ Visualization saved to pareto_frontier.html")
```

**Checkpoint 3 (2:00 PM):**
- ✅ 3D visualization working
- ✅ Can see trade-offs clearly
- **At this point, you have a working BASELINE demo!**

---

### Hour 4: Active Learning Loop (2:00 PM - 3:00 PM)

**Code Block 6: Active Learning**
```python
class ActiveLearner:
    def __init__(self, X_train, y_train, X_pool, y_pool):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()
        self.history = []

    def train_ensemble(self, n_models=5):
        """Train ensemble of models"""
        self.models = []
        for i in range(n_models):
            model = RandomForestRegressor(n_estimators=100, random_state=i)
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def predict_with_uncertainty(self, X):
        predictions = np.array([m.predict(X) for m in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0)

    def select_for_validation(self, n_samples=50, strategy='uncertainty'):
        """Select samples to query from oracle"""
        mean, uncertainty = self.predict_with_uncertainty(self.X_pool)

        if strategy == 'uncertainty':
            # Select highest uncertainty samples
            indices = np.argsort(uncertainty)[-n_samples:]
        elif strategy == 'uncertainty_weighted':
            # Weighted by performance (high predicted + high uncertainty)
            scores = mean * uncertainty
            indices = np.argsort(scores)[-n_samples:]

        return indices

    def query_oracle(self, indices):
        """Simulate oracle query (look up true labels)"""
        X_queried = self.X_pool.iloc[indices]
        y_queried = self.y_pool.iloc[indices]
        return X_queried, y_queried

    def update(self, X_new, y_new):
        """Add validated samples to training set"""
        self.X_train = pd.concat([self.X_train, X_new])
        self.y_train = pd.concat([self.y_train, y_new])

        # Remove from pool
        self.X_pool = self.X_pool.drop(X_new.index)
        self.y_pool = self.y_pool.drop(y_new.index)

    def run_iteration(self, n_samples=50):
        """One active learning iteration"""
        # Train
        self.train_ensemble()

        # Select
        indices = self.select_for_validation(n_samples, strategy='uncertainty_weighted')

        # Query oracle
        X_new, y_new = self.query_oracle(indices)

        # Evaluate before update
        mean_before, std_before = self.predict_with_uncertainty(self.X_pool)

        # Update
        self.update(X_new, y_new)

        # Record metrics
        self.history.append({
            'n_train': len(self.X_train),
            'n_pool': len(self.X_pool),
            'mean_uncertainty': std_before.mean(),
            'max_uncertainty': std_before.max()
        })

        return self.history[-1]

# Initialize
learner = ActiveLearner(X_train, y_train, X_pool, y_pool)

# Run 3-5 iterations
n_iterations = 5
for i in range(n_iterations):
    metrics = learner.run_iteration(n_samples=50)
    print(f"Iteration {i+1}: {metrics}")

print("✅ Active learning complete!")
```

**Checkpoint 4 (3:00 PM):**
- ✅ Active learning loop working
- ✅ Uncertainty decreasing over iterations
- **You now have the core HARD demo working!**

---

### Hour 5: Add Generation (3:00 PM - 4:00 PM)

**DECISION POINT:** Are you on track?
- ✅ **YES** → Proceed with generation
- ⚠️ **NO** → Skip to Hour 6 (visualization), stick with BASELINE

**Code Block 7: Simple MOF Generation (Fallback Method)**
```python
def generate_mof_variants(base_mofs, n_variants=100):
    """
    Generate new MOFs by interpolation/perturbation
    This is a simplified "generative" approach
    """
    variants = []

    # Select high-performing base MOFs
    top_mofs = base_mofs.nlargest(10, 'CO2_0.15bar_298K')

    for _ in range(n_variants):
        # Pick two random MOFs
        idx1, idx2 = np.random.choice(len(top_mofs), 2, replace=False)
        mof1 = top_mofs.iloc[idx1]
        mof2 = top_mofs.iloc[idx2]

        # Interpolate features
        alpha = np.random.uniform(0.3, 0.7)
        new_features = {}
        for feat in features:
            if feat in mof1 and feat in mof2:
                new_features[feat] = alpha * mof1[feat] + (1-alpha) * mof2[feat]

        variants.append(new_features)

    return pd.DataFrame(variants)

# Generate
generated_mofs = generate_mof_variants(mofs_df, n_variants=500)
print(f"✅ Generated {len(generated_mofs)} MOF variants")

# Score them
gen_performance, gen_uncertainty = predict_with_uncertainty(generated_mofs[features])
gen_synth = synth_model.predict_proba(
    prepare_synth_features(generated_mofs)
)[:, 1]

# Add to pool for next AL iteration
```

**Alternative: If CDVAE Works**
```python
# If you got CDVAE working in prep:
from cdvae.model import CDVAE

model = CDVAE.load_from_checkpoint('path/to/checkpoint')

# Generate with target properties
generated_structures = model.sample(
    n_samples=500,
    condition={'property': 'high_co2_uptake'}
)

# Convert to features and score
```

**Checkpoint 5 (4:00 PM):**
- ✅ Generation working (simple or CDVAE)
- ✅ Can score generated MOFs
- **Full HARD pipeline complete!**

---

### Hour 6: Interactive Dashboard (4:00 PM - 5:00 PM)

**Code Block 8: Streamlit App**
```python
# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="MOF Discovery", layout="wide")

st.title("🧪 Active Inverse Design for Synthesizable MOFs")
st.markdown("Generate, score, and validate MOFs for CO₂ capture")

# Sidebar: Controls
st.sidebar.header("Controls")

# Load data (cached)
@st.cache_data
def load_data():
    # Your data loading here
    return mofs_df, learner, objectives, pareto_indices

mofs_df, learner, objectives, pareto_indices = load_data()

# User input
target_uptake = st.sidebar.slider(
    "Target CO₂ Uptake (mmol/g)",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

if st.sidebar.button("🔄 Run Active Learning Iteration"):
    with st.spinner("Running iteration..."):
        metrics = learner.run_iteration(n_samples=50)
        st.sidebar.success(f"Iteration complete! Mean uncertainty: {metrics['mean_uncertainty']:.2f}")

# Main area: 3 columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Multi-Objective Pareto Frontier")

    # 3D plot
    fig = plot_pareto_3d(objectives, pareto_indices, iteration=len(learner.history))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Candidates")

    # Show top Pareto-optimal MOFs
    top_mofs = mofs_df.iloc[pareto_indices].nlargest(10, 'CO2_0.15bar_298K')

    for idx, row in top_mofs.iterrows():
        with st.expander(f"MOF: {row['name']}"):
            st.metric("CO₂ Uptake", f"{row['CO2_0.15bar_298K']:.2f} mmol/g")
            st.metric("Synthesizability", f"{row['synthesizable']:.2f}")
            st.write(f"**Metal:** {row['metal']}")
            st.write(f"**Topology:** {row['topology']}")

# Second row: Metrics
st.subheader("Active Learning Progress")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric(
        "Training Samples",
        len(learner.X_train),
        delta=f"+{50 * len(learner.history)}"
    )

with col4:
    if learner.history:
        current_unc = learner.history[-1]['mean_uncertainty']
        initial_unc = learner.history[0]['mean_uncertainty'] if len(learner.history) > 1 else current_unc
        st.metric(
            "Mean Uncertainty",
            f"{current_unc:.2f}",
            delta=f"{current_unc - initial_unc:.2f}"
        )

with col5:
    st.metric(
        "Pareto-Optimal MOFs",
        len(pareto_indices)
    )

# Learning curve
if learner.history:
    st.subheader("Uncertainty Reduction")

    history_df = pd.DataFrame(learner.history)
    fig_learning = go.Figure()
    fig_learning.add_trace(go.Scatter(
        x=list(range(1, len(history_df)+1)),
        y=history_df['mean_uncertainty'],
        mode='lines+markers',
        name='Mean Uncertainty'
    ))
    fig_learning.update_layout(
        xaxis_title='AL Iteration',
        yaxis_title='Mean Uncertainty (mmol/g)',
        height=300
    )
    st.plotly_chart(fig_learning, use_container_width=True)
```

**Run the app:**
```bash
streamlit run app.py
```

**Checkpoint 6 (5:00 PM):**
- ✅ Interactive dashboard working
- ✅ Can demo live
- ✅ All visualizations ready

---

### Hour 7: Polish & Presentation (5:00 PM - 6:00 PM)

**Code Block 9: Generate Presentation Materials**
```python
# generate_demo_figures.py
"""
Pre-generate all figures for presentation
So demo can't crash during presentation
"""

# Figure 1: Problem illustration
fig1 = create_problem_figure()  # Shows synthesizability gap
fig1.write_html("figs/fig1_problem.html")
fig1.write_image("figs/fig1_problem.png", width=1200, height=600)

# Figure 2: Pipeline diagram
# (Create in PowerPoint or draw.io before hackathon)

# Figure 3: Pareto frontier evolution
fig3 = create_pareto_evolution()  # Shows improvement over AL iterations
fig3.write_html("figs/fig3_pareto_evolution.html")

# Figure 4: Top discoveries
fig4 = create_top_mofs_figure()  # Gallery of best MOFs found
fig4.write_html("figs/fig4_discoveries.html")

# Figure 5: Metrics summary
fig5 = create_metrics_figure()  # Bar charts of improvements
fig5.write_html("figs/fig5_metrics.html")

print("✅ All presentation figures saved!")
```

**Presentation Slides (Create these):**

1. **Title Slide** (30 sec)
   - "Active Inverse Design for Synthesizable MOFs"
   - Your name
   - "Closing the synthesis gap with uncertainty-aware multi-objective optimization"

2. **Problem** (1 min)
   - "90% of AI-designed MOFs can't be synthesized"
   - Show figure: Performance vs. Synthesizability scatter (most in top-left = high perf, low synth)
   - "We need AI that understands reality constraints"

3. **Approach** (1 min)
   - Pipeline diagram with 4 boxes:
     - 1. Generate/Screen MOFs
     - 2. Multi-Objective Scoring (Performance + Synthesizability + Confidence)
     - 3. Active Learning (Validate uncertain predictions)
     - 4. Update & Repeat
   - "Key innovation: Uncertainty as a Pareto objective"

4. **Live Demo** (3-4 min)
   - Open Streamlit app
   - Show initial Pareto frontier
   - Click "Run AL Iteration"
   - Show frontier improvement
   - Highlight top candidates
   - "These 5 MOFs are high-performance, synthesizable, AND we're confident"

5. **Results** (1-2 min)
   - Metrics:
     - "Evaluated X MOFs"
     - "Found Y Pareto-optimal candidates"
     - "Reduced uncertainty by Z%"
     - "Active learning was W% more efficient than random"
   - Show top 3 MOF structures (with predicted properties)

6. **Impact** (30 sec)
   - "This approach:"
   - "✅ Reduces lab failures (no wasted synthesis attempts)"
   - "✅ Builds trust in AI (shows uncertainty)"
   - "✅ Accelerates discovery (focus on what matters)"
   - "Next steps: Partner with experimental lab for validation"

**Checkpoint 7 (6:00 PM):**
- ✅ All code frozen (NO MORE CHANGES)
- ✅ Figures exported
- ✅ Slides ready
- ✅ Demo rehearsed

---

## Risk Mitigation & Fallbacks

### Fallback Decision Tree

```
Hour 4 Checkpoint: Is everything working?
├─ YES → Proceed to generation (Hour 5)
└─ NO → Skip generation, polish BASELINE

Hour 5 Checkpoint: Is generation working?
├─ YES → Integrate into AL loop
├─ PARTIAL → Show as separate component (don't integrate)
└─ NO → Remove from demo, focus on screening

Hour 6: Any crashes?
├─ YES → Use pre-generated figures only (no live demo)
└─ NO → Live demo with pre-generated backup
```

### Emergency Fallback: "Simple but Perfect"

If at Hour 5 things are breaking:

```python
# fallback_demo.py
"""Ultra-simple version that WILL work"""

import pandas as pd
import plotly.express as px

# Just screening with multi-objective
mofs = pd.read_csv("data/core_mofs.csv")

# Score
mofs['performance'] = mofs['CO2_0.15bar_298K']
mofs['synthesizability'] = (mofs['metal'].isin(['Zn', 'Cu'])).astype(float)
mofs['confidence'] = 1.0  # Assume full confidence

# Plot
fig = px.scatter_3d(
    mofs,
    x='performance',
    y='synthesizability',
    z='confidence',
    title="Multi-Objective MOF Screening"
)
fig.show()

# This is BASELINE but it works
```

---

## Key Success Factors

### DO:
✅ **Test everything before hackathon**
✅ **Use pre-trained models** (don't train from scratch)
✅ **Focus on visualization** (makes up for complexity)
✅ **Have working code at every checkpoint**
✅ **Use simple methods** (Random Forest > complex GNN if time-constrained)
✅ **Pre-generate figures** (demo backup if live fails)

### DON'T:
❌ **Don't customize chemistry** (use CoRE MOF as-is)
❌ **Don't train large models** (fine-tuning only, if anything)
❌ **Don't attempt GCMC** (unless you tested it beforehand)
❌ **Don't break working code** (after Hour 6, NO changes)
❌ **Don't debug during demo** (use backup figures)

---

## Your Competitive Advantages as Solo ML Person

1. **Speed:** No coordination overhead
2. **Focus:** Deep on ML techniques (ensemble, AL, optimization)
3. **Polish:** More time for visualization and presentation
4. **Flexibility:** Can pivot quickly without team discussion
5. **Story:** "ML person tackles materials science" is compelling

### Pitch Your Angle:
> "As an ML practitioner, I approached this materials science problem with fresh eyes. I treated MOFs as abstract objects and focused on the learning dynamics. The result: a general framework for uncertainty-aware multi-objective optimization that works across domains."

This frames your lack of domain expertise as an asset (generalizability) rather than weakness.

---

## Final Checklist

### 3 Days Before:
- [ ] All software installed and tested
- [ ] CoRE MOF data downloaded and explored
- [ ] Pre-trained models (MatGL or simple GNN) working
- [ ] Basic visualization tested

### 1 Day Before:
- [ ] Complete test pipeline runs end-to-end
- [ ] Presentation template created
- [ ] Demo script written
- [ ] Sleep well!

### Morning Of:
- [ ] Coffee ☕
- [ ] Review this guide
- [ ] Run test pipeline one more time
- [ ] Breathe - you're prepared!

### During Hackathon:
- [ ] Hour 1: Foundation ✓
- [ ] Hour 2: Multi-objective ✓
- [ ] Hour 3: Visualization ✓
- [ ] **Hour 4: CHECKPOINT - Working BASELINE**
- [ ] Hour 5: Add generation (or skip)
- [ ] Hour 6: Dashboard ✓
- [ ] Hour 7: Present!

---

## You've Got This! 🚀

Your ML background is actually perfect for this:
- Active learning: Core ML technique (you know this)
- Ensemble methods: Standard ML (you know this)
- Multi-objective optimization: ML/optimization (you know this)
- MOF chemistry: Use as black box (you don't need to know this)

**Focus on what you're good at, use tools for the rest.**

Good luck! 🍀
