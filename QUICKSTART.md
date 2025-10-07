# Quick Start Guide

## Pre-Hackathon Setup (Do This First!)

### 1. Environment Setup (30 minutes)

```bash
# Navigate to project
cd /Users/kartikganapathi/Documents/Personal/random_projects/ai-for-science

# Create conda environment
conda create -n mof-discovery python=3.10
conda activate mof-discovery

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon Macs, you might need:
# conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### 2. Download Data (30 minutes)

```bash
# Create download script
mkdir -p scripts
```

Download CoRE MOF database manually:
1. Go to: https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs
2. Download `core-mof-v2.0-ddec.csv`
3. Save to `data/raw/core_mofs.csv`

### 3. Test Setup (15 minutes)

```python
# tests/test_setup.py
import torch
import torch_geometric
from matgl.ext.pymatgen import Structure2Graph
import pymatgen.core as pmg
import pandas as pd

print("✅ Testing imports...")
assert torch.cuda.is_available() or True  # Works with or without GPU
print("✅ PyTorch OK")

print("✅ Testing data...")
df = pd.read_csv("data/raw/core_mofs.csv")
print(f"✅ Loaded {len(df)} MOFs")

print("✅ All systems go!")
```

Run test:
```bash
python tests/test_setup.py
```

### 4. Review Documentation (1 hour)

**Essential Reading Order:**

1. **[Solo Implementation Guide](docs/prework/solo_implementation_guide.md)** ⭐ **READ THIS FIRST**
   - Complete hour-by-hour plan
   - All code templates included
   - Risk mitigation strategies

2. **[MOF Project Tiers](docs/prework/mof_project_tiers.md)**
   - Understand BASELINE vs HARD vs AMBITIOUS
   - Know your fallback options

3. **Skim the technical primer** (optional, for context):
   - [AI4Science Hackathon Primer](docs/prework/ai4science_hackathon_primer.md)

## Hackathon Day Strategy

### Morning (Before Hacking Starts)

```bash
# Pull latest code
git pull

# Activate environment
conda activate mof-discovery

# Quick test
python tests/test_setup.py

# Open your implementation guide
open docs/prework/solo_implementation_guide.md
```

### Hour-by-Hour Checklist

- [ ] **Hour 1 (10-11 AM)**: Data loading + basic models
  - Load CoRE MOF data
  - Train Random Forest ensemble
  - Test prediction with uncertainty

- [ ] **Hour 2 (11-12 PM)**: Multi-objective
  - Synthesizability model
  - 3-objective scoring
  - Pareto frontier calculation

- [ ] **Hour 3 (12-1 PM)**: Visualization
  - 3D Plotly scatter
  - Pareto frontier plot
  - **LUNCH BREAK**

- [ ] **Hour 4 (1-2 PM)**: Active Learning
  - Implement AL loop
  - Test with simulated oracle
  - **CHECKPOINT: Working BASELINE demo** ✅

- [ ] **Hour 5 (2-3 PM)**: Generation (if on track)
  - CDVAE or simple variants
  - OR skip and polish BASELINE

- [ ] **Hour 6 (3-4 PM)**: Dashboard
  - Streamlit app
  - Interactive controls
  - Pre-generate backup figures

- [ ] **Hour 7 (4-5 PM)**: Presentation
  - **STOP CODING at 4:30 PM**
  - Polish slides
  - Rehearse demo

### Emergency Fallbacks

**If at Hour 4 things aren't working:**
```python
# Use this ultra-simple version
import pandas as pd
import plotly.express as px

mofs = pd.read_csv("data/raw/core_mofs.csv")
mofs['performance'] = mofs['CO2_0.15bar_298K']
mofs['synth'] = (mofs['metal'].isin(['Zn', 'Cu'])).astype(float)

fig = px.scatter_3d(mofs, x='performance', y='synth', z='Density')
fig.show()
```

## Key Files to Create During Hackathon

### src/data/loader.py
```python
import pandas as pd

def load_core_mof():
    """Load CoRE MOF database"""
    return pd.read_csv("data/raw/core_mofs.csv")
```

### src/models/predictor.py
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class EnsemblePredictor:
    def __init__(self, n_models=5):
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=i)
            for i in range(n_models)
        ]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict_with_uncertainty(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)
```

### src/optimization/pareto.py
```python
import numpy as np

def compute_pareto_frontier(objectives):
    """Find Pareto-optimal points"""
    is_pareto = np.ones(len(objectives), dtype=bool)
    for i, obj in enumerate(objectives):
        if is_pareto[i]:
            # Dominated by any other Pareto point?
            is_pareto[i] = not np.any(
                np.all(objectives[is_pareto] >= obj, axis=1) &
                np.any(objectives[is_pareto] > obj, axis=1)
            )
    return np.where(is_pareto)[0]
```

### app.py (Streamlit Dashboard)
```python
import streamlit as st
import plotly.graph_objects as go

st.title("🧪 MOF Discovery Platform")

# Your code here
```

## Resources Checklist

**Before Hackathon:**
- [ ] Environment installed and tested
- [ ] CoRE MOF data downloaded
- [ ] Reviewed solo implementation guide
- [ ] Code templates ready to copy
- [ ] Backup figures strategy in place

**During Hackathon:**
- [ ] Keep solo_implementation_guide.md open
- [ ] Commit frequently (`git commit -am "Hour X: feature"`)
- [ ] Test after each component
- [ ] Have fallback ready

**Presentation:**
- [ ] Demo video recorded (backup)
- [ ] Slides with screenshots
- [ ] 5-minute pitch practiced

## Troubleshooting

### "ModuleNotFoundError: No module named 'matgl'"
```bash
pip install matgl
# Or skip MatGL and use simple sklearn models
```

### "FileNotFoundError: data/raw/core_mofs.csv"
Download from: https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs

### "Streamlit won't start"
```bash
streamlit run app.py --server.port 8501
```

### "Everything is breaking"
→ Revert to BASELINE fallback (see solo_implementation_guide.md Hour 5+)

## Success Metrics

**You're on track if:**
- ✅ Hour 1: Data loads, predictions work
- ✅ Hour 2: Pareto frontier computed
- ✅ Hour 3: 3D plot renders
- ✅ Hour 4: AL loop completes (BASELINE working)
- ✅ Hour 6: Dashboard launches

**Red flags:**
- ❌ Hour 2: Still debugging imports → Use simpler dependencies
- ❌ Hour 4: No working demo → Abandon generation, focus on polish
- ❌ Hour 6: Dashboard crashes → Use pre-generated HTML figures

## Contact & Support

During hackathon:
- Check documentation first: `docs/prework/`
- Simplify if stuck (BASELINE is enough to win!)
- Pre-generated figures are your friend

Good luck! 🚀
