# Streamlit Interactive Demo

**Interactive dashboard for Economic Active Learning**

---

## Features

### 📊 Dashboard Tab
- Key metrics at a glance (25.1% reduction, 62% efficiency, 100% compliance)
- 4-way baseline comparison table
- Visual comparison of all methods

### 🔬 Results & Metrics Tab
- Iteration-by-iteration progress tracking
- Interactive cost tracking plot
- Uncertainty reduction visualization
- Summary statistics (training growth, cost analysis, performance)

### 📈 Figures Tab
- View all publication-quality figures
- Figure selector dropdown
- Download buttons for each figure
- High-resolution display

### 🧬 MOF Explorer Tab
- Filter by metal composition
- Filter by CO₂ uptake range
- Filter by synthesis cost
- Interactive scatter plot: Performance vs Cost
- Searchable data table with all 687 MOFs

### ℹ️ About Tab
- Complete project documentation
- Problem statement and solution
- Technical details
- Results summary
- Innovation highlights

---

## Quick Start

### Run the Dashboard

```bash
# From project root
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Prerequisites

All data should be pre-computed:
1. ✅ `results/baseline_comparison.json` - Baseline results
2. ✅ `results/economic_al_crafted_integration.csv` - AL history
3. ✅ `results/figures/*.png` - All figures
4. ✅ `data/processed/crafted_mofs_co2_with_costs.csv` - MOF data

If missing, run:
```bash
# Generate all results
python run_hackathon_demo.py

# Or run individual tests
python tests/test_economic_al_crafted.py
python tests/test_economic_al_expected_value.py
python tests/test_visualizations.py
```

---

## Usage Tips

### For Presentations

1. **Start with Dashboard tab**: Shows key metrics immediately
   - 25.1% uncertainty reduction
   - 62% sample efficiency
   - 4-way comparison table

2. **Switch to Figures tab**: Walk through Figure 1 and Figure 2
   - Use dropdown to select figures
   - Zoom in if needed
   - Download for presentations

3. **Show MOF Explorer**: Interactive filtering demonstration
   - Filter by metal type (e.g., Zn, Cu, Fe)
   - Show performance vs cost trade-off
   - Demonstrate data exploration capabilities

4. **End with About tab**: Technical details and innovation claims

### For Live Demos

**Option 1: Pre-computed results (safest)**
- Default mode, loads instantly
- All visualizations ready
- No risk of errors during demo

**Option 2: Run new simulation (advanced)**
- Uncheck "Use pre-computed results"
- Adjust budget and iterations
- Run live simulation
- **Warning**: Takes 2-3 minutes, risk of errors

### Customization

**Adjust layout**:
- Use `st.set_page_config(layout="wide")` (already set)
- Modify colors in CSS section at top

**Add new tabs**:
```python
tab6 = st.tabs(["New Tab"])
with tab6:
    st.header("New Content")
```

**Add new metrics**:
```python
st.metric(
    "Your Metric",
    "Value",
    delta="Change",
    help="Description"
)
```

---

## Troubleshooting

### Port Already in Use

```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
streamlit run streamlit_app.py --server.port 8502
```

### Missing Data Files

```bash
# Check which files are missing
ls -la results/baseline_comparison.json
ls -la results/economic_al_crafted_integration.csv
ls -la results/figures/
ls -la data/processed/crafted_mofs_co2_with_costs.csv

# Regenerate everything
python run_hackathon_demo.py
```

### Import Errors

```bash
# Ensure all dependencies installed
uv sync

# Or reinstall
pip install -r requirements.txt
```

### Plots Not Showing

- Check that matplotlib figures are being generated
- Verify `st.pyplot(fig)` is being called
- Try clearing Streamlit cache: `streamlit cache clear`

---

## Keyboard Shortcuts (in browser)

- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show keyboard shortcuts

---

## Deployment

### Local Network Access

```bash
# Allow access from other devices on network
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Streamlit Cloud (Optional)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy `streamlit_app.py`

**Note**: Requires all data files in repo (may exceed size limits)

---

## Performance Tips

1. **Use `@st.cache_data` for expensive operations**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv("large_file.csv")
   ```

2. **Lazy load figures** (only when tab is opened)
   - Currently all figures load on startup
   - Could optimize with session state

3. **Reduce plot size** for faster rendering
   - Adjust `figsize` parameter
   - Use `dpi=150` instead of `dpi=300` for web

---

## Demo Script (5 minutes)

**[30 seconds] Dashboard Tab**
> "Here's our Economic AL system. 25.1% uncertainty reduction, 62% sample efficiency compared to standard AL. You can see the 4-way baseline comparison showing Random actually makes the model worse."

**[90 seconds] Figures Tab**
> "Let me show you our two main figures. Figure 1 demonstrates the ML ablation study - notice how acquisition function choice dramatically impacts learning. Figure 2 shows objective alignment - same discovery outcome with 62% fewer samples when you optimize for the right objective."

**[60 seconds] Results & Metrics Tab**
> "Here's the iteration-by-iteration progress. Every iteration stayed under our $50 budget. Uncertainty steadily decreased over 3 iterations, validating our approach."

**[60 seconds] MOF Explorer**
> "This is our dataset of 687 experimental MOFs. You can filter by metal type, performance range, cost range. Here's the performance vs cost trade-off - higher performance MOFs tend to be more expensive to synthesize, which is why budget-constrained optimization matters."

**[30 seconds] Closing**
> "This is the first budget-constrained active learning system for materials discovery. We've open-sourced everything. Happy to answer questions!"

---

## File Structure

```
streamlit_app.py           # Main application
STREAMLIT_DEMO.md         # This guide
results/
├── baseline_comparison.json    # Required
├── economic_al_crafted_integration.csv  # Required
└── figures/              # Required
    ├── figure1_ml_ablation.png
    ├── figure2_dual_objectives.png
    └── *.png
data/processed/
└── crafted_mofs_co2_with_costs.csv  # Required
```

---

## Support

**Issues?**
- Check that all pre-requisite files exist
- Try regenerating results with `python run_hackathon_demo.py`
- Check Streamlit docs: https://docs.streamlit.io

**Questions?**
- See `HACKATHON_NARRATIVE.md` for presentation talking points
- See `HACKATHON_TROUBLESHOOTING.md` for demo issues

---

**Ready to demo!** 🚀

Just run `streamlit run streamlit_app.py` and explore the dashboard.
