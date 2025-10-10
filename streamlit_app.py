"""
Economic Active Learning for MOF Discovery - Interactive Dashboard

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.active_learning import EconomicActiveLearner
from src.cost.estimator import MOFCostEstimator
from PIL import Image

# ============================================================
# GLOBAL COLOR SCHEME - Consistent across all visualizations
# ============================================================
METHOD_COLORS = {
    'Random': '#FF4444',           # Red
    'Expert': '#FFA500',           # Orange
    'AL (Exploration)': '#06A77D', # Green
    'AL (Exploitation)': '#9B59B6' # Purple
}

# Page config
st.set_page_config(
    page_title="Economic AL for MOF Discovery",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #06A77D;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text {
        color: #06A77D;
        font-weight: bold;
    }
    .warning-text {
        color: #FF6B6B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧪 Economic Active Learning for MOF Discovery</div>', unsafe_allow_html=True)
st.markdown("**Budget-Constrained Machine Learning for Materials Science**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load pre-computed results option
use_precomputed = st.sidebar.checkbox("Use pre-computed results", value=True,
                                       help="Use existing results for faster loading")

if not use_precomputed:
    st.sidebar.subheader("Run New Simulation")

    # Budget controls
    budget_per_iteration = st.sidebar.slider(
        "Budget per Iteration ($)",
        min_value=10.0,
        max_value=100.0,
        value=50.0,
        step=5.0
    )

    n_iterations = st.sidebar.slider(
        "Number of Iterations",
        min_value=1,
        max_value=5,
        value=3
    )

    strategy = st.sidebar.selectbox(
        "Acquisition Strategy",
        ["cost_aware_uncertainty (Exploration)", "expected_value (Exploitation)"],
        help="Exploration: maximize learning. Exploitation: maximize discovery efficiency"
    )

    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")
else:
    run_simulation = False

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔬 Results & Metrics",
    "📈 Figures",
    "🧬 MOF Explorer",
    "ℹ️ About"
])

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tab1:
    st.header("Economic AL Dashboard")

    # Load baseline comparison
    baseline_file = project_root / "results" / "baseline_comparison.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)

        # Key metrics at the top
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Uncertainty Reduction",
                "25.1%",
                delta="Exploration strategy",
                help="Model improvement over 3 iterations"
            )

        with col2:
            st.metric(
                "Sample Efficiency",
                "62%",
                delta="Fewer samples (72 vs 188)",
                help="Exploitation vs Exploration"
            )

        with col3:
            st.metric(
                "Budget Compliance",
                "100%",
                delta="All iterations under budget",
                help="$50 per iteration × 3"
            )

        with col4:
            st.metric(
                "Total Cost",
                "$148.99",
                delta="188 MOFs validated",
                help="Exploration strategy"
            )

        st.markdown("---")

        # Baseline comparison table
        st.subheader("4-Way Baseline Comparison")

        comparison_data = []
        for method in ['Random', 'Expert', 'AL (Exploration)', 'AL (Exploitation)']:
            if method in baseline_results:
                data = baseline_results[method]
                if method == 'Random':
                    # Random doesn't have total_cost tracked, estimate from average
                    est_cost = data['n_selected_avg'] * 0.79  # avg cost per MOF
                    comparison_data.append({
                        'Method': method,
                        'Samples': f"{data['n_selected_avg']:.0f}",
                        'Cost ($)': f"{est_cost:.2f} (est.)",
                        'Best Performance (mol/kg)': f"{data['best_performance_mean']:.2f} ± {data['best_performance_std']:.2f}",
                        'Uncertainty Reduction (%)': '-1.4% ⚠️'
                    })
                elif method == 'Expert':
                    comparison_data.append({
                        'Method': method,
                        'Samples': str(data['n_selected']),
                        'Cost ($)': f"{data['total_cost']:.2f}",
                        'Best Performance (mol/kg)': f"{data['best_performance']:.2f}",
                        'Uncertainty Reduction (%)': 'N/A'
                    })
                else:
                    unc_red = 25.1 if 'Exploration' in method else 8.6
                    comparison_data.append({
                        'Method': method,
                        'Samples': str(data['n_selected']),
                        'Cost ($)': f"{data['total_cost']:.2f}",
                        'Best Performance (mol/kg)': f"{data['best_performance']:.2f}",
                        'Uncertainty Reduction (%)': f'+{unc_red}% ✅'
                    })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Key insights
        st.info("""
        **Key Insights:**
        - 🎯 **Objective alignment matters**: AL (Exploitation) achieves same result with 62% fewer samples
        - 📉 **Random sampling fails at learning**: -1.4% (model gets worse!)
        - ✅ **Budget constraints work**: 100% compliance across all AL iterations
        - 🔬 **Exploration optimizes learning**: 25.1% uncertainty reduction
        """)
    else:
        st.warning("Baseline comparison results not found. Run baseline tests first.")

# ============================================================
# TAB 2: Results & Metrics
# ============================================================
with tab2:
    st.header("Detailed Results & Metrics")

    # Strategy selector
    strategy_choice = st.radio(
        "Select AL Strategy",
        ["AL (Exploration)", "AL (Exploitation)", "Compare Both"],
        horizontal=True,
        help="Exploration: maximize learning. Exploitation: maximize discovery efficiency"
    )

    # Load AL history files
    exploration_file = project_root / "results" / "economic_al_crafted_integration.csv"
    exploitation_file = project_root / "results" / "economic_al_expected_value.csv"

    exploration_exists = exploration_file.exists()
    exploitation_exists = exploitation_file.exists()

    if strategy_choice == "Compare Both" and exploration_exists and exploitation_exists:
        # Load both strategies
        exp_df = pd.read_csv(exploration_file)
        expt_df = pd.read_csv(exploitation_file)

        st.subheader("Strategy Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### AL (Exploration)")
            st.metric("Total MOFs Validated", int(exp_df['n_validated'].sum()))
            st.metric("Total Cost", f"${exp_df['cumulative_cost'].iloc[-1]:.2f}")
            st.metric("Uncertainty Reduction", f"{((exp_df.iloc[0]['mean_uncertainty'] - exp_df.iloc[-1]['mean_uncertainty']) / exp_df.iloc[0]['mean_uncertainty'] * 100):.1f}%")
            st.metric("Best Performance", f"{exp_df.iloc[-1]['best_predicted_performance']:.2f} mol/kg")

        with col2:
            st.markdown("### AL (Exploitation)")
            st.metric("Total MOFs Validated", int(expt_df['n_validated'].sum()))
            st.metric("Total Cost", f"${expt_df['cumulative_cost'].iloc[-1]:.2f}")
            st.metric("Uncertainty Reduction", f"{((expt_df.iloc[0]['mean_uncertainty'] - expt_df.iloc[-1]['mean_uncertainty']) / expt_df.iloc[0]['mean_uncertainty'] * 100):.1f}%")
            st.metric("Best Performance", f"{expt_df.iloc[-1]['best_predicted_performance']:.2f} mol/kg")

        # Comparison plots
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cost Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            x_exp = range(1, len(exp_df) + 1)
            x_expt = range(1, len(expt_df) + 1)
            ax.bar([x - 0.2 for x in x_exp], exp_df['iteration_cost'],
                   width=0.4, color=METHOD_COLORS['AL (Exploration)'], alpha=0.8,
                   edgecolor='black', linewidth=1.5, label='Exploration')
            ax.bar([x + 0.2 for x in x_expt], expt_df['iteration_cost'],
                   width=0.4, color=METHOD_COLORS['AL (Exploitation)'], alpha=0.8,
                   edgecolor='black', linewidth=1.5, label='Exploitation')
            ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Budget ($50)')
            ax.set_xlabel('Iteration', fontweight='bold')
            ax.set_ylabel('Cost ($)', fontweight='bold')
            ax.set_title('Budget Compliance', fontweight='bold')
            ax.set_xticks(range(1, max(len(exp_df), len(expt_df)) + 1))
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.subheader("Uncertainty Reduction Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(1, len(exp_df) + 1), exp_df['mean_uncertainty'],
                    marker='o', linewidth=3, markersize=10,
                    color=METHOD_COLORS['AL (Exploration)'], label='Exploration')
            ax.plot(range(1, len(expt_df) + 1), expt_df['mean_uncertainty'],
                    marker='s', linewidth=3, markersize=10,
                    color=METHOD_COLORS['AL (Exploitation)'], label='Exploitation')
            ax.set_xlabel('Iteration', fontweight='bold')
            ax.set_ylabel('Mean Uncertainty', fontweight='bold')
            ax.set_title('Learning Progress Comparison', fontweight='bold')
            ax.set_xticks(range(1, max(len(exp_df), len(expt_df)) + 1))
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    else:
        # Single strategy view
        if strategy_choice == "AL (Exploration)" and exploration_exists:
            history_df = pd.read_csv(exploration_file)
            strategy_color = METHOD_COLORS['AL (Exploration)']
            strategy_name = "Exploration"
        elif strategy_choice == "AL (Exploitation)" and exploitation_exists:
            history_df = pd.read_csv(exploitation_file)
            strategy_color = METHOD_COLORS['AL (Exploitation)']
            strategy_name = "Exploitation"
        else:
            st.warning(f"Results for {strategy_choice} not found. Run the Economic AL pipeline first.")
            history_df = None

        if history_df is not None:
            st.subheader(f"Iteration-by-Iteration Progress ({strategy_name})")

            # Format the dataframe for display
            display_df = history_df[[
                'iteration', 'n_validated', 'iteration_cost',
                'cumulative_cost', 'mean_uncertainty', 'best_predicted_performance'
            ]].copy()

            display_df.columns = [
                'Iteration', 'MOFs Validated', 'Cost ($)',
                'Cumulative Cost ($)', 'Mean Uncertainty', 'Best MOF (mol/kg)'
            ]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Plots
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cost Tracking")
                fig, ax = plt.subplots(figsize=(8, 5))
                iterations = range(1, len(history_df) + 1)
                ax.bar(iterations, history_df['iteration_cost'],
                       color=strategy_color, alpha=0.8, edgecolor='black', linewidth=1.5)
                ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Budget ($50)')
                ax.set_xlabel('Iteration', fontweight='bold')
                ax.set_ylabel('Cost ($)', fontweight='bold')
                ax.set_title('Budget Compliance', fontweight='bold')
                ax.set_xticks(iterations)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.subheader("Uncertainty Reduction")
                fig, ax = plt.subplots(figsize=(8, 5))
                iterations = range(1, len(history_df) + 1)
                ax.plot(iterations, history_df['mean_uncertainty'],
                        marker='o', linewidth=3, markersize=10, color=strategy_color)
                ax.set_xlabel('Iteration', fontweight='bold')
                ax.set_ylabel('Mean Uncertainty', fontweight='bold')
                ax.set_title('Learning Progress', fontweight='bold')
                ax.set_xticks(iterations)
                ax.grid(alpha=0.3)

                # Annotate reduction
                initial = history_df.iloc[0]['mean_uncertainty']
                final = history_df.iloc[-1]['mean_uncertainty']
                reduction = (initial - final) / initial * 100
                ax.text(0.5, 0.95, f'{reduction:.1f}% reduction',
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                       ha='center', va='top')
                st.pyplot(fig)

            # Summary stats
            st.subheader("Summary Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Training Set Growth**")
                st.write(f"Initial: {history_df.iloc[0]['n_train']} MOFs")
                st.write(f"Final: {history_df.iloc[-1]['n_train']} MOFs")
                st.write(f"Added: {history_df['n_validated'].sum()} MOFs")

            with col2:
                st.markdown("**Cost Analysis**")
                st.write(f"Total: ${history_df['cumulative_cost'].iloc[-1]:.2f}")
                st.write(f"Avg/iteration: ${history_df['iteration_cost'].mean():.2f}")
                st.write(f"Avg/sample: ${history_df['avg_cost_per_sample'].mean():.2f}")

            with col3:
                st.markdown("**Performance**")
                st.write(f"Best MOF: {history_df.iloc[-1]['best_predicted_performance']:.2f} mol/kg")
                st.write(f"Mean: {history_df.iloc[-1]['mean_predicted_performance']:.2f} mol/kg")
                st.write(f"Uncertainty: {history_df.iloc[-1]['mean_uncertainty']:.3f}")

# ============================================================
# TAB 3: Figures
# ============================================================
with tab3:
    st.header("Publication-Quality Figures")

    figures_dir = project_root / "results" / "figures"

    if figures_dir.exists():
        st.markdown("""
        **Main Results**: These figures demonstrate the key findings from our Economic Active Learning study.
        - **Figure 1**: ML Ablation Study - shows impact of acquisition function choice
        - **Figure 2**: Dual Objectives - demonstrates objective alignment matters
        """)

        st.markdown("---")

        # Figure 1
        st.subheader("Figure 1: ML Ablation Study")
        fig1_file = figures_dir / "figure1_ml_ablation.png"
        if fig1_file.exists():
            # Display with responsive scaling while preserving quality
            st.image(str(fig1_file), use_container_width=True, output_format='PNG')

            with open(fig1_file, "rb") as file:
                st.download_button(
                    label="📥 Download Figure 1 (High Resolution)",
                    data=file,
                    file_name="figure1_ml_ablation.png",
                    mime="image/png",
                    key="download_fig1"
                )
        else:
            st.error("Figure 1 not found")

        st.markdown("---")

        # Figure 2
        st.subheader("Figure 2: Dual Objectives")
        fig2_file = figures_dir / "figure2_dual_objectives.png"
        if fig2_file.exists():
            # Display with responsive scaling while preserving quality
            st.image(str(fig2_file), use_container_width=True, output_format='PNG')

            with open(fig2_file, "rb") as file:
                st.download_button(
                    label="📥 Download Figure 2 (High Resolution)",
                    data=file,
                    file_name="figure2_dual_objectives.png",
                    mime="image/png",
                    key="download_fig2"
                )
        else:
            st.error("Figure 2 not found")
    else:
        st.warning("Figures directory not found. Generate figures first.")

# ============================================================
# TAB 4: MOF Explorer with AL Insights
# ============================================================
with tab4:
    st.header("MOF Explorer with AL Insights")

    # Load CRAFTED data with costs
    mof_file = project_root / "data" / "processed" / "crafted_mofs_co2_with_costs.csv"
    pool_unc_file = project_root / "results" / "pool_uncertainties_initial.csv"

    if mof_file.exists():
        mof_df = pd.read_csv(mof_file)

        # Load pool uncertainties if available
        has_al_data = False
        if pool_unc_file.exists():
            pool_unc_df = pd.read_csv(pool_unc_file)
            has_al_data = True

            # Add AL status to mof_df
            mof_df['in_initial_pool'] = False
            mof_df['uncertainty'] = np.nan
            mof_df['predicted_performance'] = np.nan

            # Mark pool MOFs
            for _, row in pool_unc_df.iterrows():
                idx = row['original_index']
                if idx < len(mof_df):
                    mof_df.loc[idx, 'in_initial_pool'] = True
                    mof_df.loc[idx, 'uncertainty'] = row['uncertainty']
                    mof_df.loc[idx, 'predicted_performance'] = row['predicted_performance']

            # Mark initial training (first 100)
            mof_df['in_initial_training'] = mof_df.index < 100

            # AL Status column
            mof_df['al_status'] = 'Unknown'
            mof_df.loc[mof_df['in_initial_training'], 'al_status'] = 'Initial Training'
            mof_df.loc[mof_df['in_initial_pool'], 'al_status'] = 'Pool (Available)'

        st.subheader(f"CRAFTED Dataset: {len(mof_df)} Experimental MOFs")

        if has_al_data:
            st.info("""
            🎯 **AL Integration Active**: MOFs are labeled by their role in Active Learning
            - **Initial Training** (100 MOFs): Used to train the first model
            - **Pool (Available)** (587 MOFs): Candidates for AL selection, colored by uncertainty
            """)

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if has_al_data:
                al_status_filter = st.multiselect(
                    "AL Status",
                    options=['Initial Training', 'Pool (Available)'],
                    default=None
                )
            else:
                al_status_filter = None

        with col2:
            metal_filter = st.multiselect(
                "Filter by Metal",
                options=sorted(mof_df['metal'].unique()),
                default=None
            )

        with col3:
            co2_min, co2_max = st.slider(
                "CO₂ Uptake Range (mol/kg)",
                float(mof_df['co2_uptake_mean'].min()),
                float(mof_df['co2_uptake_mean'].max()),
                (float(mof_df['co2_uptake_mean'].min()),
                 float(mof_df['co2_uptake_mean'].max()))
            )

        with col4:
            cost_min, cost_max = st.slider(
                "Synthesis Cost Range ($/g)",
                float(mof_df['synthesis_cost'].min()),
                float(mof_df['synthesis_cost'].max()),
                (float(mof_df['synthesis_cost'].min()),
                 float(mof_df['synthesis_cost'].max()))
            )

        # Apply filters
        filtered_df = mof_df.copy()
        if has_al_data and al_status_filter:
            filtered_df = filtered_df[filtered_df['al_status'].isin(al_status_filter)]
        if metal_filter:
            filtered_df = filtered_df[filtered_df['metal'].isin(metal_filter)]
        filtered_df = filtered_df[
            (filtered_df['co2_uptake_mean'] >= co2_min) &
            (filtered_df['co2_uptake_mean'] <= co2_max) &
            (filtered_df['synthesis_cost'] >= cost_min) &
            (filtered_df['synthesis_cost'] <= cost_max)
        ]

        st.write(f"Showing {len(filtered_df)} MOFs")

        # Visualization choice
        viz_choice = st.radio(
            "Visualization",
            ["CO₂ vs Cost", "Uncertainty Map (AL Targets)", "Predicted vs Actual Performance"],
            horizontal=True
        )

        if viz_choice == "CO₂ vs Cost":
            st.subheader("CO₂ Uptake vs Synthesis Cost")
            fig, ax = plt.subplots(figsize=(10, 6))

            if has_al_data:
                # Color by AL status
                training = filtered_df[filtered_df['al_status'] == 'Initial Training']
                pool = filtered_df[filtered_df['al_status'] == 'Pool (Available)']

                if len(training) > 0:
                    ax.scatter(training['synthesis_cost'], training['co2_uptake_mean'],
                              c='blue', alpha=0.6, s=60, edgecolor='black', linewidth=0.5,
                              label='Initial Training')
                if len(pool) > 0:
                    ax.scatter(pool['synthesis_cost'], pool['co2_uptake_mean'],
                              c='orange', alpha=0.6, s=60, edgecolor='black', linewidth=0.5,
                              label='Pool (Available)')
                ax.legend(fontsize=10)
            else:
                scatter = ax.scatter(
                    filtered_df['synthesis_cost'],
                    filtered_df['co2_uptake_mean'],
                    c=filtered_df['volume'],
                    cmap='viridis',
                    alpha=0.6,
                    s=50,
                    edgecolor='black',
                    linewidth=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Volume (Å³)', fontweight='bold')

            ax.set_xlabel('Synthesis Cost ($/g)', fontweight='bold', fontsize=12)
            ax.set_ylabel('CO₂ Uptake (mol/kg)', fontweight='bold', fontsize=12)
            ax.set_title('MOF Performance vs Cost Trade-off', fontweight='bold', fontsize=14)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        elif viz_choice == "Uncertainty Map (AL Targets)" and has_al_data:
            st.subheader("Model Uncertainty Map - AL Targets High Uncertainty MOFs")

            # Filter to pool MOFs only
            pool_df = filtered_df[filtered_df['in_initial_pool']].copy()

            if len(pool_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    pool_df['synthesis_cost'],
                    pool_df['co2_uptake_mean'],
                    c=pool_df['uncertainty'],
                    cmap='Reds',
                    alpha=0.7,
                    s=80,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.set_xlabel('Synthesis Cost ($/g)', fontweight='bold', fontsize=12)
                ax.set_ylabel('CO₂ Uptake (mol/kg)', fontweight='bold', fontsize=12)
                ax.set_title('Pool MOFs: Uncertainty (Red = High Uncertainty = AL Targets)',
                           fontweight='bold', fontsize=14)
                ax.grid(alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Model Uncertainty', fontweight='bold')

                # Annotate top 5 highest uncertainty
                top5 = pool_df.nlargest(5, 'uncertainty')
                for _, row in top5.iterrows():
                    ax.annotate(row['mof_id'],
                               xy=(row['synthesis_cost'], row['co2_uptake_mean']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

                st.pyplot(fig)

                st.caption("🎯 High uncertainty (red) MOFs are prime targets for AL - validating them reduces model uncertainty most.")
            else:
                st.warning("No pool MOFs in filtered selection.")

        elif viz_choice == "Predicted vs Actual Performance" and has_al_data:
            st.subheader("Model Predictions vs Actual Performance")

            pool_df = filtered_df[filtered_df['in_initial_pool']].copy()

            if len(pool_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Perfect prediction line
                min_val = min(pool_df['co2_uptake_mean'].min(), pool_df['predicted_performance'].min())
                max_val = max(pool_df['co2_uptake_mean'].max(), pool_df['predicted_performance'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

                # Scatter with uncertainty as size
                scatter = ax.scatter(
                    pool_df['co2_uptake_mean'],
                    pool_df['predicted_performance'],
                    c=pool_df['uncertainty'],
                    s=pool_df['uncertainty'] * 500,  # Size proportional to uncertainty
                    cmap='Reds',
                    alpha=0.6,
                    edgecolor='black',
                    linewidth=0.5
                )

                ax.set_xlabel('Actual CO₂ Uptake (mol/kg)', fontweight='bold', fontsize=12)
                ax.set_ylabel('Predicted CO₂ Uptake (mol/kg)', fontweight='bold', fontsize=12)
                ax.set_title('Model Prediction Accuracy (Size = Uncertainty)',
                           fontweight='bold', fontsize=14)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=10)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Uncertainty', fontweight='bold')

                st.pyplot(fig)

                st.caption("Larger points = higher uncertainty. Points far from diagonal = prediction errors.")
            else:
                st.warning("No pool MOFs in filtered selection.")

        # Data table
        st.subheader("MOF Data")
        display_cols = ['mof_id', 'metal', 'co2_uptake_mean', 'synthesis_cost']
        if has_al_data:
            display_cols.extend(['al_status', 'uncertainty', 'predicted_performance'])
        display_cols.extend(['cell_a', 'cell_b', 'cell_c', 'volume'])

        # Only show columns that exist
        display_cols = [col for col in display_cols if col in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].sort_values('co2_uptake_mean', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("MOF data not found. Run data preprocessing first.")

# ============================================================
# TAB 5: About
# ============================================================
with tab5:
    st.header("About Economic Active Learning")

    st.markdown("""
    ## The Problem

    Materials discovery is expensive. Synthesizing and testing Metal-Organic Frameworks (MOFs)
    for CO₂ capture costs between $0.10 and $3.00 per sample. Traditional approaches either:
    - Exhaustively search the space (expensive)
    - Rely on expert intuition (limited coverage)

    **Challenge**: How do we learn the most while spending the least?

    ---

    ## Our Solution: Economic Active Learning

    Budget-constrained machine learning that maximizes information gain per dollar.

    ### Key Features:

    1. **Dual-Cost Optimization**
       - Validation cost (experimental testing)
       - Synthesis cost (production)

    2. **Two Acquisition Strategies**
       - **Exploration**: `uncertainty / cost` → Maximize learning
       - **Exploitation**: `value × uncertainty / cost` → Maximize discovery efficiency

    3. **Budget Constraints**
       - Fixed budget per iteration ($50)
       - Greedy knapsack optimization
       - 100% compliance demonstrated

    ---

    ## Results

    ### Exploration Strategy
    - **25.1% uncertainty reduction** over 3 iterations
    - 188 MOFs validated
    - $148.99 total cost
    - Optimizes model improvement

    ### Exploitation Strategy
    - **62% sample efficiency** (72 vs 188 MOFs)
    - Same best performance (9.18 mol/kg)
    - $57.45 total cost
    - Optimizes discovery outcome

    ### Key Insight: Objective Alignment Matters

    Choose the right acquisition function for your goal:
    - Building a better model? → Use exploration
    - Finding best materials efficiently? → Use exploitation

    ---

    ## Technical Details

    **Dataset**: CRAFTED (687 experimental MOFs with CO₂ uptake labels)

    **Model**: Ensemble of 5 RandomForest regressors
    - Epistemic uncertainty via ensemble standard deviation

    **Features**: Geometric properties (cell parameters, volume)

    **Baselines**: 4-way comparison
    - Random (20 trials, multi-trial statistics)
    - Expert (mechanistic heuristic)
    - AL Exploration
    - AL Exploitation

    ---

    ## Innovation

    🎯 **First budget-constrained AL for materials discovery**

    This work introduces:
    1. Dual-cost optimization (validation + synthesis)
    2. Budget constraint enforcement in materials AL
    3. Objective alignment demonstration (62% efficiency gain)
    4. Statistical rigor (multi-trial baselines, Z-scores)

    ---

    ## Repository

    📁 [GitHub Repository](https://github.com/kar-ganap/ai-for-science)

    ## Citation

    If you use this work, please cite:
    ```
    Economic Active Learning for MOF Discovery
    Budget-Constrained Machine Learning for Materials Science
    2025
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Economic Active Learning for MOF Discovery | 2025</p>
    <p>Built with Streamlit • Data from CRAFTED Database</p>
</div>
""", unsafe_allow_html=True)
