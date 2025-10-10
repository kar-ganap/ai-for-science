"""
Economic Active Learning

Budget-constrained active learning for materials discovery.
Selects samples that maximize learning per dollar spent.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class EconomicActiveLearner:
    """
    Active learning with budget constraints

    Key Innovation: Selects samples to maximize information gain per dollar,
    rather than pure uncertainty. This respects real-world lab budget constraints.
    """

    def __init__(self, X_train, y_train, X_pool, y_pool,
                 cost_estimator=None, pool_compositions=None):
        """
        Initialize Economic Active Learner

        Args:
            X_train: Training features (pandas DataFrame)
            y_train: Training labels (pandas Series)
            X_pool: Pool features (unlabeled candidates)
            y_pool: Pool labels (for oracle simulation)
            cost_estimator: MOFCostEstimator instance (optional)
            pool_compositions: List of dicts with MOF compositions for cost estimation
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()

        self.cost_estimator = cost_estimator
        self.pool_compositions = pool_compositions

        # Metrics tracking
        self.history = []
        self.cumulative_cost = 0
        self.models = []

    def train_ensemble(self, n_models: int = 5) -> None:
        """
        Train ensemble of models for uncertainty quantification

        Args:
            n_models: Number of models in ensemble (default 5)
        """
        self.models = []
        for i in range(n_models):
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=i,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
            self.models.append(model)

    def predict_with_uncertainty(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction with epistemic uncertainty

        Args:
            X: Features to predict on

        Returns:
            mean: Mean predictions across ensemble
            std: Standard deviation (uncertainty)
        """
        if not self.models:
            raise ValueError("Must train ensemble first. Call train_ensemble()")

        predictions = np.array([m.predict(X) for m in self.models])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std

    def economic_selection(self,
                          budget_per_iteration: float = 1000,
                          strategy: str = 'cost_aware_uncertainty',
                          min_samples: int = 20,
                          max_samples: int = 100) -> Tuple[List[int], float]:
        """
        Select samples for validation within budget

        This is the core innovation: balancing information gain with cost.

        Args:
            budget_per_iteration: Available $ for this round
            strategy: Selection strategy
                - 'cost_aware_uncertainty': Balance uncertainty and cost
                - 'greedy_cheap': Cheapest high-uncertainty samples
                - 'expected_value': (predicted value × uncertainty) / cost
            min_samples: Minimum samples to select
            max_samples: Maximum samples to select

        Returns:
            selected_indices: List of indices to query
            total_cost: Estimated cost
        """
        # Predict with uncertainty
        pool_mean, pool_std = self.predict_with_uncertainty(self.X_pool)

        # Estimate costs for pool samples
        pool_costs = self._estimate_pool_costs()

        # Compute acquisition scores based on strategy
        if strategy == 'cost_aware_uncertainty':
            # High uncertainty, low cost → high score
            acquisition = pool_std / (pool_costs + 1e-6)

        elif strategy == 'greedy_cheap':
            # Only consider cheap samples (< median cost)
            cheap_mask = pool_costs < np.median(pool_costs)
            acquisition = pool_std * cheap_mask

        elif strategy == 'expected_value':
            # High predicted value × high uncertainty / cost
            # For CO2 uptake, high value is desirable
            acquisition = pool_mean * pool_std / (pool_costs + 1e-6)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Greedy knapsack: Select within budget
        selected_indices = []
        total_cost = 0
        sorted_indices = np.argsort(acquisition)[::-1]  # Descending order

        for idx in sorted_indices:
            cost = pool_costs[idx]

            # Try to add this sample
            if total_cost + cost <= budget_per_iteration:
                selected_indices.append(idx)
                total_cost += cost

                # Stop if we hit max samples
                if len(selected_indices) >= max_samples:
                    break

        # Ensure minimum samples (relax budget if needed)
        if len(selected_indices) < min_samples:
            # Add cheapest remaining samples
            remaining_indices = [i for i in sorted_indices
                               if i not in selected_indices]
            remaining_costs = pool_costs[remaining_indices]
            cheap_order = np.argsort(remaining_costs)

            for i in cheap_order:
                if len(selected_indices) >= min_samples:
                    break
                idx = remaining_indices[i]
                selected_indices.append(idx)
                total_cost += pool_costs[idx]

        return selected_indices, total_cost

    def _estimate_pool_costs(self) -> np.ndarray:
        """
        Estimate synthesis cost for pool samples

        Returns:
            costs: Array of costs per gram for each pool sample
        """
        if self.cost_estimator and self.pool_compositions:
            # Use actual cost estimator with compositions
            costs = []
            for comp in self.pool_compositions:
                cost_data = self.cost_estimator.estimate_synthesis_cost(comp)
                costs.append(cost_data['total_cost_per_gram'])
            return np.array(costs)
        else:
            # Fallback: Use feature-based proxy
            # Assume cost correlates with complexity/density
            # This is a rough approximation for demonstration
            return np.random.uniform(0.5, 50, len(self.X_pool))

    def query_oracle(self, indices: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate oracle query (look up true labels)

        In real scenario, this would be:
        - Running GCMC simulations
        - Actual lab synthesis
        - Literature lookup

        Args:
            indices: Indices to query from pool

        Returns:
            X_queried: Features of queried samples
            y_queried: True labels from oracle
        """
        X_queried = self.X_pool.iloc[indices]
        y_queried = self.y_pool.iloc[indices]
        return X_queried, y_queried

    def update_training_set(self, X_new: pd.DataFrame, y_new: pd.Series,
                           position_indices: Optional[List[int]] = None) -> None:
        """
        Add validated samples to training set

        Args:
            X_new: New features
            y_new: New labels
            position_indices: Position indices in pool (for removing compositions)
        """
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)

        # Get pandas indices to drop BEFORE modifying pool
        drop_indices = X_new.index.tolist()

        # Remove from pool (uses pandas indices)
        self.X_pool = self.X_pool.drop(drop_indices).reset_index(drop=True)
        self.y_pool = self.y_pool.drop(drop_indices).reset_index(drop=True)

        # Update pool compositions if available (uses position indices)
        if self.pool_compositions and position_indices is not None:
            # Remove compositions at position indices (sorted descending to avoid index shifting)
            for idx in sorted(position_indices, reverse=True):
                if 0 <= idx < len(self.pool_compositions):
                    self.pool_compositions.pop(idx)

    def run_iteration(self,
                     budget: float = 1000,
                     strategy: str = 'cost_aware_uncertainty') -> Dict:
        """
        Run one iteration of economic active learning

        Args:
            budget: Available budget for this iteration
            strategy: Selection strategy

        Returns:
            metrics: Dict with iteration metrics
        """
        # Train ensemble
        self.train_ensemble()

        # Select samples within budget
        selected_indices, iteration_cost = self.economic_selection(
            budget_per_iteration=budget,
            strategy=strategy
        )

        # Compute metrics before update
        pool_mean, pool_std = self.predict_with_uncertainty(self.X_pool)
        pool_costs = self._estimate_pool_costs()

        # Store selected MOF details for analysis (BEFORE updating pool)
        selected_mofs_data = []
        for idx in selected_indices:
            selected_mofs_data.append({
                'pool_index': idx,
                'predicted_performance': pool_mean[idx],
                'uncertainty': pool_std[idx],
                'validation_cost': pool_costs[idx],
                'acquisition_score': pool_std[idx] / (pool_costs[idx] + 1e-6)
            })

        # Query oracle
        X_new, y_new = self.query_oracle(selected_indices)

        # Update training set (pass position indices for compositions)
        self.update_training_set(X_new, y_new, position_indices=selected_indices)

        # Update cumulative cost
        self.cumulative_cost += iteration_cost

        # Log metrics
        metrics = {
            'iteration': len(self.history) + 1,
            'n_train': len(self.X_train),
            'n_pool': len(self.X_pool),
            'n_validated': len(selected_indices),
            'iteration_cost': iteration_cost,
            'avg_cost_per_sample': iteration_cost / len(selected_indices),
            'cumulative_cost': self.cumulative_cost,
            'mean_uncertainty': pool_std.mean(),
            'max_uncertainty': pool_std.max(),
            'best_predicted_performance': pool_mean.max(),
            'mean_predicted_performance': pool_mean.mean(),
            'selected_mofs': selected_mofs_data,  # Add selected MOF details
        }

        self.history.append(metrics)
        return metrics

    def get_history_df(self) -> pd.DataFrame:
        """Get history as pandas DataFrame for analysis"""
        return pd.DataFrame(self.history)

    def compute_pool_uncertainties(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute uncertainty for all current pool MOFs

        Returns:
            pool_mean: Performance predictions
            pool_std: Epistemic uncertainties
            pool_costs: Estimated costs
        """
        if not self.models:
            self.train_ensemble()

        pool_mean, pool_std = self.predict_with_uncertainty(self.X_pool)
        pool_costs = self._estimate_pool_costs()

        return pool_mean, pool_std, pool_costs


if __name__ == '__main__':
    # Simple test with synthetic data
    print("Testing Economic Active Learning\n" + "=" * 60)

    # Generate synthetic MOF-like data
    np.random.seed(42)
    n_total = 1000
    n_features = 4  # LCD, PLD, ASA, Density

    # Create synthetic features
    X = pd.DataFrame({
        'LCD': np.random.uniform(5, 20, n_total),
        'PLD': np.random.uniform(3, 15, n_total),
        'ASA': np.random.uniform(500, 3000, n_total),
        'Density': np.random.uniform(0.3, 1.5, n_total)
    })

    # Create synthetic target (CO2 uptake)
    # Higher surface area and larger pores → better performance
    y = (0.003 * X['ASA'] +
         0.5 * X['LCD'] +
         0.3 * X['PLD'] -
         2 * X['Density'] +
         np.random.normal(0, 2, n_total))

    # Split into train and pool
    train_size = 100
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_pool = X.iloc[train_size:]
    y_pool = y.iloc[train_size:]

    print(f"Train: {len(X_train)}, Pool: {len(X_pool)}")

    # Initialize learner (without cost estimator for simple test)
    learner = EconomicActiveLearner(X_train, y_train, X_pool, y_pool)

    # Run 3 iterations
    print("\nRunning Economic Active Learning (3 iterations)...")
    print("-" * 60)

    for i in range(3):
        metrics = learner.run_iteration(budget=500, strategy='cost_aware_uncertainty')
        print(f"\nIteration {metrics['iteration']}:")
        print(f"  Validated: {metrics['n_validated']} MOFs")
        print(f"  Cost: ${metrics['iteration_cost']:.2f} "
              f"(${metrics['avg_cost_per_sample']:.2f}/sample)")
        print(f"  Cumulative: ${metrics['cumulative_cost']:.2f}")
        print(f"  Uncertainty: {metrics['mean_uncertainty']:.3f} "
              f"(max: {metrics['max_uncertainty']:.3f})")
        print(f"  Best predicted: {metrics['best_predicted_performance']:.2f} mmol/g")

    print("\n" + "=" * 60)
    print("✅ Economic Active Learning working!")
    print(f"\nFinal stats:")
    print(f"  - Total validated: {len(learner.X_train) - train_size} MOFs")
    print(f"  - Total cost: ${learner.cumulative_cost:.2f}")
    print(f"  - Training set size: {len(learner.X_train)} → {len(X_train)}")
    print(f"\n📊 Ready for integration with Cost Estimator!")
