"""
Train GNN Surrogate for MOF CO2 Uptake Prediction

Tests if GNN can improve surrogate generalization to VAE-generated MOFs
compared to Random Forest (r=0.087) and Gaussian Process (r=0.283).
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# Add VAE path
sys.path.insert(0, str(Path(".") / "src" / "generation"))
from dual_conditional_vae import DualConditionalMOFGenerator


class MOF_GNN(nn.Module):
    """
    Graph Neural Network for MOF Property Prediction

    Architecture:
    - 3 graph convolution layers
    - Global mean pooling
    - 2-layer MLP for prediction
    """

    def __init__(self, node_features=5, hidden_dim=64, num_layers=3):
        super().__init__()

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling (graph-level representation)
        x = global_mean_pool(x, batch)

        # Prediction
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x.squeeze()


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    """Evaluate model"""
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            preds.extend(pred.tolist() if pred.dim() > 0 else [pred.item()])
            targets.extend(batch.y.tolist() if batch.y.dim() > 0 else [batch.y.item()])

    return np.array(preds), np.array(targets)


def main():
    print("="*70)
    print("TRAINING GNN SURROGATE FOR MOF CO2 PREDICTION")
    print("="*70 + "\n")

    # Load dataset
    dataset_file = Path("data/processed/gnn_dataset_full.pkl")
    with open(dataset_file, 'rb') as f:
        dataset_info = pickle.load(f)

    graphs = dataset_info['graphs']
    mof_ids = dataset_info['mof_ids']

    print(f"Loaded dataset: {len(graphs)} graphs")
    print(f"  Node features: {dataset_info['num_node_features']}")
    print(f"  Edge features: {dataset_info['num_edge_features']}\n")

    # Train-test split
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    print(f"Split: {len(train_graphs)} train, {len(test_graphs)} test\n")

    # Initialize model
    model = MOF_GNN(
        node_features=dataset_info['num_node_features'],
        hidden_dim=64,
        num_layers=3
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Training
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # Validation
        val_preds, val_targets = evaluate(model, test_loader)
        val_loss = np.mean((val_preds - val_targets) ** 2)
        val_r2 = r2_score(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Val R²={val_r2:.3f}, Val MAE={val_mae:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/gnn_surrogate_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('models/gnn_surrogate_best.pt'))

    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("EVALUATION ON REAL MOFs (Test Set)")
    print(f"{'='*70}\n")

    test_preds, test_targets = evaluate(model, test_loader)
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)

    print(f"Test set performance:")
    print(f"  R²:  {test_r2:.3f}")
    print(f"  MAE: {test_mae:.3f} mol/kg")
    print(f"  RMSE: {np.sqrt(np.mean((test_preds - test_targets) ** 2)):.3f} mol/kg")

    # Compare with previous surrogates
    print(f"\n{'='*70}")
    print("COMPARISON WITH OTHER SURROGATES")
    print(f"{'='*70}\n")

    print("Random Forest (baseline):          R² = 0.234")
    print("RF + Geometric Features:           R² = 0.686")
    print("Gaussian Process + Geom Features:  R² = 0.588")
    print(f"GNN (this model):                  R² = {test_r2:.3f}")

    if test_r2 > 0.686:
        print("\n✅ GNN OUTPERFORMS RF + geometric features!")
    elif test_r2 > 0.588:
        print("\n✓ GNN outperforms GP, comparable to RF + geom")
    else:
        print("\n⚠️  GNN performance lower than expected")

    print(f"\n{'='*70}")
    print("✅ GNN TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nModel saved to: models/gnn_surrogate_best.pt")
    print(f"Ready to test on generated MOFs!")


if __name__ == '__main__':
    main()
