"""
Generate Pseudo-Labels for Unlabeled MOFs using trained GNN

Takes ~7,455 unlabeled CIF files and predicts CO2 uptake using GNN.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.core import Structure
import torch
from torch_geometric.data import Data, DataLoader
import pickle
import warnings
import time
import sys
warnings.filterwarnings('ignore')

# Import the GNN model and data preprocessing
project_root = Path(__file__).resolve().parent
if project_root.name == "data_generation":
    # Running from scripts/data_generation/ after reorganization
    sys.path.insert(0, str(project_root.parent.parent))  # Add project root
    from scripts.training.train_gnn_surrogate import MOF_GNN
    from scripts.data_preprocessing.build_gnn_dataset_fast import structure_to_graph_fast
else:
    # Running from project root (before reorganization)
    sys.path.insert(0, str(project_root))
    from train_gnn_surrogate import MOF_GNN
    from build_gnn_dataset_fast import structure_to_graph_fast


def main():
    print("="*70)
    print("GENERATING PSEUDO-LABELS WITH TRAINED GNN")
    print("="*70 + "\n")

    project_root = Path(".")

    # Load labeled MOF IDs (to exclude)
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    labeled_mofs = pd.read_csv(mof_file)
    labeled_ids = set(labeled_mofs['mof_id'].values)

    print(f"Labeled MOFs to exclude: {len(labeled_ids)}")

    # Find ALL CIF files
    cif_dir = project_root / "data/CRAFTED-2.0.0/CIF_FILES"
    all_cif_files = list(cif_dir.rglob("*.cif"))

    print(f"Total CIF files found: {len(all_cif_files)}")

    # Filter to unlabeled
    unlabeled_cifs = []
    for cif_path in all_cif_files:
        mof_id = cif_path.stem  # Filename without extension
        if mof_id not in labeled_ids:
            unlabeled_cifs.append(cif_path)

    print(f"Unlabeled CIF files: {len(unlabeled_cifs)}\n")

    # For speed, let's use a sample first
    max_samples = 1000  # Process 1000 for now
    if len(unlabeled_cifs) > max_samples:
        print(f"Processing first {max_samples} unlabeled MOFs for speed...\n")
        unlabeled_cifs = unlabeled_cifs[:max_samples]

    # Convert to graphs (without labels)
    print("Converting CIF files to graphs...")
    start_time = time.time()

    graphs = []
    mof_ids = []
    skipped = 0

    for idx, cif_path in enumerate(unlabeled_cifs):
        try:
            structure = Structure.from_file(str(cif_path))
            # Use dummy label 0.0 (we'll replace with prediction)
            graph = structure_to_graph_fast(structure, label=0.0)

            graphs.append(graph)
            mof_ids.append(cif_path.stem)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(unlabeled_cifs) - idx - 1) / rate
                print(f"  [{idx+1}/{len(unlabeled_cifs)}] {len(graphs)} successful | "
                      f"~{remaining:.0f}s remaining")

        except Exception as e:
            skipped += 1

    elapsed = time.time() - start_time
    print(f"\n✓ Converted {len(graphs)} MOFs in {elapsed:.1f}s\n")

    # Load trained GNN
    print("Loading trained GNN model...")
    model = MOF_GNN(node_features=5, hidden_dim=64, num_layers=3)
    model.load_state_dict(torch.load('models/gnn_surrogate_best.pt'))
    model.eval()
    print("✓ Model loaded\n")

    # Generate predictions (pseudo-labels)
    print("Generating pseudo-labels...")
    loader = DataLoader(graphs, batch_size=32)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            if pred.dim() == 0:
                predictions.append(pred.item())
            else:
                predictions.extend(pred.tolist())

    predictions = np.array(predictions)

    print(f"✓ Generated {len(predictions)} pseudo-labels")
    print(f"  Range: {predictions.min():.2f} - {predictions.max():.2f} mol/kg")
    print(f"  Mean:  {predictions.mean():.2f} ± {predictions.std():.2f} mol/kg\n")

    # Save pseudo-labeled dataset
    pseudo_data = pd.DataFrame({
        'mof_id': mof_ids,
        'co2_uptake_pseudo': predictions,
        'source': 'pseudo_labeled'
    })

    output_file = project_root / "data/processed/pseudo_labeled_mofs.csv"
    pseudo_data.to_csv(output_file, index=False)

    print(f"✓ Saved pseudo-labels to: {output_file}")

    # Also save the graphs for potential future use
    pseudo_graphs_file = project_root / "data/processed/pseudo_labeled_graphs.pkl"
    # Update labels in graphs
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([predictions[i]], dtype=torch.float)

    with open(pseudo_graphs_file, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'mof_ids': mof_ids,
            'predictions': predictions
        }, f)

    print(f"✓ Saved graphs to: {pseudo_graphs_file}")

    print("\n" + "="*70)
    print("✅ PSEUDO-LABELING COMPLETE")
    print("="*70)
    print(f"\nNext step: Train RF/GP on {len(labeled_ids)} real + {len(predictions)} pseudo")


if __name__ == '__main__':
    main()
