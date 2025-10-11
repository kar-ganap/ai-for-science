"""
Fast GNN Dataset Builder - Optimized for all 687 MOFs

Uses simpler connectivity algorithm for speed.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.core import Structure
import torch
from torch_geometric.data import Data
import pickle
import warnings
import time
warnings.filterwarnings('ignore')


def structure_to_graph_fast(structure: Structure, label: float, max_neighbors=12) -> Data:
    """
    Fast conversion using distance-based connectivity
    (Faster than CrystalNN)
    """
    # Node features
    node_features = []
    for site in structure:
        element = site.specie
        features = [
            float(element.Z),
            float(element.X) if element.X else 1.5,
            float(element.atomic_radius) if element.atomic_radius else 1.0,
            float(element.group) if element.group else 1.0,
            float(element.row) if element.row else 1.0,
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Distance-based edges (much faster than CrystalNN)
    edge_index = []
    edge_attr = []

    # For each atom, find nearest neighbors
    for i in range(len(structure)):
        # Get distances to all other atoms
        neighbors = structure.get_neighbors(structure[i], 5.0)  # 5 Angstrom cutoff

        # Keep only closest max_neighbors
        neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:max_neighbors]

        for neighbor in neighbors:
            j = neighbor.index
            distance = neighbor.nn_distance

            edge_index.append([i, j])
            edge_attr.append([distance])

    if len(edge_index) == 0:
        # Fallback: at least connect to self
        for i in range(len(structure)):
            edge_index.append([i, i])
            edge_attr.append([0.0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float)
    )

    return graph


def main():
    print("="*70)
    print("FAST GNN DATASET BUILDER (ALL 687 MOFs)")
    print("="*70 + "\n")

    project_root = Path(".")
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    cif_dir = project_root / "data/CRAFTED-2.0.0/CIF_FILES"

    mof_data = pd.read_csv(mof_file)
    print(f"Processing all {len(mof_data)} labeled MOFs\n")

    start_time = time.time()

    graphs = []
    skipped = []
    mof_ids_processed = []

    for idx, mof in mof_data.iterrows():
        mof_id = mof['mof_id']
        label = mof['co2_uptake_mean']

        # Find CIF file
        cif_candidates = list(cif_dir.rglob(f"{mof_id}.cif"))

        if not cif_candidates:
            skipped.append((mof_id, "CIF not found"))
            continue

        try:
            structure = Structure.from_file(str(cif_candidates[0]))
            graph = structure_to_graph_fast(structure, label)

            graphs.append(graph)
            mof_ids_processed.append(mof_id)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(mof_data) - idx - 1) / rate
                print(f"  [{idx+1}/{len(mof_data)}] {len(graphs)} successful | "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        except Exception as e:
            skipped.append((mof_id, str(e)[:50]))

    elapsed = time.time() - start_time
    print(f"\n✓ Complete in {elapsed:.1f}s ({elapsed/len(graphs):.2f}s per MOF)")
    print(f"  Success: {len(graphs)}/{len(mof_data)} MOFs")
    print(f"  Skipped: {len(skipped)}\n")

    # Save
    output_file = project_root / "data/processed/gnn_dataset_full.pkl"

    dataset_info = {
        'graphs': graphs,
        'mof_ids': mof_ids_processed,
        'num_node_features': graphs[0].x.shape[1] if graphs else 0,
        'num_edge_features': graphs[0].edge_attr.shape[1] if graphs else 0,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"✓ Saved to: {output_file}")

    # Stats
    num_nodes = [g.x.shape[0] for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    labels = [g.y.item() for g in graphs]

    print(f"\nDataset statistics:")
    print(f"  Nodes: {np.mean(num_nodes):.1f} ± {np.std(num_nodes):.1f}")
    print(f"  Edges: {np.mean(num_edges):.1f} ± {np.std(num_edges):.1f}")
    print(f"  CO2:   {np.min(labels):.2f} - {np.max(labels):.2f} mol/kg")

    print("\n" + "="*70)
    print("✅ READY FOR GNN TRAINING")
    print("="*70)


if __name__ == '__main__':
    main()
