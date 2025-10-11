"""
Build GNN Dataset from CRAFTED CIF Files

Processes 687 MOFs with CO2 uptake labels, converts to PyTorch Geometric graphs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
import torch
from torch_geometric.data import Data
import pickle
import warnings
warnings.filterwarnings('ignore')

def structure_to_graph(structure: Structure, label: float) -> Data:
    """Convert pymatgen Structure to PyTorch Geometric Data"""

    # Build connectivity
    crystal_nn = CrystalNN()
    struct_graph = StructureGraph.with_local_env_strategy(structure, crystal_nn)

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

    # Edge indices
    edge_index = []
    edge_attr = []

    for i, j, props in struct_graph.graph.edges(data=True):
        edge_index.append([i, j])
        edge_index.append([j, i])

        distance = structure.get_distance(i, j)
        edge_attr.append([distance])
        edge_attr.append([distance])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float)
    )

    return graph


def main():
    print("="*70)
    print("BUILDING GNN DATASET FROM CRAFTED CIF FILES")
    print("="*70 + "\n")

    # Load labeled data
    project_root = Path(".")
    mof_file = project_root / "data/processed/crafted_mofs_co2_with_costs.csv"
    cif_dir = project_root / "data/CRAFTED-2.0.0/CIF_FILES"

    mof_data = pd.read_csv(mof_file)
    print(f"Loaded {len(mof_data)} MOFs with CO2 labels")

    # FOR TESTING: Use subset first
    import sys
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mof_data = mof_data.head(n_samples)
    print(f"Processing first {len(mof_data)} MOFs for testing\n")

    # Process each MOF
    graphs = []
    skipped = []
    mof_ids_processed = []

    for idx, mof in mof_data.iterrows():
        mof_id = mof['mof_id']
        label = mof['co2_uptake_mean']

        # Find CIF file (could be in subdirectory)
        cif_candidates = list(cif_dir.rglob(f"{mof_id}.cif"))

        if not cif_candidates:
            skipped.append((mof_id, "CIF file not found"))
            continue

        cif_path = cif_candidates[0]

        try:
            # Load and convert
            structure = Structure.from_file(str(cif_path))
            graph = structure_to_graph(structure, label)

            graphs.append(graph)
            mof_ids_processed.append(mof_id)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(mof_data)} MOFs... ({len(graphs)} successful)")

        except Exception as e:
            skipped.append((mof_id, str(e)))

    print(f"\n✓ Processing complete!")
    print(f"  Successfully converted: {len(graphs)} MOFs")
    print(f"  Skipped: {len(skipped)} MOFs\n")

    if skipped:
        print(f"Skipped MOFs (first 10):")
        for mof_id, reason in skipped[:10]:
            print(f"  {mof_id}: {reason}")
        print()

    # Save dataset
    output_file = project_root / "data/processed/gnn_dataset.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        'graphs': graphs,
        'mof_ids': mof_ids_processed,
        'num_node_features': graphs[0].x.shape[1] if graphs else 0,
        'num_edge_features': graphs[0].edge_attr.shape[1] if graphs else 0,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"✓ Dataset saved to: {output_file}")
    print(f"  Num graphs: {len(graphs)}")
    print(f"  Node features: {dataset_info['num_node_features']}")
    print(f"  Edge features: {dataset_info['num_edge_features']}")

    # Statistics
    num_nodes = [g.x.shape[0] for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    labels = [g.y.item() for g in graphs]

    print(f"\nDataset statistics:")
    print(f"  Nodes per graph: {np.mean(num_nodes):.1f} ± {np.std(num_nodes):.1f}")
    print(f"  Edges per graph: {np.mean(num_edges):.1f} ± {np.std(num_edges):.1f}")
    print(f"  Label range: {np.min(labels):.2f} - {np.max(labels):.2f} mol/kg")

    print("\n" + "="*70)
    print("✅ DATASET READY FOR GNN TRAINING")
    print("="*70)


if __name__ == '__main__':
    main()
