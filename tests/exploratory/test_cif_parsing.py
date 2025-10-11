"""
Test CIF file parsing and graph construction
"""

from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
import torch
from torch_geometric.data import Data

# Test with one CIF file
cif_file = Path("data/CRAFTED-2.0.0/CIF_FILES/PACMOF/PUPNAQ.cif")

print("="*70)
print("TESTING CIF PARSING AND GRAPH CONSTRUCTION")
print("="*70 + "\n")

# Step 1: Load structure
print(f"Loading CIF file: {cif_file.name}")
structure = Structure.from_file(str(cif_file))

print(f"✓ Structure loaded successfully")
print(f"  Formula: {structure.composition.reduced_formula}")
print(f"  Num atoms: {len(structure)}")
print(f"  Cell volume: {structure.volume:.2f} Å³\n")

# Step 2: Build graph
print("Building molecular graph...")
try:
    crystal_nn = CrystalNN()
    struct_graph = StructureGraph.with_local_env_strategy(structure, crystal_nn)
    print(f"✓ Graph constructed successfully")
    print(f"  Num nodes (atoms): {struct_graph.graph.number_of_nodes()}")
    print(f"  Num edges (bonds): {struct_graph.graph.number_of_edges()}\n")
except Exception as e:
    print(f"❌ Error building graph: {e}")
    exit(1)

# Step 3: Convert to PyTorch Geometric format
print("Converting to PyTorch Geometric format...")

# Node features (one per atom)
node_features = []
for site in structure:
    element = site.specie  # This is already an Element
    features = [
        float(element.Z),                           # Atomic number
        float(element.X) if element.X else 1.5,     # Electronegativity
        float(element.atomic_radius) if element.atomic_radius else 1.0,  # Radius
        float(element.group) if element.group else 1.0,       # Group
        float(element.row) if element.row else 1.0,           # Row
    ]
    node_features.append(features)

x = torch.tensor(node_features, dtype=torch.float)
print(f"  Node features shape: {x.shape}")

# Edge indices
edge_index = []
edge_attr = []

for i, j, props in struct_graph.graph.edges(data=True):
    edge_index.append([i, j])
    edge_index.append([j, i])  # Undirected graph

    # Edge feature: distance
    distance = structure.get_distance(i, j)
    edge_attr.append([distance])
    edge_attr.append([distance])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

print(f"  Edge index shape: {edge_index.shape}")
print(f"  Edge features shape: {edge_attr.shape}")

# Create PyTorch Geometric Data object
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(f"\n✓ PyTorch Geometric graph created")
print(f"  Graph: {graph}")

print("\n" + "="*70)
print("✅ SUCCESS! Ready to process full dataset")
print("="*70)
