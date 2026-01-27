# Phase 1: GNN Verifier Foundation - Research

**Researched:** 2026-01-27
**Domain:** Graph Neural Networks for Anomaly Detection (PyTorch Geometric)
**Confidence:** HIGH

## Summary

This phase builds a topology-aware GNN verifier that transforms SSEN grid metadata into per-node anomaly scores. The research confirms PyTorch Geometric (PyG) 2.7.0 as the standard library for graph neural networks in Python, with full compatibility for the project's existing PyTorch 2.1+ and Python 3.11 stack.

The locked decisions from CONTEXT.md specify GAT (Graph Attention Network) with 3 layers and 64-dimensional hidden states. Research confirms GATv2Conv is the preferred implementation over standard GATConv, as it solves the "static attention problem" and allows dynamic attention rankings conditional on query nodes. For the temporal encoding (Claude's discretion), research supports 1D convolution as the optimal choice for this use case: faster than LSTM, better captures local patterns, and aligns with the fixed time-window snapshot approach already decided.

The primary challenge is oversmoothing prevention in a 3-layer GNN. Research confirms that residual/skip connections combined with LayerNorm are mathematically proven to prevent oversmoothing. The recommended architecture uses initial residual connections (GCNII-style) where each layer receives a weighted sum of the previous layer output and the initial node features.

**Primary recommendation:** Use GATv2Conv (not GATConv), 4 attention heads per layer, 1D-Conv temporal encoder, and initial-residual skip connections with LayerNorm after each GAT layer.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch-geometric | ^2.7.0 | Graph neural network operations | De facto standard for GNNs in PyTorch; supports torch.compile, full PyTorch 2.x integration |
| torch | ^2.1.0 | Already in project | Base deep learning framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch-scatter | Latest via wheel | Optimized sparse reductions | Performance optimization (optional but recommended) |
| torch-sparse | Latest via wheel | SparseTensor support | Large graphs, memory optimization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| GATv2Conv | GATConv | GATv2 strictly more expressive; GATConv has static attention limitation |
| PyG | DGL | DGL is viable but PyG has better PyTorch 2.x integration and torch.compile support |
| torch-geometric-temporal | Manual temporal | Provides StaticGraphTemporalSignal but adds dependency; manual approach sufficient for fixed windows |

**Installation:**
```bash
# Core (minimal, sufficient for this phase)
poetry add torch-geometric

# Optional performance optimizations (CUDA 11.8 example)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Or for CPU-only
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

## Architecture Patterns

### Recommended Project Structure
```
src/fyp/
├── gnn/                     # NEW: GNN verifier module
│   ├── __init__.py
│   ├── graph_builder.py     # SSEN metadata -> PyG Data
│   ├── temporal_encoder.py  # 1D-Conv time series encoder
│   ├── gat_verifier.py      # GAT model architecture
│   └── layers.py            # Custom layers (skip connections, norm)
├── selfplay/
│   └── verifier.py          # Existing (will integrate GNN in Phase 2)
└── ingestion/
    └── ssen_ingestor.py     # Existing (provides graph metadata)
```

### Pattern 1: Graph Construction from SSEN Hierarchy
**What:** Transform SSEN feeder lookup into PyG Data objects
**When to use:** Every forward pass requires graph structure
**Example:**
```python
# Source: PyTorch Geometric Data class documentation
import torch
from torch_geometric.data import Data

def build_grid_graph(
    substations: list[str],
    feeders: list[str],
    households: list[str],
    feeder_to_substation: dict[str, str],
    household_to_feeder: dict[str, str],
    node_features: torch.Tensor,  # [num_nodes, num_features]
) -> Data:
    """Build hierarchical grid graph.

    Three-level hierarchy: Substations -> Feeders -> Households
    Edges: bidirectional connectivity (undirected graph)
    """
    # Create node ID mappings
    all_nodes = substations + feeders + households
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Build edges (COO format: [2, num_edges])
    edges = []
    for feeder, substation in feeder_to_substation.items():
        i, j = node_to_idx[feeder], node_to_idx[substation]
        edges.extend([[i, j], [j, i]])  # Bidirectional

    for household, feeder in household_to_feeder.items():
        i, j = node_to_idx[household], node_to_idx[feeder]
        edges.extend([[i, j], [j, i]])  # Bidirectional

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Node type encoding (0=substation, 1=feeder, 2=household)
    node_types = torch.tensor(
        [0] * len(substations) + [1] * len(feeders) + [2] * len(households),
        dtype=torch.long
    )

    return Data(
        x=node_features,
        edge_index=edge_index,
        node_type=node_types,
        num_nodes=len(all_nodes),  # Explicit for isolated node safety
    )
```

### Pattern 2: GATv2 with Skip Connections and LayerNorm
**What:** 3-layer GAT with oversmoothing prevention
**When to use:** Core verifier architecture
**Example:**
```python
# Source: PyG GATv2Conv docs + oversmoothing research
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GATVerifier(nn.Module):
    """GAT-based anomaly verifier with oversmoothing prevention."""

    def __init__(
        self,
        in_channels: int,      # Input feature dim
        hidden_channels: int = 64,
        out_channels: int = 1,  # Anomaly score per node
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Initial projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # GATv2 with residual support
            conv = GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,  # Concatenate head outputs
                dropout=dropout,
                add_self_loops=True,
                residual=True,  # Built-in residual connection
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid(),  # Anomaly score in [0, 1]
        )

        # Initial residual weight (learnable, GCNII-style)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with initial residual connections.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Anomaly scores [num_nodes, 1]
        """
        # Project to hidden dim
        h = self.input_proj(x)
        h0 = h  # Store initial features for residual

        for conv, norm in zip(self.convs, self.norms):
            # GAT layer (has internal residual)
            h_new = conv(h, edge_index)

            # Initial residual (GCNII-style oversmoothing prevention)
            h_new = self.alpha * h_new + (1 - self.alpha) * h0

            # Layer normalization
            h = norm(h_new)

        # Per-node anomaly scores
        return self.output_head(h)
```

### Pattern 3: 1D-Conv Temporal Encoder
**What:** Encode time-window snapshots before GNN
**When to use:** Pre-process temporal features per node
**Example:**
```python
# Source: Temporal GNN survey (arxiv:2307.03759)
import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """1D-Conv encoder for fixed time-window features.

    Processes [current_load, avg_24h, peak_7d, ...] style features
    into a dense embedding per node.
    """

    def __init__(
        self,
        input_features: int,   # e.g., 3 for [current, avg_24h, peak_7d]
        embed_dim: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 1D convolution treats features as sequence
        self.conv1 = nn.Conv1d(1, 32, kernel_size=min(kernel_size, input_features), padding='same')
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=min(kernel_size, input_features), padding='same')

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)

        # Fallback for very small feature sets
        self.linear_fallback = nn.Linear(input_features, embed_dim)
        self.use_conv = input_features >= 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features.

        Args:
            x: [num_nodes, num_temporal_features]

        Returns:
            [num_nodes, embed_dim]
        """
        if not self.use_conv:
            return self.norm(self.linear_fallback(x))

        # [N, F] -> [N, 1, F] for Conv1d
        h = x.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)  # [N, embed_dim]

        return self.norm(h)
```

### Anti-Patterns to Avoid
- **Deep GNN without residuals:** 3+ layers will cause oversmoothing; always use skip connections
- **GATConv instead of GATv2Conv:** Static attention limits expressiveness; GATv2 is strictly better
- **Averaging attention heads in early layers:** Use concatenation (concat=True) for first/middle layers; only average in final layer if needed
- **Manual edge list construction:** Always use COO format [2, num_edges] with `.t().contiguous()` if transposing
- **Ignoring isolated nodes:** Set `num_nodes` explicitly; PyG infers from max edge index which fails for isolated nodes

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph batching | Manual padding/masking | `torch_geometric.loader.DataLoader` | Handles sparse block diagonal adjacency automatically |
| Attention mechanism | Custom attention weights | `GATv2Conv` | Proven implementation, handles edge cases, supports edge features |
| Graph construction | Custom adjacency matrix | `torch_geometric.data.Data` class | COO format, GPU transfer, batching support built-in |
| Sparse operations | Dense matrix on sparse graphs | `torch-scatter` / native PyG ops | Memory efficiency for large graphs |
| Subgraph sampling | Manual node selection | `NeighborLoader` (if needed) | Mini-batch training on large graphs |

**Key insight:** PyTorch Geometric provides highly optimized, CUDA-accelerated implementations. Custom graph operations are slower, buggier, and harder to maintain. Use PyG primitives wherever possible.

## Common Pitfalls

### Pitfall 1: Oversmoothing at 3 Layers
**What goes wrong:** Node embeddings converge to similar vectors; hierarchy levels become indistinguishable
**Why it happens:** Message passing averages neighbor features; repeated application smooths signal
**How to avoid:**
- Use initial residual connections (GCNII-style): `h = alpha * h_conv + (1-alpha) * h0`
- Apply LayerNorm after each GAT layer
- GATv2Conv has `residual=True` parameter for built-in skip connections
**Warning signs:** Validation shows similar anomaly scores across hierarchy levels; t-SNE of embeddings shows clustering collapse

### Pitfall 2: edge_index Format Errors
**What goes wrong:** Graph has wrong connectivity; model trains but produces garbage
**Why it happens:** PyG expects COO format [2, num_edges], not list of tuples or adjacency matrix
**How to avoid:**
```python
# CORRECT
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

# WRONG (list of tuples)
edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
edge_index = torch.tensor(edges).t()  # Missing .contiguous()

# CORRECT from tuples
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
```
**Warning signs:** Unexpected number of edges; graph connectivity doesn't match expected topology

### Pitfall 3: Isolated Nodes Cause Silent Failures
**What goes wrong:** Graph has fewer nodes than expected; some node features ignored
**Why it happens:** PyG infers `num_nodes` from `edge_index.max() + 1`; isolated nodes have no edges
**How to avoid:** Always set `num_nodes` explicitly in Data constructor
```python
data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
```
**Warning signs:** `data.num_nodes` doesn't match `x.size(0)`

### Pitfall 4: Attention Head Dimension Mismatch
**What goes wrong:** Shape errors in forward pass or output layer
**Why it happens:** With `concat=True`, output dim = `out_channels * heads`
**How to avoid:**
```python
# If hidden_dim=64 and heads=4, use out_channels=16
conv = GATv2Conv(64, 16, heads=4, concat=True)  # Output: 64 (16*4)

# For final layer, can use heads=1 or concat=False
final = GATv2Conv(64, 64, heads=4, concat=False)  # Output: 64 (averaged)
```
**Warning signs:** RuntimeError about tensor shapes in Linear layers after GATConv

### Pitfall 5: Missing Node Type Information
**What goes wrong:** Model can't distinguish substations/feeders/households
**Why it happens:** Without explicit type encoding, GNN sees all nodes identically
**How to avoid:** Add node type as feature (learnable embedding or one-hot):
```python
# Learnable type embedding
self.type_embed = nn.Embedding(3, 16)  # 3 types, 16-dim embed
node_type_feat = self.type_embed(node_types)
x = torch.cat([temporal_features, node_type_feat], dim=-1)
```
**Warning signs:** Model produces same scores for different hierarchy levels

## Code Examples

Verified patterns from official sources:

### Creating a Hierarchical Grid Graph
```python
# Source: PyG Data class documentation
from torch_geometric.data import Data
import torch

def create_test_grid_graph():
    """Create minimal test graph for verification."""
    # 1 substation, 2 feeders, 4 households
    # Substation 0 -> Feeders 1,2 -> Households 3,4,5,6
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3, 1, 4, 2, 5, 2, 6],  # Source
        [1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2],  # Target (bidirectional)
    ], dtype=torch.long)

    # Node features: [num_nodes=7, features=4]
    x = torch.randn(7, 4)

    # Node types: 0=substation, 1=feeder, 2=household
    node_type = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        num_nodes=7,
    )
```

### Batching Multiple Graphs
```python
# Source: PyG DataLoader documentation
from torch_geometric.loader import DataLoader

# Create list of Data objects (one per time window or subgraph)
graphs = [create_test_grid_graph() for _ in range(32)]

# DataLoader handles batching automatically
loader = DataLoader(graphs, batch_size=32, shuffle=True)

for batch in loader:
    # batch.x: [total_nodes_in_batch, features]
    # batch.edge_index: [2, total_edges_in_batch]
    # batch.batch: [total_nodes_in_batch] - maps node to graph
    scores = model(batch.x, batch.edge_index)

    # To get per-graph outputs, use batch.batch for indexing
    # e.g., scatter_mean(scores, batch.batch, dim=0)
```

### Inference Latency Optimization
```python
# Source: PyTorch inference optimization docs
import torch

model = GATVerifier(in_channels=64, hidden_channels=64)
model.eval()

# Option 1: torch.inference_mode (recommended)
@torch.inference_mode()
def predict_batch(model, batch):
    return model(batch.x, batch.edge_index)

# Option 2: torch.compile for repeated inference (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# Measure latency
import time
batch = create_test_grid_graph()  # batch_size=1
batch.x = batch.x.repeat(32 * 7, 1)  # Simulate batch of 32 graphs with 7 nodes each

start = time.perf_counter()
for _ in range(100):
    with torch.inference_mode():
        _ = model(batch.x, batch.edge_index)
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = (time.perf_counter() - start) / 100 * 1000
print(f"Latency: {elapsed:.2f}ms")  # Target: <30ms for batch_size=32
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GATConv | GATv2Conv | 2021 (ICLR 2022) | Dynamic attention, strictly more expressive |
| Deep GNN without normalization | Residual + LayerNorm | 2024-2025 | Proven to prevent oversmoothing mathematically |
| Conda installation for PyG | pip with wheels | PyG 2.5+ (2024) | Conda no longer supported; pip only |
| Manual sparse operations | torch-scatter/native PyG | Ongoing | Native operations increasingly optimized |

**Deprecated/outdated:**
- GATConv: Use GATv2Conv instead (static attention limitation)
- PyTorch Geometric < 2.3: Python 3.11 and torch.compile support added in 2.3
- Conda installation: No longer available for PyTorch > 2.5

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal attention heads for grid topology**
   - What we know: Literature uses 2-8 heads; 4-8 common for first layers
   - What's unclear: No specific studies on power grid hierarchies
   - Recommendation: Start with 4 heads, tune during training

2. **Node feature engineering from SSEN metadata**
   - What we know: Have total_mpan_count, hierarchy IDs, postcodes
   - What's unclear: Which features most predictive for anomaly detection
   - Recommendation: Start with [temporal_encoding, node_type_embed, log(mpan_count)]

3. **Inference latency target feasibility**
   - What we know: Target is <30ms for batch_size=32
   - What's unclear: Exact node count per batch in production
   - Recommendation: Profile early with realistic graph sizes

## Sources

### Primary (HIGH confidence)
- [PyTorch Geometric Data class](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) - Graph construction, attributes
- [GATv2Conv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html) - Layer parameters, forward signature
- [torch-geometric PyPI](https://pypi.org/project/torch-geometric/) - Version 2.7.0, Python requirements
- [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html) - Dependencies, wheel URLs

### Secondary (MEDIUM confidence)
- [Residual Connections and Normalization Prevent Oversmoothing](https://arxiv.org/abs/2406.02997) - Mathematical proof, GCNII-style residuals
- [GNN Temporal Survey](https://arxiv.org/abs/2307.03759) - 1D-Conv vs LSTM vs MLP comparison
- [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/) - StaticGraphTemporalSignal patterns

### Tertiary (LOW confidence)
- Various blog posts on attention head tuning - No authoritative source; treat as starting points
- 2025 position paper on oversmoothing narrative - Suggests deep GNNs possible with proper normalization; verify with experiments

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyG is undisputed standard; versions verified via PyPI
- Architecture: HIGH - GATv2Conv, residuals, LayerNorm all well-documented
- Pitfalls: HIGH - Common issues verified in multiple sources and official docs
- Node feature engineering: MEDIUM - Domain-specific; requires experimentation

**Research date:** 2026-01-27
**Valid until:** 2026-03-27 (60 days - PyG ecosystem stable)
