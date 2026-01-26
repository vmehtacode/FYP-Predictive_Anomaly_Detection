# Technology Stack Research: Grid Guardian

**Project:** Grid Guardian - Energy anomaly detection using AZR self-play methodology with GNN-based Verifier
**Researched:** 2026-01-26
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

Grid Guardian requires a specialized technology stack that combines graph neural networks for spatial relationship modeling, real-time streaming infrastructure for live energy data, and seamless integration with UK energy market APIs. This research evaluates practical implementation options with a focus on production readiness and Python ecosystem maturity.

**Key Recommendation:** Use PyTorch Geometric (PyG) as the primary GNN framework with PyTorch Geometric Temporal for spatiotemporal modeling, integrate Elexon BMRS API via ElexonDataPortal Python client, and implement real-time streaming using Apache Kafka with confluent-kafka-python client backed by Redis for state management.

---

## 1. Graph Neural Network Libraries

### Recommended: PyTorch Geometric (PyG)

**Why PyG over DGL:**

PyTorch Geometric is the recommended choice for Grid Guardian based on the following factors:

1. **Ecosystem Maturity:** PyG has the larger community (13,700+ GitHub stars vs DGL's 8,800), providing more examples, tutorials, and troubleshooting resources
2. **PyTorch-First Design:** Seamlessly integrates with existing PyTorch ecosystem - Grid Guardian already uses `torch = "^2.1.0"` in pyproject.toml
3. **Energy Domain Applications:** Recent 2025-2026 research shows PyG being actively used for power grid applications (PowerGNN, gnn-powerflow projects)
4. **Temporal Extensions:** PyTorch Geometric Temporal library provides ready-made spatiotemporal GNN architectures
5. **Ease of Experimentation:** Faster prototyping with extensive pre-built GNN architectures

**Trade-offs:**
- DGL has superior memory management and multi-GPU scaling for very large graphs (millions of nodes)
- DGL offers backend flexibility (PyTorch/TensorFlow), but Grid Guardian is PyTorch-only
- For energy grids (typically 10K-100K nodes), PyG's memory footprint is acceptable

### Installation

```bash
# Core PyTorch Geometric (add to pyproject.toml)
pip install torch-geometric

# Optional extensions for performance (recommended for production)
# For PyTorch 2.1.0 with CUDA 12.1 (adjust CUDA version as needed)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# For CPU-only development
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# PyTorch Geometric Temporal for spatiotemporal GNNs
pip install torch-geometric-temporal
```

**Version Compatibility (as of 2026):**
- PyTorch 2.8.* (CUDA: cpu|cu126|cu128|cu129)
- PyTorch 2.7.* (CUDA: cpu|cu118|cu126|cu128)
- PyTorch 2.6.* (CUDA: cpu|cu118|cu124|cu126)

**Note:** Conda packages no longer available for PyTorch >2.5.0. Use pip exclusively.

### PyTorch Geometric: Key Features for Grid Guardian

| Feature | Capability | Why It Matters |
|---------|-----------|----------------|
| **MessagePassing Base** | Create custom GNN layers | Essential for domain-specific verifier architectures |
| **NeighborLoader** | Efficient mini-batch sampling | Handles large grid graphs with 10K+ nodes |
| **Temporal Support** | PyTorch Geometric Temporal library | Models time-evolving energy consumption patterns |
| **Built-in Architectures** | GraphSAGE, GAT, GCN, GIN, etc. | Rapid prototyping of verifier architectures |
| **Heterogeneous Graphs** | Different node/edge types | Model buses, substations, transformers as distinct entities |

### Code Example: Basic GraphSAGE for Energy Grid

```python
import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

# Define energy grid as graph
# Nodes: buses/substations, Edges: transmission lines
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
node_features = torch.randn(3, 10)  # 3 nodes, 10 features (voltage, power, etc.)

data = Data(x=node_features, edge_index=edge_index)

# GraphSAGE model for verifier
model = GraphSAGE(
    in_channels=10,        # Input features per node
    hidden_channels=64,    # Hidden layer size
    num_layers=2,          # 2-hop neighborhood aggregation
    out_channels=32,       # Embedding dimension
    dropout=0.1
)

# Forward pass - get node embeddings
embeddings = model(data.x, data.edge_index)

# For large grids, use neighbor sampling
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # Sample 10 neighbors per layer
    batch_size=128,
    shuffle=True
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    # Use out for anomaly scoring
```

### Alternative: DGL (Not Recommended for Grid Guardian)

**When to Use DGL Instead:**
- If you need TensorFlow backend
- If dealing with truly massive graphs (>1M nodes) where memory is critical
- If you require state-of-the-art multi-GPU scaling

**Installation:**
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
# For CUDA 12.1
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

**Confidence:** HIGH (PyG recommendation backed by Context7-equivalent documentation, recent 2025-2026 research papers, and active community)

---

## 2. Temporal Graph Neural Networks

### Recommended: PyTorch Geometric Temporal

**Why:** First open-source library specifically designed for spatiotemporal signal processing on graphs. Ideal for energy time series where spatial topology (grid structure) and temporal dynamics (consumption patterns) interact.

### Installation

```bash
pip install torch-geometric-temporal
```

**Latest Version:** 0.56.2 (includes index-batching for memory efficiency)

### Key Architectures for Energy Anomaly Detection

| Architecture | Use Case | Implementation |
|-------------|----------|----------------|
| **TGCN (Temporal GCN)** | Short-term forecasting | Combines GCN + GRU for temporal dependencies |
| **A3TGCN** | Attention-based forecasting | Adds attention mechanism to TGCN |
| **DCRNN** | Long-range dependencies | Diffusion convolution with encoder-decoder |
| **GConvGRU/GConvLSTM** | Recurrent spatial-temporal | Graph convolution + RNN cells |
| **MPNN-LSTM** | Custom message passing | Flexible for domain-specific constraints |

### Code Example: Temporal GNN for Energy Time Series

```python
import torch
from torch_geometric_temporal import TemporalData
from torch_geometric_temporal.nn.recurrent import TGCN

# Energy grid with time-evolving node features (e.g., power consumption)
# Snapshots: 48 half-hourly readings (24 hours)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # Static grid topology

# Node features over time: shape [num_nodes, num_features] per snapshot
features = [torch.randn(3, 10) for _ in range(48)]

# Create temporal data
dataset = TemporalData(edge_index=edge_index, features=features)

# TGCN model for anomaly detection
model = TGCN(in_channels=10, out_channels=32)

# Process temporal sequence
hidden_state = None
for time_step, snapshot in enumerate(dataset):
    node_features = snapshot.x
    hidden_state = model(node_features, edge_index, hidden_state)

    # hidden_state contains learned representations at each time step
    # Use for anomaly scoring (e.g., reconstruction error, prediction error)
```

**Integration with AZR Self-Play:**

The Verifier component can use temporal GNNs to assess if proposed scenarios violate temporal constraints:

```python
class TemporalGraphVerifier(nn.Module):
    """Verifier using Temporal GNN to check physical plausibility."""

    def __init__(self, node_features=10, hidden_dim=64):
        super().__init__()
        self.tgcn = TGCN(in_channels=node_features, out_channels=hidden_dim)
        self.predictor = nn.Linear(hidden_dim, node_features)

    def forward(self, temporal_sequence, edge_index):
        """
        Verify scenario by checking prediction error.
        High error = implausible scenario.
        """
        hidden = None
        predictions = []

        for t in range(len(temporal_sequence) - 1):
            hidden = self.tgcn(temporal_sequence[t], edge_index, hidden)
            pred = self.predictor(hidden)
            predictions.append(pred)

        # Compute verification score (lower = more plausible)
        actual = temporal_sequence[1:]
        error = torch.mean((torch.stack(predictions) - torch.stack(actual)) ** 2)

        return error  # Use as reward signal in self-play
```

**Confidence:** HIGH (official library documentation, recent 2025 updates with memory optimizations)

---

## 3. Elexon BMRS API Integration

### Overview

The Balancing Mechanism Reporting Service (BMRS) API provides real-time and historical data on UK electricity generation, demand, and grid status. Critical for Grid Guardian's live anomaly detection capabilities.

### Recommended: ElexonDataPortal Python Client

**Why:** ElexonDataPortal significantly reduces complexity through:
1. Standardized parameter names across 50+ data streams
2. Automatic date-range orchestration (splits large queries into batches)
3. Returns pandas DataFrames (seamlessly integrates with existing codebase using polars/pandas)
4. Active maintenance by OSUKED (Oxford researchers)

### Installation

```bash
pip install ElexonDataPortal
```

**Latest Version:** 2.0.16

### Authentication

```python
from ElexonDataPortal import api

# Option 1: Direct API key
client = api.Client('your_api_key_here')

# Option 2: Environment variable (recommended for production)
# Set BMRS_API_KEY in environment
client = api.Client()  # Auto-loads from environment
```

**Get API Key:** Register at [Elexon Portal](https://www.elexonportal.co.uk/registration/newuser)

### Rate Limits

**Critical:** Elexon Insights API enforces:
- **5,000 requests per minute per user**
- Exceeding limit returns HTTP 429 (Too Many Requests)

**Mitigation Strategy:**
```python
import time
from functools import wraps

def rate_limit(max_per_minute=4500):  # Buffer below 5000
    """Decorator to enforce rate limiting."""
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(max_per_minute=4500)
def fetch_data(client, stream, start, end):
    """Rate-limited data fetching."""
    return getattr(client, f'get_{stream}')(start, end)
```

### Key Data Streams for Grid Guardian

| Stream | Description | Update Frequency | Use Case |
|--------|-------------|------------------|----------|
| **B1610** | Actual generation output per unit | 30 min | Validate generation patterns |
| **FREQ** | Rolling system frequency | 1 min | Detect grid instability events |
| **FUELHH** | Half-hourly generation by fuel type | 30 min | Contextualize anomalies (e.g., wind ramp) |
| **SYSDEM** | System-wide demand | 30 min | Aggregate demand validation |
| **INDOD** | Indicated demand data | 5 min | Near-real-time demand tracking |
| **PHYBMDATA** | Physical BM unit data | Real-time | Unit-level generation monitoring |

### Code Example: Fetching Real-Time Energy Data

```python
from ElexonDataPortal import api
from datetime import datetime, timedelta
import polars as pl

# Initialize client
client = api.Client()  # Reads BMRS_API_KEY from environment

# Fetch half-hourly fuel mix for last 24 hours
end_date = datetime.now()
start_date = end_date - timedelta(days=1)

# ElexonDataPortal handles date range automatically
df_fuel = client.get_FUELHH(
    start_date.strftime('%Y-%m-%d %H:%M'),
    end_date.strftime('%Y-%m-%d %H:%M')
)

# Convert to polars (Grid Guardian uses polars for performance)
pl_fuel = pl.from_pandas(df_fuel)

# Example: Detect anomalous wind generation
wind_gen = pl_fuel.filter(pl.col('fuelType') == 'WIND')
wind_mean = wind_gen['quantity'].mean()
wind_std = wind_gen['quantity'].std()

anomalies = wind_gen.filter(
    (pl.col('quantity') > wind_mean + 3 * wind_std) |
    (pl.col('quantity') < wind_mean - 3 * wind_std)
)

print(f"Detected {len(anomalies)} anomalous wind generation periods")
```

### Alternative: Sheffield Solar's Elexon-BMRS-API

**GitHub:** [SheffieldSolar/Elexon-BMRS-API](https://github.com/SheffieldSolar/Elexon-BMRS-API)

**When to Use:**
- Need lower-level control over API requests
- Working with Python 3.9+ specifically
- Prefer manual query orchestration

**Not recommended for Grid Guardian:** ElexonDataPortal's automation better suits rapid prototyping needs.

### Data Formats

BMRS API returns JSON, which ElexonDataPortal converts to pandas DataFrames:

```python
# Typical FUELHH response structure (as DataFrame)
#   settlementDate  settlementPeriod  fuelType  quantity
# 0  2026-01-26      1                 CCGT      8250.5
# 1  2026-01-26      1                 WIND      4120.3
# 2  2026-01-26      1                 NUCLEAR   5200.0
```

**Grid Guardian Integration:**
- Store in HDF5 format (already using `h5py = "^3.10.0"`) for fast time-series queries
- Use DVC to version snapshot datasets
- Update rolling window cache in Redis for real-time inference

**Confidence:** HIGH (official library with active maintenance, clear rate limit documentation, production-ready Python client)

---

## 4. Real-Time Data Streaming Architecture

### Recommended Stack: Apache Kafka + Confluent Python Client + Redis

**Architecture Pattern:**

```
[Elexon BMRS API]
       ↓ (ElexonDataPortal, 30s polling)
[Kafka Producer] → [Apache Kafka Broker] → [Kafka Consumer]
                                               ↓
                                          [GNN Verifier]
                                               ↓
                                          [Redis Cache] ← [Grafana Dashboard]
```

### Why This Stack?

| Component | Purpose | Why Chosen |
|-----------|---------|------------|
| **Apache Kafka** | Durable message streaming | Industry standard, handles high-throughput energy data, supports replay for debugging |
| **Confluent Python Client** | Python bindings for Kafka | Built on librdkafka (C library), significantly faster than pure Python, supports exactly-once semantics |
| **Redis** | State store & cache | Sub-millisecond latency for GNN inference results, supports time-series data with Redis Streams |

### Installation

```bash
# Kafka Python client (add to pyproject.toml)
pip install confluent-kafka

# Redis client
pip install redis

# Optional: Schema Registry for data validation
pip install confluent-kafka[avro]
```

### Apache Kafka: Confluent Python Client

**Why confluent-kafka-python over kafka-python:**
1. **Performance:** Built on librdkafka (C library), orders of magnitude faster than pure Python
2. **Enterprise Features:** Transactions, exactly-once semantics, Schema Registry integration
3. **Active Maintenance:** Confluent provides commercial support and regular updates

### Code Example: Kafka Producer for Energy Data

```python
from confluent_kafka import Producer
import socket
import json
from datetime import datetime

# Kafka configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Change for production
    'client.id': socket.gethostname(),
    'compression.type': 'lz4',  # Compress time-series data
    'batch.size': 16384,  # Batch messages for throughput
    'linger.ms': 10  # Wait 10ms for batching
}

producer = Producer(conf)

def delivery_report(err, msg):
    """Callback for message delivery confirmation."""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

# Produce energy data to Kafka
def send_energy_reading(reading_data):
    """Send energy reading to Kafka topic."""
    topic = 'energy.readings.raw'

    # Serialize to JSON
    payload = json.dumps({
        'timestamp': datetime.now().isoformat(),
        'bus_id': reading_data['bus_id'],
        'voltage': reading_data['voltage'],
        'power': reading_data['power'],
        'frequency': reading_data['frequency']
    }).encode('utf-8')

    # Asynchronous send with callback
    producer.produce(
        topic,
        key=str(reading_data['bus_id']).encode('utf-8'),
        value=payload,
        callback=delivery_report
    )

    # Trigger delivery report callbacks
    producer.poll(0)

# Flush before shutdown
producer.flush()
```

### Code Example: Kafka Consumer with GNN Processing

```python
from confluent_kafka import Consumer, KafkaError
import json
import torch
from typing import Dict

# Kafka consumer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'gnn-verifier-group',
    'auto.offset.reset': 'earliest',  # Start from beginning if no offset
    'enable.auto.commit': False  # Manual commit for exactly-once semantics
}

consumer = Consumer(conf)
consumer.subscribe(['energy.readings.raw'])

# Assuming GNN verifier is already loaded
# from fyp.selfplay.verifier import GNNVerifier
# verifier = GNNVerifier.load_from_checkpoint('checkpoints/verifier.ckpt')

def process_energy_stream():
    """Consume energy readings and run GNN anomaly detection."""
    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f'Error: {msg.error()}')
                    break

            # Deserialize message
            reading = json.loads(msg.value().decode('utf-8'))

            # Convert to tensor for GNN
            node_features = torch.tensor([
                reading['voltage'],
                reading['power'],
                reading['frequency']
            ]).unsqueeze(0)

            # Run GNN verifier (pseudo-code)
            # anomaly_score = verifier(node_features, edge_index)
            # if anomaly_score > threshold:
            #     publish_alert(reading, anomaly_score)

            # Manually commit offset after processing
            consumer.commit(asynchronous=False)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
```

### Redis Integration: State Store for GNN Results

**Use Cases:**
1. Cache latest GNN embeddings for fast lookup
2. Store rolling window of anomaly scores
3. Publish alerts to subscribers (e.g., Grafana)

```python
import redis
import json

# Redis client
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_anomaly_result(bus_id: str, anomaly_score: float, metadata: Dict):
    """Cache GNN anomaly detection result in Redis."""

    # Store in hash for fast lookup
    result = {
        'bus_id': bus_id,
        'score': anomaly_score,
        'timestamp': metadata['timestamp'],
        'features': json.dumps(metadata['features'])
    }

    # Hash key: anomaly:{bus_id}
    r.hset(f'anomaly:{bus_id}', mapping=result)

    # Set expiry (e.g., 24 hours)
    r.expire(f'anomaly:{bus_id}', 86400)

    # Add to time-series sorted set for historical query
    r.zadd('anomaly:timeline', {
        f'{bus_id}:{metadata["timestamp"]}': anomaly_score
    })

    # Publish to pub/sub channel for real-time alerts
    if anomaly_score > 0.8:  # High confidence anomaly
        alert = {
            'bus_id': bus_id,
            'score': anomaly_score,
            'timestamp': metadata['timestamp']
        }
        r.publish('alerts:anomalies', json.dumps(alert))

def get_recent_anomalies(bus_id: str):
    """Retrieve recent anomaly scores for a bus."""
    return r.hgetall(f'anomaly:{bus_id}')

def get_top_anomalies(limit=10):
    """Get top N anomalies by score from timeline."""
    return r.zrevrange('anomaly:timeline', 0, limit, withscores=True)
```

### Alternative: Bytewax for Python-Native Streaming

**GitHub:** [bytewax/bytewax](https://github.com/bytewax/bytewax)

**Why Consider:**
- Pure Python (easier debugging than Kafka)
- Designed for ML/AI workflows
- Hark (energy company) successfully deployed in days

**When to Use:**
- Rapid prototyping phase
- Smaller scale (< 100K events/sec)
- Team lacks Kafka expertise

**Not recommended for production Grid Guardian:**
- Kafka is industry standard for critical infrastructure
- Better ecosystem support (monitoring, debugging tools)
- Proven scalability for UK grid-scale data

**Confidence:** HIGH (Kafka is proven for energy sector, confluent-kafka-python is production-ready, Redis is standard for low-latency caching)

---

## 5. GNN Architectures for Anomaly Detection

### Recommended Architectures

Based on 2025-2026 research, the following architectures are proven effective for graph-based anomaly detection:

#### 1. Hybrid GNN (GCN + GAT + GAE)

**Paper:** "Anomaly detection in graph databases using graph neural networks" (2025)

**Architecture:**
- **GCN layers:** Extract local structural features (bus → neighbors)
- **GAT layers:** Adaptive attention (weight important connections, e.g., high-voltage lines)
- **Graph Autoencoder:** Unsupervised reconstruction for anomaly scoring

**Implementation Strategy:**
```python
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GAE

class HybridGNNVerifier(nn.Module):
    """Hybrid GNN for energy grid anomaly detection."""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        # GCN branch - structural features
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        # GAT branch - attention mechanism
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)

        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)

        # Decoder for reconstruction
        self.decoder = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        # GCN branch
        gcn_out = torch.relu(self.gcn1(x, edge_index))
        gcn_out = self.gcn2(gcn_out, edge_index)

        # GAT branch
        gat_out = torch.relu(self.gat1(x, edge_index))
        gat_out = self.gat2(gat_out, edge_index)

        # Combine
        fused = torch.cat([gcn_out, gat_out], dim=1)
        embedding = torch.relu(self.fusion(fused))

        # Reconstruction
        reconstruction = self.decoder(embedding)

        return embedding, reconstruction

    def anomaly_score(self, x, edge_index):
        """Compute anomaly score as reconstruction error."""
        _, reconstruction = self.forward(x, edge_index)
        error = torch.mean((x - reconstruction) ** 2, dim=1)
        return error  # Higher error = more anomalous
```

**Why Effective for Energy Grids:**
- GCN captures physical topology (power flow follows graph structure)
- GAT learns to weight critical transmission lines dynamically
- Autoencoder enables unsupervised learning (no anomaly labels needed)

#### 2. MTAD-GAT (Multivariate Time-series Anomaly Detection via GAT)

**Paper:** Zhao et al. (2020), arXiv:2009.02040
**Implementation:** [github.com/ML4ITS/mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch)

**Architecture Components:**
1. 1-D convolution for temporal smoothing
2. Parallel GAT layers (feature-oriented and time-oriented)
3. GRU layers for sequential patterns
4. Dual models: forecasting + reconstruction

**When to Use:**
- Multivariate time series with known correlations (e.g., voltage-power-frequency)
- Baseline to compare against Grid Guardian's AZR approach

**Training Example:**
```bash
python train.py --dataset grid_guardian --lookback 150 --epochs 10
```

#### 3. Graph U-Net with Global Edge Embeddings

**Source:** NVIDIA's GNN-based autoencoder (2025)

**Key Innovation:**
- Hierarchical graph coarsening (like U-Net in image segmentation)
- Learns multi-resolution embeddings (substation-level → region-level → grid-level)
- Global edge embeddings improve anomaly scoring

**Performance:** Outperforms Anomal-E (previous SOTA), achieves real-time throughput with NVIDIA Morpheus

**Relevance to Grid Guardian:**
- Energy grids have natural hierarchy (bus → substation → region)
- Multi-scale detection catches localized faults and systemic issues
- Real-time performance critical for production deployment

#### 4. PowerGNN (Topology-Aware GNN)

**Paper:** "PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids" (March 2025)

**Specialization:**
- Custom graph construction modeling buses and transmission lines
- GraphSAGE convolutions adapted for power flow physics
- Designed for high renewable integration (relevant for UK grid)

**Code:** PyTorch + PyTorch Geometric implementation

**Why Relevant:**
- Purpose-built for power grids (not generic GNN)
- Recent (2025), incorporates latest GNN advances
- Addresses renewable energy integration (key UK grid challenge)

### Comparison Matrix

| Architecture | Unsupervised | Temporal | Hierarchical | Real-Time | Energy-Specific |
|--------------|--------------|----------|--------------|-----------|-----------------|
| Hybrid GNN | ✅ | ❌ | ❌ | ✅ | ❌ |
| MTAD-GAT | ✅ | ✅ | ❌ | ✅ | ❌ |
| Graph U-Net | ✅ | ❌ | ✅ | ✅ | ❌ |
| PowerGNN | ❌ (supervised) | ❌ | ❌ | ✅ | ✅ |
| **Recommended: Temporal Graph U-Net** | ✅ | ✅ | ✅ | ✅ | ⚠️ (adaptable) |

**Grid Guardian Custom Architecture Recommendation:**

Combine strengths of multiple approaches:

1. **Base:** Graph U-Net for hierarchical grid structure
2. **Temporal:** Integrate PyTorch Geometric Temporal (TGCN/A3TGCN)
3. **Attention:** Add GAT layers for critical transmission lines
4. **Physics Constraints:** Inject power flow constraints (from PowerGNN inspiration)

```python
class GridGuardianVerifier(nn.Module):
    """Custom GNN verifier for Grid Guardian."""

    def __init__(self, node_features=10, hidden_dim=64, num_levels=3):
        super().__init__()

        # Temporal processing
        self.temporal_encoder = TGCN(
            in_channels=node_features,
            out_channels=hidden_dim
        )

        # Hierarchical spatial processing (Graph U-Net style)
        self.spatial_encoder = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4)
            for _ in range(num_levels)
        ])

        # Physics-informed constraint checker
        self.constraint_head = nn.Linear(hidden_dim, 3)  # voltage, power, frequency

        # Anomaly scoring head
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, temporal_sequence, edge_index):
        # Process temporal dimension
        hidden = None
        for t in range(len(temporal_sequence)):
            hidden = self.temporal_encoder(temporal_sequence[t], edge_index, hidden)

        # Process spatial dimension with hierarchy
        spatial_features = hidden
        for gat_layer in self.spatial_encoder:
            spatial_features = torch.relu(gat_layer(spatial_features, edge_index))

        # Check physics constraints
        predicted_physics = self.constraint_head(spatial_features)

        # Compute anomaly score
        anomaly_score = torch.sigmoid(self.score_head(spatial_features))

        return anomaly_score, predicted_physics
```

**Confidence:** MEDIUM-HIGH (multiple recent papers validate approaches, but custom architecture needs empirical validation on Grid Guardian dataset)

---

## 6. Additional Infrastructure Components

### Data Versioning: DVC (Already Integrated)

**Status:** Grid Guardian already uses `dvc = {extras = ["s3", "azure"], version = "^3.30.0"}`

**Why DVC for Energy Data:**
- Version large HDF5 files (UK-DALE 6.33 GB, LCL 8.54 GB)
- Track model checkpoints (GNN verifier weights)
- Reproduce experiments with exact data snapshots

### Experiment Tracking: MLflow (Already Integrated)

**Status:** `mlflow = "^2.8.0"` in pyproject.toml

**Recommended Tracking for GNN Experiments:**
```python
import mlflow
import mlflow.pytorch

with mlflow.start_run(run_name="gnn_verifier_v1"):
    # Log hyperparameters
    mlflow.log_params({
        "gnn_architecture": "GraphSAGE",
        "hidden_channels": 64,
        "num_layers": 2,
        "learning_rate": 0.001
    })

    # Train model
    for epoch in range(num_epochs):
        loss = train_step(model, data)
        mlflow.log_metric("train_loss", loss, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "gnn_verifier")
```

### Time Series Database: InfluxDB (Optional)

**When to Consider:**
- If real-time dashboard requirements grow beyond Grafana + Redis
- Need specialized time-series queries (e.g., downsampling, continuous queries)

**Not critical for MVP:** Redis Streams + HDF5 archives sufficient for initial deployment

### Monitoring: Prometheus + Grafana

**For Production:**
- Prometheus: Scrape metrics from Kafka, GNN inference service, API endpoints
- Grafana: Visualize anomaly scores, system health, BMRS data feed status

**Example Metrics:**
- `gnn_inference_latency_ms`: Time to compute anomaly score
- `kafka_consumer_lag`: Backlog in energy data stream
- `bmrs_api_requests_total`: Rate limit monitoring

---

## 7. Dependency Management

### Additions to pyproject.toml

```toml
[tool.poetry.dependencies]
# Existing dependencies remain...

# Graph Neural Networks
torch-geometric = "^2.5.0"
torch-geometric-temporal = "^0.56.2"
pyg-lib = {version = "^0.4.0", optional = true}
torch-scatter = {version = "^2.1.2", optional = true}
torch-sparse = {version = "^0.6.18", optional = true}

# Elexon BMRS API
ElexonDataPortal = "^2.0.16"

# Real-time Streaming
confluent-kafka = "^2.3.0"
redis = "^5.0.0"

# Optional: Schema validation
confluent-kafka-avro = {version = "^2.3.0", optional = true}

[tool.poetry.extras]
# Performance extensions for PyG
pyg-extensions = ["pyg-lib", "torch-scatter", "torch-sparse"]
# Schema Registry support
schema-registry = ["confluent-kafka-avro"]
```

### System Requirements

**Hardware:**
- **GPU:** NVIDIA GPU with CUDA 12.1+ (for GNN training)
- **RAM:** 16GB minimum (32GB recommended for large graph processing)
- **Storage:** SSD for HDF5 time-series access

**Software:**
- **Python:** 3.11+ (already specified in pyproject.toml)
- **CUDA:** 12.1+ (if using GPU acceleration)
- **Kafka:** 3.x (containerized via Docker recommended)
- **Redis:** 7.x (Docker or local)

### Docker Compose for Local Development

```yaml
version: '3.8'

services:
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  grafana-data:
```

**Start services:**
```bash
docker-compose up -d
```

---

## 8. Implementation Roadmap

### Phase 1: Core GNN Infrastructure (Weeks 1-2)

1. Install PyTorch Geometric + extensions
2. Implement basic GraphSAGE verifier
3. Integrate with existing PyTorch codebase
4. Validate on synthetic grid graphs

**Success Criteria:** GNN forward pass completes on 10K node graph

### Phase 2: Temporal Extensions (Weeks 3-4)

1. Install PyTorch Geometric Temporal
2. Implement TGCN-based temporal verifier
3. Adapt for energy time series (half-hourly resolution)
4. Test on SSEN feeder data

**Success Criteria:** Temporal GNN processes 48-hour sequences

### Phase 3: BMRS API Integration (Week 5)

1. Register for Elexon API key
2. Install ElexonDataPortal
3. Implement rate-limited data fetcher
4. Store in HDF5 with DVC versioning

**Success Criteria:** Automated daily fetches of FUELHH, FREQ, SYSDEM

### Phase 4: Streaming Pipeline (Weeks 6-7)

1. Deploy Kafka + Redis via Docker Compose
2. Implement Kafka producer for BMRS data
3. Implement Kafka consumer with GNN inference
4. Cache results in Redis

**Success Criteria:** End-to-end latency < 1 second for single reading

### Phase 5: AZR Self-Play Integration (Weeks 8+)

1. Integrate GNN verifier with Proposer/Solver
2. Implement physics-informed reward signals
3. Train self-play loop with temporal constraints
4. Evaluate on real SSEN distribution network data

**Success Criteria:** Verifier successfully scores proposed anomalous scenarios

---

## 9. Known Limitations and Risks

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **GNN memory explosion** | Cannot process full UK grid (100K+ nodes) | Use NeighborLoader mini-batching, hierarchical sampling |
| **BMRS rate limits** | API throttling during high-frequency polling | Implement exponential backoff, cache aggressively in Redis |
| **Kafka operational complexity** | Steep learning curve for team | Use Docker Compose for local dev, managed Kafka (e.g., Confluent Cloud) for production |
| **Temporal GNN training instability** | RNN components prone to vanishing gradients | Use gradient clipping, LayerNorm, shorter sequence lengths initially |

### Research Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Novel GNN architecture fails** | Verifier cannot distinguish plausible/implausible scenarios | Start with proven baselines (MTAD-GAT), iterate gradually |
| **AZR self-play doesn't converge** | Proposer generates trivial or unsolvable scenarios | Curriculum learning, reward shaping, baseline comparisons |
| **Real-world grid data mismatch** | Models trained on SSEN don't generalize | Validate on multiple distribution networks, data augmentation |

### Data Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **BMRS API downtime** | No live data for real-time demo | Cache historical data, implement fallback to static datasets |
| **SSEN data access issues** | Limited validation data | Use pseudo-feeders from LCL aggregations (as planned) |

---

## 10. Confidence Assessment

| Component | Confidence | Justification |
|-----------|-----------|---------------|
| **PyTorch Geometric** | HIGH | Official docs, active 2025-2026 research in energy domain, 13K+ GitHub stars |
| **PyTorch Geometric Temporal** | HIGH | Official library, recent updates, proven in time-series papers |
| **Elexon BMRS API** | HIGH | Stable API, clear rate limits, production Python clients available |
| **ElexonDataPortal** | MEDIUM-HIGH | Active maintenance, but smaller community than Kafka/Redis |
| **Kafka + confluent-kafka-python** | HIGH | Industry standard, proven in energy sector, excellent Python client |
| **Redis** | HIGH | Standard for low-latency caching, mature Python client |
| **GNN Architectures** | MEDIUM | Multiple recent papers, but custom architecture needs validation |
| **Real-time Streaming Pattern** | MEDIUM-HIGH | Proven pattern, but Grid Guardian-specific integration untested |

**Overall Stack Confidence:** MEDIUM-HIGH

**Reasoning:**
- Core components (PyG, Kafka, Redis) are production-proven
- Energy-specific applications (PowerGNN, BMRS API) validated in 2025-2026 research
- Custom GNN verifier architecture requires empirical validation
- Integration complexity (GNN + streaming + self-play) is novel

---

## 11. Alternative Architectures Considered

### Alternative 1: DGL + TensorFlow

**Why Not:**
- Grid Guardian is PyTorch-based (torch = "^2.1.0")
- Switching to TensorFlow requires rewriting existing models
- PyG has stronger energy domain examples

### Alternative 2: Bytewax + In-Memory State

**Why Not:**
- Kafka is industry standard for energy infrastructure
- Better monitoring/debugging ecosystem
- Proven at scale for UK grid data volumes

### Alternative 3: Pure PyTorch (No GNN Library)

**Why Not:**
- Reinventing message passing is unnecessary
- PyG provides optimized sparse operations
- Faster prototyping with pre-built architectures

### Alternative 4: Cloud-Native (AWS Kinesis + Lambda)

**Why Not:**
- Higher complexity for academic project
- Cost considerations
- Local development easier with Docker Compose
- **Reconsider for production deployment** if Grid Guardian scales

---

## Sources

### PyTorch Geometric vs DGL
- [PyTorch Geometric vs Deep Graph Library | Exxact Blog](https://www.exxactcorp.com/blog/Deep-Learning/pytorch-geometric-vs-deep-graph-library)
- [DGL vs. Pytorch Geometric - Deep Graph Library Discussion](https://discuss.dgl.ai/t/dgl-vs-pytorch-geometric/346)
- [Performance Benchmarks — DGL 2.5 documentation](https://www.dgl.ai/dgl_docs/performance.html)
- [PyTorch Geometric vs Deep Graph Library | Medium](https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802)

### Elexon BMRS API
- [GitHub - OSUKED/ElexonDataPortal](https://github.com/OSUKED/ElexonDataPortal)
- [Elexon Data Portal Documentation](https://osuked.github.io/ElexonDataPortal/)
- [BMRS API Documentation](https://bmrs.elexon.co.uk/api-documentation)
- [ElexonDataPortal PyPI](https://pypi.org/project/ElexonDataPortal/)

### Real-Time Streaming
- [Transforming Energy Monitoring with Modern Stream Processing – bytewax](https://bytewax.io/blog/transforming-energy-monitoring-with-data-streaming)
- [Real Time Data Ingestion Platform (RTDIP) - LF Energy](https://lfenergy.org/projects/real-time-data-ingestion-platform-rtdip/)
- [Data Streaming in Real Life: Utilities & Energy](https://www.confluent.io/resources/online-talk/data-streaming-in-real-life-utilities-and-energy/)
- [The Data Streaming Landscape 2026 - Kai Waehner](https://www.kai-waehner.de/blog/2025/12/05/the-data-streaming-landscape-2026/)

### GNN Anomaly Detection
- [Applying Autoencoder-Based GNNs for Network Anomaly Detection | NVIDIA](https://developer.nvidia.com/blog/applying-autoencoder-based-gnns-for-high-throughput-network-anomaly-detection-in-netflow-data/)
- [Anomaly detection in graph databases | ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1110866525001288)
- [Graph Anomaly Detection with GNNs: Current Status | arXiv](https://arxiv.org/abs/2209.14930)
- [GTAD: Graph and Temporal Neural Network | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9222957/)

### Energy-Specific GNNs
- [PowerGNN: A Topology-Aware GNN for Electricity Grids](https://arxiv.org/html/2503.22721v1)
- [GitHub - mukhlishga/gnn-powerflow](https://github.com/mukhlishga/gnn-powerflow)
- [Accelerating Electric Grid Optimization Using GNNs | Medium](https://medium.com/stanford-cs224w/accelerating-electric-grid-optimization-using-graph-neural-networks-4d095d5c7729)
- [Dynamical Graph Neural Networks for Power Grid Analysis](https://www.mdpi.com/2079-9292/15/3/493)

### PyTorch Geometric Installation
- [Installation — pytorch_geometric documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html)
- [torch-geometric · PyPI](https://pypi.org/project/torch-geometric/)

### Temporal GNNs
- [PyTorch Geometric Temporal Documentation](https://pytorch-geometric-temporal.readthedocs.io/)
- [torch-geometric-temporal · PyPI](https://pypi.org/project/torch-geometric-temporal/)
- [Temporal Graph Learning in 2024 | Towards Data Science](https://towardsdatascience.com/temporal-graph-learning-in-2024-feaa9371b8e2/)

### Kafka Python Clients
- [Python Client for Apache Kafka | Confluent Documentation](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [GitHub - confluentinc/confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python)
- [Choosing a Python Kafka client | Quix](https://quix.io/blog/choosing-python-kafka-client-comparative-analysis)

### GraphSAGE Implementation
- [torch_geometric.nn.models.GraphSAGE documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.models.GraphSAGE.html)
- [pytorch_geometric/examples/graph_sage_unsup_ppi.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py)
- [GraphSAGE: Scaling up GNNs | Towards Data Science](https://towardsdatascience.com/introduction-to-graphsage-in-python-a9e7f9ecf9d7/)

### MTAD-GAT
- [GitHub - ML4ITS/mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch)
- [MTAD-GAT Paper | arXiv](https://arxiv.org/abs/2009.02040)

---

**End of Document**

**Next Steps:**
1. Review with project advisor
2. Install PyTorch Geometric and run initial graph tests
3. Register for Elexon BMRS API key
4. Set up Docker Compose for Kafka/Redis local development
