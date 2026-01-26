# Research Summary: Grid Guardian Technology Stack

**Project:** Grid Guardian - Energy anomaly detection using AZR self-play methodology with GNN-based Verifier
**Research Focus:** Stack/Technologies
**Researched:** 2026-01-26
**Researcher:** GSD Project Researcher Agent
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

This research evaluated technology options for implementing Grid Guardian's GNN-based anomaly detection system with real-time UK energy data integration. The recommended stack prioritizes PyTorch Geometric for graph neural networks, ElexonDataPortal for BMRS API access, and Apache Kafka with Redis for real-time streaming infrastructure.

**Key Finding:** All core technologies have proven implementations in 2025-2026 energy sector applications, reducing technical risk. The main uncertainty lies in integrating these components with the novel AZR self-play methodology.

---

## Quick Recommendations

| Component | Recommendation | Confidence | Rationale |
|-----------|---------------|-----------|-----------|
| **GNN Library** | PyTorch Geometric | HIGH | Best PyTorch integration, active energy research (PowerGNN 2025), 13K+ stars |
| **Temporal GNNs** | PyTorch Geometric Temporal | HIGH | Official extension, recent memory optimizations, purpose-built for spatiotemporal data |
| **Energy API** | ElexonDataPortal | MEDIUM-HIGH | Battle-tested Python client, handles rate limits (5K req/min), auto-orchestrates queries |
| **Streaming** | Kafka + confluent-kafka-python | HIGH | Industry standard, proven in energy sector, excellent performance via librdkafka |
| **State Store** | Redis | HIGH | Sub-millisecond latency for GNN results, Redis Streams for time-series |
| **GNN Architecture** | Hybrid (Graph U-Net + TGCN + GAT) | MEDIUM | Components proven separately, but custom combination needs validation |

---

## Critical Implementation Details

### 1. PyTorch Geometric Installation

**Commands:**
```bash
# Base library
pip install torch-geometric

# Performance extensions (recommended for 10K+ node graphs)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Temporal extensions
pip install torch-geometric-temporal
```

**Important:** Conda packages no longer available for PyTorch >2.5.0. Use pip exclusively.

### 2. Elexon BMRS API Rate Limits

**Critical Constraint:** 5,000 requests per minute per user

**Mitigation:** Implement rate limiter decorator (see STACK.md section 3) and aggressive Redis caching.

**Registration:** Required at [Elexon Portal](https://www.elexonportal.co.uk/registration/newuser)

### 3. Real-Time Architecture Pattern

```
[BMRS API] → [Kafka Producer] → [Kafka Broker] → [Consumer + GNN] → [Redis Cache] → [Dashboard]
```

**Latency Target:** < 1 second end-to-end for single energy reading

### 4. Key Data Streams for Anomaly Detection

| Stream | Frequency | Purpose |
|--------|-----------|---------|
| FUELHH | 30 min | Generation mix (detect renewable ramps) |
| FREQ | 1 min | Grid frequency (detect instability) |
| SYSDEM | 30 min | System demand (validate aggregates) |
| INDOD | 5 min | Near-real-time demand tracking |

---

## Research Confidence Breakdown

### HIGH Confidence Components (Verified via Official Docs + Recent Research)

1. **PyTorch Geometric:** Official documentation, 2025-2026 energy papers (PowerGNN, gnn-powerflow)
2. **Kafka + confluent-kafka-python:** Industry standard, proven in utilities sector
3. **Redis:** Standard for low-latency caching, mature ecosystem
4. **BMRS API:** Stable production API with clear rate limits

**Sources:** Context7-equivalent official documentation, recent arXiv papers (2025-2026), production case studies

### MEDIUM Confidence Components (Recent Research, Limited Production Examples)

1. **PyTorch Geometric Temporal:** Official library but fewer real-world energy deployments
2. **ElexonDataPortal:** Active maintenance but smaller community than core libraries
3. **Custom GNN Architecture:** Components proven separately, but hybrid needs empirical validation

**Sources:** WebSearch verified with official docs, GitHub repositories with recent commits

### Areas Requiring Validation

1. **AZR Self-Play + GNN Integration:** Novel combination, no prior work found
2. **Real-Time GNN Inference at Scale:** Lab benchmarks exist, but production latency needs testing
3. **Temporal GNN Training Stability:** RNN components prone to vanishing gradients on long sequences

---

## PyTorch Geometric vs DGL Decision

**Winner:** PyTorch Geometric

**Key Factors:**

| Criterion | PyG | DGL | Winner |
|-----------|-----|-----|--------|
| PyTorch Integration | Native | Backend-agnostic | PyG |
| Community Size | 13,700 stars | 8,800 stars | PyG |
| Energy Research | PowerGNN (2025), multiple papers | SE3-Transformer (proteins) | PyG |
| Memory Management | Good | Excellent | DGL |
| Multi-GPU Scaling | Good | Excellent | DGL |
| Ease of Use | High | Medium | PyG |

**Verdict:** PyG wins on ecosystem fit and energy domain activity. DGL's memory advantages matter for 1M+ node graphs, but Grid Guardian targets 10K-100K node range where PyG is sufficient.

**Source Confidence:** HIGH - multiple comparison articles from 2025-2026, official performance benchmarks

---

## GNN Architecture Recommendation

### Recommended: Hybrid Temporal Graph U-Net

**Components:**
1. **TGCN (from PyTorch Geometric Temporal):** Process temporal sequences (half-hourly energy readings)
2. **Graph U-Net:** Hierarchical spatial processing (bus → substation → region)
3. **GAT Layers:** Attention for critical transmission lines
4. **Physics Constraint Head:** Check power flow constraints (inspired by PowerGNN)

**Why Hybrid:**
- Energy grids have temporal + spatial + hierarchical structure
- No single architecture addresses all three
- Recent papers (NVIDIA 2025, PowerGNN 2025) show benefits of specialization

**Training Strategy:**
1. Pre-train on supervised power flow prediction (if labeled data available)
2. Fine-tune with AZR self-play (unsupervised anomaly generation)
3. Validate on SSEN distribution network data

**Risk:** Custom architecture needs empirical validation. Fallback to proven baseline (MTAD-GAT) if convergence issues.

---

## Real-Time Streaming: Kafka vs Alternatives

### Recommended: Apache Kafka with confluent-kafka-python

**Alternatives Considered:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Kafka** | Industry standard, proven in energy, excellent tooling | Operational complexity | **Recommended** |
| **Bytewax** | Pure Python, faster prototyping, Hark (energy) success | Less mature, fewer monitoring tools | Prototyping only |
| **AWS Kinesis** | Managed service, zero ops | Vendor lock-in, higher cost | Reconsider for production |
| **Redis Streams** | Simple, built-in to Redis | Not designed for high-throughput durable streaming | State store, not primary stream |

**Decision:** Kafka for durable event streaming + Redis for low-latency state access

**Local Development:** Docker Compose with Kafka + Zookeeper + Redis (see STACK.md section 7)

---

## Implementation Roadmap

### Phase 1: Core GNN (Weeks 1-2)
- Install PyTorch Geometric + extensions
- Implement GraphSAGE baseline verifier
- Test on synthetic 10K node grid

**Success Metric:** Forward pass completes in <100ms

### Phase 2: Temporal Extensions (Weeks 3-4)
- Install PyTorch Geometric Temporal
- Implement TGCN verifier
- Process 48-hour energy sequences

**Success Metric:** Temporal model converges on SSEN data

### Phase 3: BMRS Integration (Week 5)
- Register for API key
- Install ElexonDataPortal
- Implement rate-limited fetcher

**Success Metric:** Daily automated data pulls with no 429 errors

### Phase 4: Streaming Pipeline (Weeks 6-7)
- Deploy Kafka/Redis via Docker
- Build producer/consumer
- Integrate GNN inference

**Success Metric:** <1s end-to-end latency

### Phase 5: AZR Self-Play (Weeks 8+)
- Connect Proposer → Solver → Verifier
- Train with physics-informed rewards
- Validate on real grid data

**Success Metric:** Verifier distinguishes plausible/implausible scenarios with >80% accuracy

---

## Known Risks and Mitigations

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GNN memory explosion on large graphs | Cannot process full UK grid | Medium | NeighborLoader mini-batching, hierarchical sampling |
| BMRS API rate limit throttling | Delayed data ingestion | High | Rate limiter + Redis cache + exponential backoff |
| Kafka operational complexity | Steep learning curve | Medium | Docker Compose for local dev, thorough documentation |
| Temporal GNN training instability | Model divergence | Medium | Gradient clipping, shorter sequences, curriculum learning |

### Research Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Custom GNN architecture fails to converge | Verifier cannot score scenarios | Medium | Start with MTAD-GAT baseline, iterate gradually |
| AZR self-play doesn't produce diverse scenarios | Weak anomaly detection | Medium | Reward shaping, curriculum learning, manual scenario injection |
| Real-world grid validation shows poor generalization | Model overfits to training networks | Low | Multi-network validation, data augmentation |

### Data Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| BMRS API downtime during demo | No live data | Low | Cache 1-week historical data, fallback datasets |
| SSEN time-series data access issues | Limited validation | Medium | Use pseudo-feeders from LCL aggregations (as planned) |

---

## Gap Analysis

### Fully Researched (Ready to Implement)

- PyTorch Geometric installation and basic usage
- Elexon BMRS API authentication and rate limits
- Kafka producer/consumer patterns
- Redis caching strategies
- Existing GNN architectures (GraphSAGE, GAT, TGCN)

### Partially Researched (Needs Phase-Specific Exploration)

- Custom hybrid GNN architecture hyperparameter tuning
- AZR self-play reward shaping for graph anomalies
- Production deployment (containerization, scaling)
- Monitoring and alerting (Prometheus/Grafana integration)

### Not Yet Researched (Defer to Later Phases)

- Model interpretability (which graph features drive anomaly scores?)
- Adversarial robustness (can attackers fool the GNN verifier?)
- Multi-modal fusion (integrate weather data, grid topology changes)
- Transfer learning (pre-train on other power grids?)

---

## Dependencies to Add

**Add to `pyproject.toml`:**

```toml
[tool.poetry.dependencies]
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

[tool.poetry.extras]
pyg-extensions = ["pyg-lib", "torch-scatter", "torch-sparse"]
```

---

## Next Actions for Project Team

1. **Immediate (This Week):**
   - Install PyTorch Geometric: `pip install torch-geometric`
   - Register for Elexon BMRS API key
   - Review STACK.md section 5 (GNN architectures)

2. **Short-Term (Next 2 Weeks):**
   - Implement GraphSAGE baseline on synthetic grid
   - Fetch sample BMRS data (FUELHH stream)
   - Set up local Kafka/Redis via Docker Compose

3. **Medium-Term (Weeks 3-6):**
   - Integrate temporal GNN with SSEN data
   - Build end-to-end streaming pipeline
   - Prototype AZR self-play loop

4. **Before Full Implementation:**
   - Validate GNN memory usage on target graph sizes
   - Benchmark inference latency (target <100ms per batch)
   - Test BMRS API rate limits under load

---

## Questions for Advisor Review

1. **Graph Size:** What's the target node count for Grid Guardian? (10K buses? 100K meters?)
   - Impacts PyG vs DGL decision and memory requirements

2. **Real-Time Requirements:** Is <1s latency acceptable, or do we need <100ms?
   - Impacts batch sizes and model complexity

3. **Labeled Data:** Do we have any labeled anomalies for supervised pre-training?
   - Affects training strategy (fully unsupervised vs transfer learning)

4. **Deployment Target:** Local machine, university cluster, or cloud?
   - Impacts Docker vs Kubernetes decisions

5. **Budget:** Any constraints on API usage (BMRS) or cloud resources?
   - 5K BMRS requests/min is generous, but sustained usage adds up

---

## Files Produced

| File | Purpose | Location |
|------|---------|----------|
| **STACK.md** | Comprehensive technology stack research | `.planning/research/STACK.md` |
| **RESEARCH_SUMMARY.md** | Executive summary (this document) | `.planning/research/RESEARCH_SUMMARY.md` |

**Status:** Research complete. Ready for roadmap creation.

**DO NOT COMMIT:** Orchestrator will handle commits after all research phases complete.

---

## Source Quality Assessment

### HIGH Quality Sources (Official Docs, Peer-Reviewed Papers)
- PyTorch Geometric official documentation
- Confluent Kafka documentation
- Elexon BMRS API official docs
- PowerGNN paper (arXiv 2025)
- NVIDIA GNN anomaly detection paper (2025)

### MEDIUM Quality Sources (GitHub, Technical Blogs)
- OSUKED/ElexonDataPortal GitHub (active maintenance)
- PyTorch Geometric Temporal GitHub
- Medium articles on GNN energy applications

### LOW Quality Sources (Unverified WebSearch Only)
- None - all key findings verified with official sources

**Research Methodology:** WebSearch for discovery → Official docs for verification → Code examples for validation

---

**Research Complete: 2026-01-26**
