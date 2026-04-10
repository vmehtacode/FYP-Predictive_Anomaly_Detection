# Grid Guardian

Topology-aware anomaly detection for UK electricity distribution networks using Graph Attention Networks and self-play verification.

## Overview

This is a final-year BSc Computer Science project at Aston University. The goal is to detect anomalies in electricity distribution networks by combining graph neural networks with physics-based constraint verification.

The core idea is a three-layer hybrid verifier. The first layer checks forecasts against known physics constraints (voltage limits, capacity bounds, ramp rates) drawn from UK electrical standards (BS EN 50160, BS 7671). The second layer is a Graph Attention Network (GATv2Conv) trained on synthetic anomaly data that learns to spot patterns across connected nodes in the grid topology. The third layer checks whether anomalies propagate through neighbors in a way that suggests a real cascade event rather than an isolated reading error.

The project also includes a self-play training framework where a proposer agent generates challenging energy consumption scenarios, a solver agent forecasts under those conditions, and a verifier agent checks the results. This was inspired by the propose-solve-verify paradigm from the Absolute Zero Reasoner paper. In practice, the self-play loop did not outperform simple baselines on periodic time-series data -- this negative result is documented and led to the pivot towards the GNN-based verification approach, which is the main contribution.

All evaluation is on synthetic anomaly data generated to match SSEN grid characteristics. There are no ground-truth anomaly labels in the real datasets, so the results should be read as proof-of-concept rather than production validation.

## Architecture

The pipeline has four stages:

**Data ingestion** -- Three UK energy datasets are processed into a common schema (30-min resolution Parquet files). Each dataset has its own ingestor class (`LCLIngestor`, `UKDALEIngestor`, `SSENIngestor`) inheriting from `BaseIngestor`. The SSEN metadata is used to build the grid graph.

**Graph construction** -- `GridGraphBuilder` transforms SSEN feeder/substation metadata into PyTorch Geometric `Data` objects. The graph has three node types: primary substations (type 0), secondary substations (type 1), and LV feeders (type 2). Edges are bidirectional and represent physical connectivity.

**GNN verification** -- `GATVerifier` is a 3-layer GATv2Conv model with 4 attention heads and 64 hidden channels. It uses a 1D-Conv temporal encoder (`TemporalEncoder`) per node, learnable node-type embeddings, and GCNII-style initial residual connections to prevent oversmoothing. Trained on `SyntheticAnomalyDataset` which generates labeled graph data with five anomaly types: spike, dropout, cascade, ramp violation, and normal.

**Hybrid ensemble** -- `HybridVerifierAgent` combines the three verification layers with configurable weights. Physics-only nodes can early-exit before GNN inference. The `CascadeLogicLayer` scores nodes based on how many of their neighbors are also flagged, distinguishing propagation events from isolated spikes. All thresholds are loaded from YAML config, not hardcoded.

## Key Results

GATVerifier performance on held-out synthetic test data (500 graphs, 44 nodes each, seed=9999):

| Metric | Value |
|--------|-------|
| Accuracy | 98.3% |
| Precision | 93.8% |
| Recall | 95.5% |
| F1 | 0.946 |
| Test nodes | 22,000 |

Ablation study on hybrid verifier components (200 samples, seed=42):

| Configuration | ROC-AUC | Optimal F1 |
|---------------|---------|------------|
| Baseline (VerifierAgent) | 0.50 | 0.00 |
| Physics only | 0.50 | 0.00 |
| GNN only | 1.00 | 1.00 |
| Cascade only | 0.90 | 0.89 |
| Full hybrid | 1.00 | 1.00 |

The GNN layer is the primary driver of detection performance. The cascade layer adds value for propagation-type anomalies. Physics constraints provide a useful first filter but don't discriminate well on their own in this synthetic setup.

These numbers are on synthetic data designed to match grid characteristics. Real-world performance would likely be lower, and the system has not been validated against labeled real anomalies.

The test suite has 349 test functions covering ingestion, models, self-play, GNN components, and the hybrid verifier.

## Project Structure

```
src/fyp/
  ingestion/       Data loading for LCL, UK-DALE, SSEN datasets
  gnn/             GATVerifier model, graph builder, synthetic dataset, trainer
  selfplay/        Proposer, Solver, Verifier agents and training loop
  models/          PatchTST, ensemble forecaster, autoencoder
  evaluation/      Ablation study and benchmark framework
  baselines/       Simple forecasting and anomaly detection baselines
  config.py        Pydantic config models for experiments
  metrics.py       Forecasting and anomaly detection metrics
  runner.py        CLI entry point for running baselines

tests/             349 tests (unit, integration, smoke)
notebooks/         Jupyter notebooks for data exploration (LCL, UK-DALE, SSEN)
scripts/           Training scripts, evaluation runners, experiment drivers
examples/          Demo scripts for self-play and BDH enhancements
configs/           YAML configuration files
data/              DVC-tracked datasets (raw, processed, derived)
app.py             Streamlit dashboard for interactive exploration
docs/              Design docs, dataset notes, experiment specs
```

## Setup and Running

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
# Install dependencies
poetry install

# Activate the environment
poetry shell

# Run the test suite
pytest tests/

# Run the Streamlit dashboard
streamlit run app.py

# Run forecasting baselines on sample data
python -m fyp.runner forecast --dataset lcl --use-samples

# Run anomaly detection baselines
python -m fyp.runner anomaly --dataset ukdale --use-samples

# Ingest data (sample mode, no downloads needed)
python -m fyp.ingestion.cli lcl --use-samples
python -m fyp.ingestion.cli ukdale --use-samples
python -m fyp.ingestion.cli ssen --use-samples
```

## Datasets

| Dataset | Size | Records | Entities | Role |
|---------|------|---------|----------|------|
| London Smart Meters (LCL) | 8.5 GB | ~167M readings | 5,567 households | Training and validation |
| UK-DALE | 6.3 GB | ~114M readings | 5 houses | Appliance-level analysis |
| SSEN LV Feeder | 37 MB | 100K metadata records | 100K feeders (28 with time-series) | Grid topology and validation |

All large files are tracked with DVC. Small synthetic samples are included in `data/samples/` for testing without downloading the full datasets. See `docs/download_links.md` for access instructions.

The SSEN dataset provides real distribution network topology (primary substations, secondary substations, LV feeders) used to build the graph structure. The LCL and UK-DALE datasets provide household consumption patterns used for the self-play training loop and baseline evaluation.

## References

1. Zhao et al. (2025). "Absolute Zero: Reinforced Self-Play Reasoning with Zero Data." arXiv:2505.03335. -- The propose-solve-verify self-play paradigm adapted for energy forecasting.
2. Brody, Alon, and Yahav (2022). "How Attentive are Graph Attention Networks?" ICLR 2022. -- GATv2Conv, which fixes the static attention problem in the original GAT.
3. Chen et al. (2020). "Simple and Deep Graph Convolutional Networks." ICML 2020. -- GCNII initial residual connections used to prevent oversmoothing.
4. Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. -- PatchTST architecture used in the forecasting baselines.
5. Kosowski et al. (2025). "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain." arXiv:2509.26507. -- BDH-inspired Hebbian constraint adaptation and graph-based scenario relationships.

## License

MIT License. See [LICENSE](LICENSE) for details.
