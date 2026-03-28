"""Grid Guardian -- Streamlit dashboard for FYP showcase and viva.

Run with: PYTHONPATH=src streamlit run app.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Grid Guardian",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette and style constants
# ---------------------------------------------------------------------------
NAVY = "#1B2838"
TEAL = "#0F6E56"
RED = "#993C1D"
STEEL = "#378ADD"
GREY_BG = "#F5F6F8"
GREY_TEXT = "#5A6270"
WHITE = "#FFFFFF"

PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = [STEEL, TEAL, RED, "#B0BEC5", "#7B8794", "#C4A35A", "#6C4F9C", "#2CA58D"]

# Common layout kwargs for transparent plotly charts on dark Streamlit
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Minimal CSS -- let Streamlit dark theme handle colors
st.markdown(
    """
    <style>
    h1, h2, h3 { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### Grid Guardian")
    st.caption("Predictive Anomaly Detection for UK Power Grids")
    st.divider()
    st.markdown(f"**Author:** Vatsal Mehta")
    st.markdown(f"**Supervisor:** Dr. Farzaneh Farhadi")
    st.markdown(f"**Institution:** Aston University")
    st.markdown(f"**Programme:** BSc Computer Science")
    st.divider()
    st.caption("Final Year Project 2025-26")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
SSEN_CONSTRAINTS = "data/derived/ssen_constraints.json"
GNN_CHECKPOINT = "data/derived/models/gnn/gnn_verifier_v1.pth"
HYBRID_CONFIG = "configs/hybrid_verifier.yaml"
SSEN_METADATA = "data/processed/ssen_metadata.parquet"
BENCHMARK_JSON = "data/derived/evaluation/benchmark_results.json"
ABLATION_JSON = "data/derived/evaluation/ablation_results.json"


@st.cache_resource
def load_graph_data():
    """Build SSEN grid graph from metadata."""
    from fyp.gnn.graph_builder import GridGraphBuilder

    builder = GridGraphBuilder()
    df = pd.read_parquet(SSEN_METADATA)
    return builder.build_from_metadata(df), df


GNN_NUM_NODES = 44  # GNN checkpoint was trained on 44-node graphs


@st.cache_resource
def load_gnn_graph():
    """Build a 44-node graph compatible with the trained GNN checkpoint.

    GridGraphBuilder produces 49 node_type/feature entries but the GNN
    was trained on 44 nodes.  We slice tensors and filter edges to
    produce a Data object the GNN can consume without dimension errors.
    """
    from torch_geometric.data import Data

    full_graph, meta_df = load_graph_data()
    n = GNN_NUM_NODES

    # Slice node attributes to first n nodes
    node_type = full_graph.node_type[:n]
    x = full_graph.x[:n] if full_graph.x is not None else None

    # Filter edges: keep only those where both src and dst < n
    ei = full_graph.edge_index
    mask = (ei[0] < n) & (ei[1] < n)
    edge_index = ei[:, mask]

    data = Data(x=x, edge_index=edge_index, node_type=node_type)
    data.num_nodes = n
    return data


@st.cache_resource
def load_hybrid_verifier(_graph_data):
    """Load HybridVerifierAgent with trained GNN checkpoint."""
    from fyp.selfplay.hybrid_verifier import create_hybrid_verifier

    return create_hybrid_verifier(
        config_path=HYBRID_CONFIG,
        graph_data=_graph_data,
    )


@st.cache_resource
def load_proposer():
    """Load ProposerAgent for scenario generation."""
    from fyp.selfplay.proposer import ProposerAgent

    return ProposerAgent(
        ssen_constraints_path=SSEN_CONSTRAINTS,
        random_seed=None,
    )


@st.cache_resource
def load_gnn_model():
    """Load trained GATVerifier model."""
    from fyp.gnn.gat_verifier import GATVerifier

    checkpoint = torch.load(GNN_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = GATVerifier(temporal_features=5, hidden_channels=64, num_layers=3, heads=4)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_benchmark_results():
    """Load saved benchmark results JSON."""
    with open(BENCHMARK_JSON) as f:
        return json.load(f)


@st.cache_data
def load_ablation_results():
    """Load saved ablation results JSON."""
    with open(ABLATION_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "System overview",
    "Live anomaly detection",
    "Grid topology",
    "Evaluation results",
    "Self-play training",
])


# =========================================================================
# TAB 1 -- System overview
# =========================================================================
with tabs[0]:
    st.header("System overview")

    # -- Metric cards --
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Records ingested", "281M")
    m2.metric("Datasets", "3", help="UK-DALE, LCL, SSEN")
    m3.metric("Tests passing", "212")
    m4.metric("GNN accuracy", "98.33%")
    m5.metric("Inference latency", "16.56 ms")

    st.divider()

    # -- Research timeline --
    st.subheader("Research timeline")

    milestones = [
        ("Sep 2025", "AZR self-play experiments"),
        ("Oct-Dec 2025", "Negative result discovery"),
        ("Jan 2026", "Pivot to GNN verification"),
        ("Mar 2026", "Hybrid verifier integration"),
        ("Mar 2026", "Evaluation complete"),
    ]

    fig_tl = go.Figure()
    xs = list(range(len(milestones)))
    labels = [m[0] for m in milestones]
    descriptions = [m[1] for m in milestones]

    fig_tl.add_trace(go.Scatter(
        x=xs, y=[0] * len(xs),
        mode="lines+markers+text",
        marker=dict(size=14, color=STEEL, symbol="circle"),
        line=dict(color=STEEL, width=2),
        text=labels,
        textposition="top center",
        textfont=dict(size=11, color="#E0E4E8"),
        hovertext=descriptions,
        hoverinfo="text",
        showlegend=False,
    ))

    for i, desc in enumerate(descriptions):
        fig_tl.add_annotation(
            x=i, y=-0.15, text=desc,
            showarrow=False, font=dict(size=10, color="#A0AAB4"),
            xanchor="center",
        )

    fig_tl.update_layout(
        template=PLOTLY_TEMPLATE,
        height=160,
        margin=dict(l=20, r=20, t=10, b=60),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.4, 0.3]),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    st.divider()

    # -- Architecture diagram --
    st.subheader("Three-layer hybrid verifier architecture")

    fig_arch = go.Figure()

    # ── Palette ──
    A_BLUE = "#5BA3E6"     # Physics layer
    A_TEAL = "#3DBFA0"     # GNN layer
    A_AMBER = "#D4A843"    # Cascade layer
    A_RED = "#E07050"      # Early-exit / decision
    A_GREY = "#8899AA"     # Arrows, secondary text
    A_LIGHT = "#C8D0D8"    # Input/output text
    A_WHITE = "#E8ECF0"    # Primary text
    A_DIM = "#5A6674"      # Faint guides
    A_ENSEMBLE = "#9B7FD4" # Ensemble

    # ── Coordinate system ──
    # X: 0..16   Y: 0..22 (top=22)
    XR = [-0.5, 16.5]
    YR = [-0.8, 22.5]

    def _rect(x0, y0, x1, y1, color, opacity=0.10, width=1.5, dash=None):
        fig_arch.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=color, opacity=opacity,
            line=dict(color=color, width=width, dash=dash),
        )

    def _label(x, y, text, color=A_WHITE, size=11, bold=False, anchor="middle"):
        prefix = "<b>" if bold else ""
        suffix = "</b>" if bold else ""
        fig_arch.add_annotation(
            x=x, y=y, text=f"{prefix}{text}{suffix}",
            showarrow=False, font=dict(size=size, color=color),
            xanchor=anchor if anchor != "middle" else "center",
        )

    def _arrow(x0, y0, x1, y1, color=A_GREY, width=1.5, dash=None):
        fig_arch.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=width, arrowcolor=color,
        )

    # ================================================================
    # INPUT ROW  (y ~ 21)
    # ================================================================
    _rect(1.5, 20.4, 5.5, 21.6, A_LIGHT, opacity=0.08)
    _label(3.5, 21.2, "Forecast input", A_LIGHT, size=12, bold=True)
    _label(3.5, 20.7, "f(n): per-node values, n = 44 nodes", A_GREY, size=9)

    _rect(10.5, 20.4, 14.5, 21.6, A_LIGHT, opacity=0.08)
    _label(12.5, 21.2, "SSEN graph topology", A_LIGHT, size=12, bold=True)
    _label(12.5, 20.7, "G(V, E): 44 nodes, 60 edges, 3 types", A_GREY, size=9)

    # Arrows from inputs down
    _arrow(3.5, 20.4, 3.5, 19.6)           # forecast -> physics
    _arrow(12.5, 20.4, 12.5, 19.0)         # graph -> right side (long, to GNN)

    # ================================================================
    # LAYER 1: PHYSICS CONSTRAINTS  (y ~ 16-19.5)
    # ================================================================
    _rect(0.3, 15.6, 10.0, 19.5, A_BLUE, opacity=0.06, width=2)
    _label(0.8, 19.1, "Layer 1: Physics constraints", A_BLUE, size=13, bold=True, anchor="left")
    _label(6.5, 19.1, "Tolerance band scoring", A_GREY, size=9)

    # Sub-components
    _rect(0.8, 16.8, 3.5, 18.5, A_BLUE, opacity=0.12)
    _label(2.15, 18.1, "Voltage", A_BLUE, size=11, bold=True)
    _label(2.15, 17.65, "BS EN 50160", A_GREY, size=8)
    _label(2.15, 17.25, "230V nominal", A_WHITE, size=9)
    _label(2.15, 16.9, "Safe: -6% / +8%", A_GREY, size=8)

    _rect(3.9, 16.8, 6.6, 18.5, A_BLUE, opacity=0.12)
    _label(5.25, 18.1, "Capacity", A_BLUE, size=11, bold=True)
    _label(5.25, 17.65, "BS 7671:2018", A_GREY, size=8)
    _label(5.25, 17.25, "15 kW typical", A_WHITE, size=9)
    _label(5.25, 16.9, "100 kW absolute max", A_GREY, size=8)

    _rect(7.0, 16.8, 9.6, 18.5, A_BLUE, opacity=0.12)
    _label(8.3, 18.1, "Ramp rate", A_BLUE, size=11, bold=True)
    _label(8.3, 17.65, "Rate of change", A_GREY, size=8)
    _label(8.3, 17.25, "3.5 kW/interval warn", A_WHITE, size=9)
    _label(8.3, 16.9, "5.0 kW/interval max", A_GREY, size=8)

    # Output annotation
    _label(5.15, 16.15, "Output: severity scores per node [0, 1]", A_BLUE, size=9)
    _label(5.15, 15.75, "Combined = max(voltage, capacity, ramp) per node", A_DIM, size=8)

    # ================================================================
    # EARLY-EXIT DECISION  (y ~ 13.5-15.5)
    # ================================================================
    # Diamond-style decision box
    _rect(2.8, 13.5, 7.2, 15.2, A_RED, opacity=0.10, width=2, dash="dot")
    _label(5.0, 14.7, "Early-exit decision", A_RED, size=12, bold=True)
    _label(5.0, 14.2, "severity > 0.9 ?", A_WHITE, size=11)
    _label(5.0, 13.7, "Auto-detect: voltage scoring skipped if values < 103V", A_DIM, size=8)

    # Arrow from physics down to decision
    _arrow(5.0, 15.6, 5.0, 15.2)

    # YES path -- skip GNN, go right to ensemble
    _label(8.4, 14.7, "YES", A_RED, size=10, bold=True)
    _label(8.4, 14.3, "Skip GNN", A_RED, size=9)
    _arrow(7.2, 14.5, 8.0, 14.5, color=A_RED, width=2)

    # Arrow from YES to ensemble (right side, curves down)
    _rect(10.5, 13.8, 14.5, 15.2, A_RED, opacity=0.06, dash="dot")
    _label(12.5, 14.7, "Early-exit path", A_RED, size=10, bold=True)
    _label(12.5, 14.2, "Use physics score only", A_GREY, size=9)
    _label(12.5, 13.9, "Weights become (1.0, 0.0, 0.0)", A_DIM, size=8)
    _arrow(8.8, 14.5, 10.5, 14.5, color=A_RED, width=1.5, dash="dot")

    # NO path -- continue to GNN
    _label(5.0, 13.1, "NO: continue", A_TEAL, size=9)
    _arrow(5.0, 13.5, 5.0, 12.6, color=A_TEAL, width=2)

    # ================================================================
    # LAYER 2: GNN VERIFIER  (y ~ 8.5-12.5)
    # ================================================================
    _rect(0.3, 8.2, 10.0, 12.5, A_TEAL, opacity=0.06, width=2)
    _label(0.8, 12.1, "Layer 2: GNN verifier", A_TEAL, size=13, bold=True, anchor="left")
    _label(6.5, 12.1, "GATVerifier", A_GREY, size=9)

    # Graph topology input arrow from right
    _arrow(12.5, 13.8, 10.0, 11.0, color=A_GREY, width=1)
    _label(12.0, 12.3, "edge_index, node_type", A_GREY, size=8)

    # Sub-components (2x2 grid)
    _rect(0.8, 10.0, 4.8, 11.7, A_TEAL, opacity=0.12)
    _label(2.8, 11.3, "GATv2Conv attention", A_TEAL, size=11, bold=True)
    _label(2.8, 10.85, "3 layers, 4 heads per layer", A_WHITE, size=9)
    _label(2.8, 10.5, "Dynamic attention (not static GAT)", A_GREY, size=8)
    _label(2.8, 10.15, "concat=True, 64 hidden channels", A_GREY, size=8)

    _rect(5.2, 10.0, 9.6, 11.7, A_TEAL, opacity=0.12)
    _label(7.4, 11.3, "Oversmoothing prevention", A_TEAL, size=11, bold=True)
    _label(7.4, 10.85, "GCNII-style initial residual", A_WHITE, size=9)
    _label(7.4, 10.5, "Learnable alpha per layer", A_GREY, size=8)
    _label(7.4, 10.15, "Preserves node distinguishability", A_GREY, size=8)

    _rect(0.8, 8.5, 4.8, 9.7, A_TEAL, opacity=0.12)
    _label(2.8, 9.3, "Temporal encoder", A_TEAL, size=11, bold=True)
    _label(2.8, 8.9, "1D-Conv, captures local patterns", A_WHITE, size=9)
    _label(2.8, 8.6, "5 temporal features per node", A_GREY, size=8)

    _rect(5.2, 8.5, 9.6, 9.7, A_TEAL, opacity=0.12)
    _label(7.4, 9.3, "Output head", A_TEAL, size=11, bold=True)
    _label(7.4, 8.9, "Sigmoid activation -> [0, 1]", A_WHITE, size=9)
    _label(7.4, 8.6, "Per-node anomaly probability", A_GREY, size=8)

    # ================================================================
    # LAYER 3: CASCADE LOGIC  (y ~ 4.5-7.8)
    # ================================================================
    _arrow(5.0, 8.2, 5.0, 7.8)  # GNN -> cascade
    _arrow(12.5, 13.8, 12.5, 7.8, color=A_GREY, width=1)  # graph -> cascade
    _label(13.0, 10.5, "Graph", A_GREY, size=8)
    _label(13.0, 10.1, "topology", A_GREY, size=8)

    _rect(0.3, 4.5, 14.5, 7.8, A_AMBER, opacity=0.06, width=2)
    _label(0.8, 7.4, "Layer 3: Cascade logic", A_AMBER, size=13, bold=True, anchor="left")
    _label(6.5, 7.4, "Neighbor propagation scoring", A_GREY, size=9)

    _rect(0.8, 5.0, 4.8, 7.0, A_AMBER, opacity=0.12)
    _label(2.8, 6.6, "BFS propagation", A_AMBER, size=11, bold=True)
    _label(2.8, 6.2, "2-hop neighborhood traversal", A_WHITE, size=9)
    _label(2.8, 5.85, "Decay: 0.7 per hop", A_WHITE, size=9)
    _label(2.8, 5.45, "Hop 0: 1.0  Hop 1: 0.70  Hop 2: 0.49", A_GREY, size=8)
    _label(2.8, 5.1, "Max 30% of nodes affected", A_GREY, size=8)

    _rect(5.2, 5.0, 9.6, 7.0, A_AMBER, opacity=0.12)
    _label(7.4, 6.6, "Anomaly aggregation", A_AMBER, size=11, bold=True)
    _label(7.4, 6.2, "Score = f(neighbor anomalies)", A_WHITE, size=9)
    _label(7.4, 5.85, "High score = neighbors also anomalous", A_GREY, size=8)
    _label(7.4, 5.45, "Distinguishes isolated spikes", A_GREY, size=8)
    _label(7.4, 5.1, "from cascading failures", A_GREY, size=8)

    _rect(10.0, 5.0, 14.2, 7.0, A_AMBER, opacity=0.12)
    _label(12.1, 6.6, "Adjacency construction", A_AMBER, size=11, bold=True)
    _label(12.1, 6.2, "From edge_index (COO format)", A_WHITE, size=9)
    _label(12.1, 5.85, "Primary -> Secondary -> LV", A_GREY, size=8)
    _label(12.1, 5.45, "Bidirectional edges", A_GREY, size=8)

    # ================================================================
    # ENSEMBLE  (y ~ 1.5-4)
    # ================================================================
    _arrow(5.0, 4.5, 5.0, 4.0)   # cascade -> ensemble
    _arrow(12.5, 13.8, 14.0, 4.0, color=A_RED, width=1, dash="dot")  # early-exit to ensemble

    _rect(0.3, 1.2, 14.5, 4.0, A_ENSEMBLE, opacity=0.06, width=2)
    _label(0.8, 3.6, "Ensemble combination", A_ENSEMBLE, size=13, bold=True, anchor="left")

    # Weight boxes
    _rect(0.8, 1.6, 4.2, 3.2, A_BLUE, opacity=0.10)
    _label(2.5, 2.85, "Physics score", A_BLUE, size=10, bold=True)
    _label(2.5, 2.4, "w_p = 0.4", A_WHITE, size=12, bold=True)
    _label(2.5, 1.95, "Constraint violations", A_GREY, size=8)

    _rect(4.6, 1.6, 8.0, 3.2, A_TEAL, opacity=0.10)
    _label(6.3, 2.85, "GNN score", A_TEAL, size=10, bold=True)
    _label(6.3, 2.4, "w_g = 0.4", A_WHITE, size=12, bold=True)
    _label(6.3, 1.95, "Learned patterns", A_GREY, size=8)

    _rect(8.4, 1.6, 11.6, 3.2, A_AMBER, opacity=0.10)
    _label(10.0, 2.85, "Cascade score", A_AMBER, size=10, bold=True)
    _label(10.0, 2.4, "w_c = 0.2", A_WHITE, size=12, bold=True)
    _label(10.0, 1.95, "Topology propagation", A_GREY, size=8)

    _rect(12.0, 1.6, 14.2, 3.2, A_RED, opacity=0.08, dash="dot")
    _label(13.1, 2.85, "Early-exit", A_RED, size=10, bold=True)
    _label(13.1, 2.4, "(1, 0, 0)", A_WHITE, size=11, bold=True)
    _label(13.1, 1.95, "Physics only", A_GREY, size=8)

    _label(7.5, 1.35, "combined = w_p * physics + w_g * gnn + w_c * cascade    (per node)", A_DIM, size=9)

    # ================================================================
    # OUTPUT  (y ~ 0)
    # ================================================================
    _arrow(7.5, 1.2, 7.5, 0.6)
    _rect(4.5, -0.2, 10.5, 0.6, A_LIGHT, opacity=0.08)
    _label(7.5, 0.4, "Verification reward", A_LIGHT, size=12, bold=True)
    _label(7.5, 0.0, "r in [-1, +1]   |   FN penalty ratio: 2.0", A_GREY, size=9)

    # ================================================================
    # Layout
    # ================================================================
    fig_arch.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=XR),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=YR,
                   scaleanchor="x", scaleratio=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_arch, use_container_width=True)


# =========================================================================
# TAB 2 -- Live anomaly detection
# =========================================================================
with tabs[1]:
    st.header("Live anomaly detection")

    try:
        gnn_graph = load_gnn_graph()
        verifier = load_hybrid_verifier(gnn_graph)
        proposer = load_proposer()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # -- Controls --
    c1, c2, c3 = st.columns(3)
    scenario_type = c1.selectbox(
        "Scenario type",
        ["EV_SPIKE", "COLD_SNAP", "PEAK_SHIFT", "OUTAGE", "MISSING_DATA"],
    )
    severity = c2.slider("Severity", 0.1, 1.0, 0.6, 0.05)
    horizon = c3.selectbox("Forecast horizon (intervals)", [24, 48, 96], index=1)

    if st.button("Generate and evaluate"):
      try:
        with st.spinner("Running hybrid verifier pipeline..."):
            # Generate scenario
            np.random.seed(int(time.time()) % 10000)
            context = np.random.rand(336) * 5 + 0.5  # Realistic kW range
            scenario = proposer.propose_scenario(
                context,
                forecast_horizon=horizon,
                graph_data=gnn_graph,
            )
            # Override type and magnitude
            scenario.scenario_type = scenario_type
            scenario.magnitude = severity * 3.0

            # Apply scenario to create forecast
            forecast_1d = scenario.apply_to_timeseries(context[:horizon])

            # Pad or trim to GNN node count (44)
            eval_input = np.zeros(GNN_NUM_NODES)
            n_copy = min(len(forecast_1d), GNN_NUM_NODES)
            eval_input[:n_copy] = forecast_1d[:n_copy]

            # Run through hybrid verifier
            reward, details = verifier.evaluate(
                eval_input,
                scenario=scenario,
                return_details=True,
            )

        # -- Results layout --
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Forecast with anomaly injection")
            fig_fc = go.Figure()
            t = np.arange(len(forecast_1d))
            baseline = context[:horizon]

            fig_fc.add_trace(go.Scatter(
                x=t, y=baseline, name="Baseline",
                line=dict(color=STEEL, width=1.5, dash="dot"),
            ))
            fig_fc.add_trace(go.Scatter(
                x=t, y=forecast_1d, name="With anomaly",
                line=dict(color=RED, width=2),
            ))

            # Shade anomalous region
            start = scenario.metadata.get("start_offset", 0)
            end = min(start + scenario.duration, len(forecast_1d))
            fig_fc.add_vrect(
                x0=start, x1=end,
                fillcolor=RED, opacity=0.08,
                line_width=0, annotation_text="Anomaly region",
                annotation_position="top left",
                annotation_font_color=RED,
            )

            fig_fc.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Interval (30 min)",
                yaxis_title="Value (kW)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with right:
            st.subheader("Layer scores")

            breakdown = details.get("_breakdown", {})
            physics_mean = float(np.mean(breakdown.get("physics_scores", [0])))
            gnn_mean = float(np.mean(breakdown.get("gnn_scores", [0])))
            cascade_mean = float(np.mean(breakdown.get("cascade_scores", [0])))
            combined_mean = float(np.mean(breakdown.get("combined_scores", [0])))

            layers = ["Physics", "GNN", "Cascade", "Ensemble"]
            scores = [physics_mean, gnn_mean, cascade_mean, combined_mean]
            colors = [STEEL if s < 0.5 else RED for s in scores]

            fig_bars = go.Figure(go.Bar(
                y=layers, x=scores,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
                textfont=dict(size=12),
            ))
            fig_bars.update_layout(
                template=PLOTLY_TEMPLATE,
                height=250,
                margin=dict(l=80, r=40, t=10, b=30),
                xaxis=dict(range=[0, 1.15], title="Score"),
                yaxis=dict(autorange="reversed"),
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_bars, use_container_width=True)

            reward_color = TEAL if reward > 0 else RED
            st.markdown(
                f"**Verification reward:** "
                f"<span style='color:{reward_color};font-size:1.3em'>{reward:.4f}</span>",
                unsafe_allow_html=True,
            )

            early_exits = breakdown.get("early_exit_count", 0)
            st.markdown(f"**Early exits:** {early_exits}/{GNN_NUM_NODES} nodes")

        # -- Expandable details --
        with st.expander("Raw verification details"):
            det_physics = details.get("physics", {})
            det_gnn = details.get("gnn", {})
            det_cascade = details.get("cascade", {})
            cols = st.columns(3)
            cols[0].markdown("**Physics layer**")
            cols[0].json(det_physics)
            cols[1].markdown("**GNN layer**")
            cols[1].json(det_gnn)
            cols[2].markdown("**Cascade layer**")
            cols[2].json(det_cascade)

            st.markdown("**Scenario metadata**")
            meta_display = {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in scenario.metadata.items()
            }
            # Truncate affected_nodes for display
            if "affected_nodes" in meta_display and isinstance(meta_display["affected_nodes"], dict):
                an = meta_display["affected_nodes"]
                if len(an) > 10:
                    meta_display["affected_nodes"] = dict(list(an.items())[:10])
                    meta_display["affected_nodes_truncated"] = f"...{len(an)} total"
            st.json(meta_display)
      except Exception as e:
        st.error(f"Anomaly detection failed: {e}")
        import traceback
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# =========================================================================
# TAB 3 -- Grid topology
# =========================================================================
with tabs[2]:
    st.header("Grid topology")

    try:
        graph_data, meta_df = load_graph_data()
    except Exception as e:
        st.error(f"Failed to load graph: {e}")
        st.stop()

    node_types = graph_data.node_type.numpy()
    # Use node_type length as authoritative count (may differ from num_nodes)
    num_nodes = len(node_types)
    num_edges = graph_data.edge_index.shape[1]

    # Stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Nodes", num_nodes)
    s2.metric("Edges", num_edges)
    s3.metric("Primary substations", int((node_types == 0).sum()))
    s4.metric("LV feeders", int((node_types == 2).sum()))

    st.divider()

    # Build layout with hierarchy
    edge_index = graph_data.edge_index.numpy()

    # Assign positions by type for clear hierarchy
    np.random.seed(42)
    pos_x = np.zeros(num_nodes)
    pos_y = np.zeros(num_nodes)
    type_names = {0: "Primary substation", 1: "Secondary substation", 2: "LV feeder"}
    type_sizes = {0: 18, 1: 12, 2: 7}

    for ntype in [0, 1, 2]:
        mask = node_types == ntype
        count = mask.sum()
        y_base = {0: 2.0, 1: 1.0, 2: 0.0}[ntype]
        pos_x[mask] = np.linspace(0, 4, count) + np.random.randn(count) * 0.1
        pos_y[mask] = y_base + np.random.randn(count) * 0.15

    # Initialize anomaly scores (all zero = no anomaly)
    if "anomaly_scores" not in st.session_state:
        st.session_state.anomaly_scores = np.zeros(num_nodes)

    anomaly_scores = st.session_state.anomaly_scores
    # Ensure anomaly_scores matches current node count
    if len(anomaly_scores) != num_nodes:
        anomaly_scores = np.zeros(num_nodes)
        st.session_state.anomaly_scores = anomaly_scores

    # Cascade injection button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("Inject cascade anomaly"):
            try:
                proposer = load_proposer()
                context = np.random.rand(336) * 3
                scenario = proposer.propose_scenario(
                    context,
                    graph_data=graph_data,
                )
                affected = scenario.metadata.get("affected_nodes", {})
                new_scores = np.zeros(num_nodes)
                for node_idx, magnitude in affected.items():
                    idx = int(node_idx)
                    if idx < num_nodes:
                        new_scores[idx] = float(magnitude)
                st.session_state.anomaly_scores = new_scores
                anomaly_scores = new_scores
                st.rerun()
            except Exception as e:
                st.error(f"Cascade injection failed: {e}")

    with col_info:
        n_affected = int((anomaly_scores > 0).sum())
        if n_affected > 0:
            st.markdown(
                f"Cascade active: {n_affected} nodes affected "
                f"(seeds at magnitude 1.0, decay 0.7 per hop)"
            )
        else:
            st.markdown("No active cascade. Click to inject.")

    # Build network figure
    fig_net = go.Figure()

    # Draw edges (skip any that reference out-of-bounds nodes)
    edge_x, edge_y = [], []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < num_nodes and dst < num_nodes:
            edge_x.extend([pos_x[src], pos_x[dst], None])
            edge_y.extend([pos_y[src], pos_y[dst], None])

    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#4A5568"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Draw nodes by type, colored by anomaly score
    for ntype in [0, 1, 2]:
        mask = node_types == ntype
        indices = np.where(mask)[0]
        scores_subset = anomaly_scores[indices]

        # Color: grey (normal) to red (anomalous)
        node_colors = []
        for s in scores_subset:
            if s > 0.01:
                # Interpolate from light orange to dark red
                r = int(153 + (255 - 153) * (1 - s))
                g = int(60 * (1 - s))
                b = int(29 * (1 - s))
                node_colors.append(f"rgb({r},{g},{b})")
            else:
                node_colors.append({0: "#B0BEC5", 1: STEEL, 2: TEAL}[ntype])

        hover_texts = [
            f"Node {idx} ({type_names[ntype]})\nAnomaly: {anomaly_scores[idx]:.2f}"
            for idx in indices
        ]

        fig_net.add_trace(go.Scatter(
            x=pos_x[indices],
            y=pos_y[indices],
            mode="markers",
            marker=dict(
                size=type_sizes[ntype],
                color=node_colors,
                line=dict(width=1, color=WHITE),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=type_names[ntype],
        ))

    fig_net.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_net, use_container_width=True)


# =========================================================================
# TAB 4 -- Evaluation results
# =========================================================================
with tabs[3]:
    st.header("Evaluation results")

    try:
        bench_results = load_benchmark_results()
        ablation_results = load_ablation_results()
    except Exception as e:
        st.error(f"Failed to load evaluation results: {e}")
        st.stop()

    configs = bench_results.get("configurations", {})

    # -- ROC-AUC bar chart --
    st.subheader("ROC-AUC across configurations")

    sorted_configs = sorted(
        configs.items(),
        key=lambda x: x[1].get("roc_auc") or 0,
        reverse=True,
    )
    names = [c[0] for c in sorted_configs]
    roc_aucs = [c[1].get("roc_auc") or 0 for c in sorted_configs]
    bar_colors = [TEAL if v >= 0.9 else STEEL if v >= 0.5 else RED for v in roc_aucs]

    fig_roc = go.Figure(go.Bar(
        y=names, x=roc_aucs,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in roc_aucs],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_roc.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(l=140, r=60, t=10, b=40),
        xaxis=dict(range=[0, 1.12], title="ROC-AUC"),
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    # -- Ablation table --
    left_abl, right_abl = st.columns(2)

    with left_abl:
        st.subheader("Component ablation")

        component = ablation_results.get("component_isolation", {})
        abl_rows = []
        for name, metrics in sorted(
            component.items(),
            key=lambda x: x[1].get("roc_auc") or 0,
            reverse=True,
        ):
            roc = metrics.get("roc_auc")
            opt_f1 = metrics.get("optimal_f1")
            abl_rows.append({
                "Configuration": name,
                "ROC-AUC": f"{roc:.4f}" if roc is not None else "N/A",
                "Optimal F1": f"{opt_f1:.4f}" if opt_f1 is not None else "N/A",
            })

        st.dataframe(
            pd.DataFrame(abl_rows),
            use_container_width=True,
            hide_index=True,
        )

    with right_abl:
        st.subheader("Component insights")

        st.markdown(
            "- **GNN** (GATv2Conv) is the primary discriminator -- "
            "ROC-AUC=1.0 alone and in all combinations\n"
            "- **Cascade logic** adds real topological signal (ROC-AUC=0.89) "
            "by detecting neighbor propagation patterns\n"
            "- **Physics layer** provides no discrimination on synthetic data "
            "(ROC-AUC=0.50) -- expected since test data uses normalised "
            "features, not real voltages\n"
            "- Physics layer auto-detects data type: voltage scoring is "
            "skipped when values are below 103V threshold\n"
            "- **Autoencoder** is the strongest standalone baseline "
            "(ROC-AUC=1.0, 0.08ms latency)"
        )

    st.divider()

    # -- Early-exit sweep --
    st.subheader("Early-exit threshold sweep")

    sweep = ablation_results.get("early_exit_sweep", {}).get("sweep_results", [])
    if sweep:
        thresholds = [p["threshold"] for p in sweep]
        sweep_roc = [p.get("roc_auc") or 0 for p in sweep]
        sweep_latency = [p.get("mean_latency_ms", 0) for p in sweep]

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=thresholds, y=sweep_roc,
            mode="lines+markers",
            name="ROC-AUC",
            line=dict(color=TEAL, width=2),
            marker=dict(size=8),
            yaxis="y",
        ))
        fig_sweep.add_trace(go.Scatter(
            x=thresholds, y=sweep_latency,
            mode="lines+markers",
            name="Latency (ms)",
            line=dict(color=STEEL, width=2, dash="dot"),
            marker=dict(size=8),
            yaxis="y2",
        ))
        fig_sweep.add_annotation(
            text="Post-fix: voltage auto-detection heuristic prevents<br>early-exit degradation. See debugging narrative below.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=11, color="#A0AAB4"),
            bgcolor="rgba(30,40,50,0.7)",
            borderpad=8,
        )
        fig_sweep.update_layout(
            template=PLOTLY_TEMPLATE,
            height=300,
            margin=dict(l=60, r=60, t=10, b=40),
            xaxis=dict(title="Early-exit threshold"),
            yaxis=dict(title="ROC-AUC", side="left", range=[0, 1.1]),
            yaxis2=dict(
                title="Latency (ms)", side="right",
                overlaying="y", range=[0, max(sweep_latency) * 1.3],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        st.markdown(
            "At threshold=1.0 (no early exit), the GNN processes all nodes and achieves "
            "perfect discrimination (ROC-AUC=1.0). Lower thresholds cause the physics "
            "layer to route all nodes through early exit, bypassing the GNN entirely. "
            "This was discovered during evaluation debugging when the voltage "
            "auto-detection heuristic was added."
        )

    st.divider()

    # -- Statistical significance --
    st.subheader("Statistical significance")

    sig = ablation_results.get("significance_tests", {})
    for test_name, data in sig.items():
        p_val = data.get("p_value")
        significant = data.get("significant", False)
        method = data.get("method", "unknown")
        label = data.get("test_name", test_name)
        sig_text = "significant" if significant else "not significant"
        color = TEAL if significant else GREY_TEXT

        if p_val is not None:
            st.markdown(
                f"**{label}:** p={p_val:.4f} ({method}) -- "
                f"<span style='color:{color}'>{sig_text}</span>",
                unsafe_allow_html=True,
            )

    # -- Voltage debugging narrative --
    with st.expander("Voltage auto-detection debugging narrative"):
        st.markdown(
            "During initial evaluation, all hybrid verifier configurations "
            "produced identical ROC-AUC=0.50 (random). Investigation revealed:\n\n"
            "1. The synthetic test data generates forecast values in [0, 1] "
            "(averaged graph node features)\n"
            "2. The physics layer's voltage constraint expects values around "
            "230V (BS EN 50160 standard)\n"
            "3. Values of 0.4V are far below the 207V lower limit, producing "
            "severity=1.0 for ALL nodes\n"
            "4. With all physics scores at 1.0, any early_exit_threshold < 1.0 "
            "triggers 100% early exit\n"
            "5. The GNN (the actual discriminator) never runs\n\n"
            "**Fix:** Added a voltage auto-detect heuristic to PhysicsConstraintLayer. "
            "If max(abs(forecast)) < voltage_lower_limit / 2 (~103V), voltage "
            "scoring is skipped. This matches the existing capacity auto-detect "
            "pattern already in the code.\n\n"
            "**Result:** After the fix, hybrid_full achieves ROC-AUC=1.0 and the "
            "ablation study reveals clear component differentiation."
        )


# =========================================================================
# TAB 5 -- Self-play training
# =========================================================================
with tabs[4]:
    st.header("Self-play training")

    st.markdown(
        "The propose-solve-verify loop follows the Absolute Zero Reasoner (AZR) "
        "paradigm. Experiments showed this approach does not outperform baselines "
        "on periodic time series with strong temporal patterns. This negative "
        "result motivated the pivot to topology-aware GNN verification."
    )

    st.divider()

    num_episodes = st.slider("Number of episodes", 1, 10, 5)

    if st.button("Run training"):
        try:
            from fyp.selfplay.proposer import ProposerAgent
            from fyp.selfplay.solver import SolverAgent
            from fyp.selfplay.verifier import VerifierAgent
            from fyp.selfplay.trainer import SelfPlayTrainer

            gnn_graph_train = load_gnn_graph()

            with st.spinner("Initializing agents..."):
                proposer = ProposerAgent(
                    ssen_constraints_path=SSEN_CONSTRAINTS,
                    random_seed=42,
                )
                solver = SolverAgent(device="cpu")
                base_verifier = VerifierAgent(
                    ssen_constraints_path=SSEN_CONSTRAINTS,
                )
                trainer = SelfPlayTrainer(
                    proposer=proposer,
                    solver=solver,
                    verifier=base_verifier,
                    graph_data=gnn_graph_train,
                )

            progress = st.progress(0)
            status = st.empty()

            rewards_history = []
            solver_losses = []
            curriculum_levels = []
            scenario_types = []

            for ep in range(num_episodes):
                status.markdown(f"Episode {ep + 1}/{num_episodes}")
                progress.progress((ep + 1) / num_episodes)

                # Generate synthetic training batch
                batch = [
                    (np.random.rand(336) * 5, np.random.rand(48) * 5)
                    for _ in range(3)
                ]

                try:
                    metrics = trainer.train_episode(batch)
                    rewards_history.append(
                        metrics.get("avg_verification_reward", 0)
                    )
                    solver_losses.append(
                        metrics.get("avg_solver_loss", 0)
                    )
                    curriculum_levels.append(
                        metrics.get("scenario_diversity", 0)
                    )
                    # scenarios is a list of types per batch item
                    ep_scenarios = metrics.get("scenarios", [])
                    scenario_types.extend(
                        [str(s) for s in ep_scenarios] if ep_scenarios else ["UNKNOWN"]
                    )
                except Exception as ep_err:
                    rewards_history.append(0)
                    solver_losses.append(0)
                    curriculum_levels.append(0)
                    scenario_types.append("ERROR")
                    st.warning(f"Episode {ep + 1} error: {ep_err}")

            progress.progress(1.0)
            status.markdown("Training complete")

            # -- Charts --
            row1_left, row1_right = st.columns(2)

            eps_x = list(range(1, num_episodes + 1))

            with row1_left:
                fig_rew = go.Figure()
                fig_rew.add_trace(go.Scatter(
                    x=eps_x, y=rewards_history,
                    mode="lines+markers",
                    name="Mean reward",
                    line=dict(color=TEAL, width=2),
                    marker=dict(size=6),
                ))
                fig_rew.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="Verification rewards per episode",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40),
                    xaxis_title="Episode",
                    yaxis_title="Reward",
                    **DARK_LAYOUT,
                )
                st.plotly_chart(fig_rew, use_container_width=True)

            with row1_right:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=eps_x, y=solver_losses,
                    mode="lines+markers",
                    name="Solver loss",
                    line=dict(color=RED, width=2),
                    marker=dict(size=6),
                ))
                fig_loss.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="Solver loss per episode",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40),
                    xaxis_title="Episode",
                    yaxis_title="Loss",
                    **DARK_LAYOUT,
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            # Curriculum level chart
            row2_left, row2_right = st.columns(2)

            with row2_left:
                fig_cur = go.Figure()
                fig_cur.add_trace(go.Scatter(
                    x=eps_x, y=curriculum_levels,
                    mode="lines+markers",
                    name="Scenario diversity",
                    line=dict(color=STEEL, width=2),
                    marker=dict(size=6),
                ))
                fig_cur.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="Scenario diversity per episode",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40),
                    xaxis_title="Episode",
                    yaxis_title="Diversity (unique types / total)",
                    **DARK_LAYOUT,
                )
                st.plotly_chart(fig_cur, use_container_width=True)

            with row2_right:
                # Scenario distribution
                if scenario_types:
                    type_counts = {}
                    for t in scenario_types:
                        type_counts[t] = type_counts.get(t, 0) + 1

                    fig_dist = go.Figure(go.Bar(
                        x=list(type_counts.keys()),
                        y=list(type_counts.values()),
                        marker_color=CHART_COLORS[:len(type_counts)],
                    ))
                    fig_dist.update_layout(
                        template=PLOTLY_TEMPLATE,
                        title="Scenario type distribution",
                        height=300,
                        margin=dict(l=50, r=20, t=40, b=40),
                        xaxis_title="Scenario type",
                        yaxis_title="Count",
                        **DARK_LAYOUT,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
