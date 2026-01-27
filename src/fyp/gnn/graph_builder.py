"""Graph construction pipeline for SSEN distribution network topology.

This module transforms SSEN feeder/substation metadata into PyTorch Geometric
Data objects, representing the UK distribution network as a graph structure
that captures physical topology.

The graph has three node types:
    - Type 0: Primary substations (highest level)
    - Type 1: Secondary substations / feeders (mid level)
    - Type 2: LV feeders (lowest level, connect to households)

Edges are bidirectional, representing physical connectivity in the network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GridGraphBuilder:
    """Build PyG graphs from SSEN distribution network topology.

    This builder transforms SSEN metadata DataFrames into PyTorch Geometric
    Data objects suitable for GNN-based anomaly detection.

    The resulting graph has:
        - Three-level node hierarchy (substations -> feeders -> households)
        - Bidirectional edges representing physical connectivity
        - Node features including type encoding and optional metadata
        - Explicit num_nodes for safe handling of isolated nodes

    Example:
        >>> builder = GridGraphBuilder()
        >>> data = builder.build_from_metadata(metadata_df)
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")

    Attributes:
        exclude_incomplete: If True, skip nodes with missing metadata
    """

    # Node type constants
    NODE_TYPE_PRIMARY_SUBSTATION = 0
    NODE_TYPE_SECONDARY_SUBSTATION = 1
    NODE_TYPE_LV_FEEDER = 2

    # Expected columns in SSEN metadata
    REQUIRED_COLUMNS = [
        "primary_substation_id",
        "secondary_substation_id",
        "lv_feeder_id",
    ]

    def __init__(self, exclude_incomplete: bool = True) -> None:
        """Initialize the graph builder.

        Args:
            exclude_incomplete: If True, skip nodes with missing metadata
                (recommended for cleaner training signal). Defaults to True.
        """
        self.exclude_incomplete = exclude_incomplete
        self._node_to_idx: dict[str, int] = {}
        self._idx_to_node: dict[int, str] = {}

    def build_from_metadata(
        self,
        metadata_df: pd.DataFrame,
        node_features: dict[str, torch.Tensor] | None = None,
    ) -> Data:
        """Build graph from SSEN metadata DataFrame.

        Transforms the hierarchical SSEN network structure into a PyG Data object.
        The graph captures the three-level hierarchy:
            Primary Substations -> Secondary Substations -> LV Feeders

        Args:
            metadata_df: DataFrame with SSEN metadata. Required columns:
                - lv_feeder_id: Low voltage feeder identifier
                - secondary_substation_id: Secondary substation identifier
                - primary_substation_id: Primary substation identifier
                Optional columns:
                - total_mpan_count: Number of customer meters per feeder
                - postcode: Geographic location

            node_features: Optional dict mapping node IDs to feature tensors.
                If not provided, default features are generated (one-hot type
                encoding + optional log(total_mpan_count + 1)).

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: COO format edges [2, num_edges]
                - node_type: Node type tensor (0=primary, 1=secondary, 2=lv_feeder)
                - node_ids: List of original node IDs for reverse lookup
                - num_nodes: Explicit node count (handles isolated nodes)

        Raises:
            ValueError: If required columns are missing from metadata_df
        """
        # Validate required columns
        self._validate_columns(metadata_df)

        # Clean data if exclude_incomplete is set
        if self.exclude_incomplete:
            clean_df = self._clean_data(metadata_df)
        else:
            clean_df = metadata_df.copy()

        if len(clean_df) == 0:
            logger.warning("No valid rows after cleaning, returning empty graph")
            return self._create_empty_graph()

        # Extract unique nodes by type
        primary_substations = self._extract_unique_nodes(
            clean_df, "primary_substation_id"
        )
        secondary_substations = self._extract_unique_nodes(
            clean_df, "secondary_substation_id"
        )
        lv_feeders = self._extract_unique_nodes(clean_df, "lv_feeder_id")

        # Build node ID mappings
        self._build_node_mappings(
            primary_substations, secondary_substations, lv_feeders
        )

        num_nodes = len(self._node_to_idx)
        logger.info(
            f"Extracted {len(primary_substations)} primary substations, "
            f"{len(secondary_substations)} secondary substations, "
            f"{len(lv_feeders)} LV feeders ({num_nodes} total nodes)"
        )

        # Build edges (COO format)
        edge_index = self._build_edges(clean_df)
        logger.info(f"Built {edge_index.size(1)} edges (bidirectional)")

        # Build node type tensor
        node_type = self._build_node_types(
            len(primary_substations),
            len(secondary_substations),
            len(lv_feeders),
        )

        # Build node features
        if node_features is not None:
            x = self._build_custom_features(node_features)
        else:
            x = self._build_default_features(clean_df, node_type)

        # Build node ID list for reverse lookup
        node_ids = [self._idx_to_node[i] for i in range(num_nodes)]

        # Create PyG Data object with explicit num_nodes
        data = Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            num_nodes=num_nodes,
        )

        # Store node IDs as attribute (not a tensor)
        data.node_ids = node_ids

        return data

    def build_from_parquet(self, parquet_path: Path | str) -> Data:
        """Convenience method to load from SSEN metadata parquet.

        Args:
            parquet_path: Path to SSEN metadata parquet file.
                Expected to have columns matching SSEN feeder lookup format.

        Returns:
            PyG Data object (see build_from_metadata for details)

        Raises:
            FileNotFoundError: If parquet file doesn't exist
            ValueError: If required columns are missing
        """
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logger.info(f"Loading SSEN metadata from: {parquet_path}")
        metadata_df = pd.read_parquet(parquet_path)

        # Standardize column names (lowercase with underscores)
        metadata_df.columns = [
            col.lower().replace(" ", "_") for col in metadata_df.columns
        ]

        return self.build_from_metadata(metadata_df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If any required columns are missing
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            available = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {available}"
            )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing required ID columns.

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with only rows that have all required IDs
        """
        original_len = len(df)
        clean_df = df.dropna(subset=self.REQUIRED_COLUMNS)
        dropped = original_len - len(clean_df)

        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows with missing IDs "
                f"({dropped / original_len * 100:.1f}%)"
            )

        return clean_df

    def _extract_unique_nodes(self, df: pd.DataFrame, column: str) -> list[str]:
        """Extract unique node IDs from a column.

        Args:
            df: DataFrame with node IDs
            column: Column name to extract from

        Returns:
            Sorted list of unique node ID strings
        """
        unique_ids = df[column].dropna().unique()
        # Convert to strings and sort for deterministic ordering
        return sorted([str(node_id).strip() for node_id in unique_ids])

    def _build_node_mappings(
        self,
        primary_substations: list[str],
        secondary_substations: list[str],
        lv_feeders: list[str],
    ) -> None:
        """Build bidirectional node ID <-> index mappings.

        Node ordering: primary substations first, then secondary, then LV feeders.
        This ordering matches the node_type tensor construction.

        Args:
            primary_substations: List of primary substation IDs
            secondary_substations: List of secondary substation IDs
            lv_feeders: List of LV feeder IDs
        """
        self._node_to_idx = {}
        self._idx_to_node = {}

        idx = 0
        for node_id in primary_substations:
            self._node_to_idx[node_id] = idx
            self._idx_to_node[idx] = node_id
            idx += 1

        for node_id in secondary_substations:
            self._node_to_idx[node_id] = idx
            self._idx_to_node[idx] = node_id
            idx += 1

        for node_id in lv_feeders:
            self._node_to_idx[node_id] = idx
            self._idx_to_node[idx] = node_id
            idx += 1

    def _build_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Build bidirectional edges in COO format.

        Creates edges between:
            - Primary substations <-> Secondary substations
            - Secondary substations <-> LV feeders

        All edges are bidirectional (each physical connection = 2 directed edges).

        Args:
            df: DataFrame with network hierarchy columns

        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        edges: list[list[int]] = []

        # Build edge sets to avoid duplicates
        ps_to_ss_edges: set[tuple[int, int]] = set()
        ss_to_lv_edges: set[tuple[int, int]] = set()

        for _, row in df.iterrows():
            ps_id = str(row["primary_substation_id"]).strip()
            ss_id = str(row["secondary_substation_id"]).strip()
            lv_id = str(row["lv_feeder_id"]).strip()

            # Get indices
            ps_idx = self._node_to_idx.get(ps_id)
            ss_idx = self._node_to_idx.get(ss_id)
            lv_idx = self._node_to_idx.get(lv_id)

            # Primary -> Secondary edges (bidirectional)
            if ps_idx is not None and ss_idx is not None:
                edge = (min(ps_idx, ss_idx), max(ps_idx, ss_idx))
                ps_to_ss_edges.add(edge)

            # Secondary -> LV feeder edges (bidirectional)
            if ss_idx is not None and lv_idx is not None:
                edge = (min(ss_idx, lv_idx), max(ss_idx, lv_idx))
                ss_to_lv_edges.add(edge)

        # Convert to bidirectional edge list
        for i, j in ps_to_ss_edges:
            edges.append([i, j])
            edges.append([j, i])

        for i, j in ss_to_lv_edges:
            edges.append([i, j])
            edges.append([j, i])

        if not edges:
            # Return empty edge tensor with correct shape
            return torch.zeros((2, 0), dtype=torch.long)

        # Convert to COO format tensor
        # Pattern from research: tensor(...).t().contiguous()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index

    def _build_node_types(
        self,
        num_primary: int,
        num_secondary: int,
        num_lv: int,
    ) -> torch.Tensor:
        """Build node type tensor.

        Args:
            num_primary: Number of primary substations
            num_secondary: Number of secondary substations
            num_lv: Number of LV feeders

        Returns:
            Tensor of shape [num_nodes] with type values 0, 1, or 2
        """
        types = (
            [self.NODE_TYPE_PRIMARY_SUBSTATION] * num_primary
            + [self.NODE_TYPE_SECONDARY_SUBSTATION] * num_secondary
            + [self.NODE_TYPE_LV_FEEDER] * num_lv
        )
        return torch.tensor(types, dtype=torch.long)

    def _build_default_features(
        self,
        df: pd.DataFrame,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        """Build default node features.

        Default features:
            - One-hot node type encoding [3 dims]
            - log(total_mpan_count + 1) if available [1 dim]

        Total: 4-dimensional features

        Args:
            df: DataFrame with metadata
            node_type: Node type tensor

        Returns:
            Feature tensor of shape [num_nodes, 4]
        """
        num_nodes = len(node_type)

        # One-hot encoding for node types (3 dims)
        type_one_hot = torch.zeros(num_nodes, 3)
        type_one_hot.scatter_(1, node_type.unsqueeze(1), 1.0)

        # MPAN count feature (1 dim)
        mpan_features = torch.zeros(num_nodes, 1)

        if "total_mpan_count" in df.columns:
            # Build node ID to MPAN count mapping
            mpan_counts: dict[str, float] = {}

            for _, row in df.iterrows():
                lv_id = str(row["lv_feeder_id"]).strip()
                if pd.notna(row.get("total_mpan_count")):
                    mpan_counts[lv_id] = float(row["total_mpan_count"])

            # Fill MPAN features for LV feeders
            for node_id, idx in self._node_to_idx.items():
                if node_id in mpan_counts:
                    # Use log(count + 1) for better scaling
                    mpan_features[idx, 0] = torch.log(
                        torch.tensor(mpan_counts[node_id] + 1.0)
                    )

        # Concatenate features
        x = torch.cat([type_one_hot, mpan_features], dim=1)

        return x

    def _build_custom_features(
        self,
        node_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build feature tensor from custom node features.

        Args:
            node_features: Dict mapping node IDs to feature tensors

        Returns:
            Feature tensor of shape [num_nodes, feature_dim]

        Raises:
            ValueError: If feature dimensions are inconsistent
        """
        num_nodes = len(self._node_to_idx)

        # Determine feature dimension from first entry
        if not node_features:
            return torch.zeros(num_nodes, 1)

        first_feat = next(iter(node_features.values()))
        feat_dim = first_feat.size(-1) if first_feat.dim() > 0 else 1

        # Initialize with zeros
        x = torch.zeros(num_nodes, feat_dim)

        # Fill in features
        for node_id, feat in node_features.items():
            if node_id in self._node_to_idx:
                idx = self._node_to_idx[node_id]
                if feat.dim() == 0:
                    x[idx, 0] = feat.item()
                else:
                    x[idx] = feat

        return x

    def _create_empty_graph(self) -> Data:
        """Create an empty graph for edge cases.

        Returns:
            PyG Data object with zero nodes and edges
        """
        return Data(
            x=torch.zeros(0, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            node_type=torch.zeros(0, dtype=torch.long),
            num_nodes=0,
        )

    def get_node_id(self, idx: int) -> str | None:
        """Get node ID from index.

        Args:
            idx: Node index

        Returns:
            Node ID string, or None if index not found
        """
        return self._idx_to_node.get(idx)

    def get_node_idx(self, node_id: str) -> int | None:
        """Get node index from ID.

        Args:
            node_id: Node ID string

        Returns:
            Node index, or None if ID not found
        """
        return self._node_to_idx.get(node_id)
