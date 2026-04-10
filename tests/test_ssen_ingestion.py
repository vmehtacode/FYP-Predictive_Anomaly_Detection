"""Tests for SSEN time-series ingestion with metadata enrichment.

This module tests the enhanced SSEN ingestor that processes real operational
distribution network data with metadata enrichment.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from fyp.ingestion.ssen_ingestor import SSENIngestor


class TestSSENTimeSeriesIngestion:
    """Test suite for SSEN time-series ingestion with real consumption data."""

    def test_sample_ingestion_completes(self):
        """Test that sample-based ingestion runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Check output exists
            output_dir = Path(tmpdir) / "ssen_data"
            assert output_dir.exists()
            assert list(output_dir.glob("*.parquet"))

    def test_schema_compliance(self):
        """Test that output schema matches unified EnergyReading schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Read output
            output_dir = Path(tmpdir) / "ssen_data"
            schema = pq.read_schema(list(output_dir.glob("*.parquet"))[0])

            # Verify required columns
            required_cols = {
                "dataset",
                "entity_id",
                "ts_utc",
                "interval_mins",
                "energy_kwh",
                "source",
                "extras",
            }
            actual_cols = set(schema.names)
            assert required_cols == actual_cols

    def test_metadata_enrichment(self):
        """Test that consumption records are enriched with feeder metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load and parse
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # Check extras contains enrichment
            first_extras = json.loads(df["extras"][0])
            enrichment_fields = {
                "source_file",
                "ingestion_version",
                "device_count",
                "dno_name",
            }

            assert enrichment_fields.issubset(set(first_extras.keys()))
            assert first_extras["ingestion_version"] == "v2.1_timeseries"

    def test_energy_unit_conversion(self):
        """Test that Wh is correctly converted to kWh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load data
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # Energy should be in kWh (positive, reasonable scale)
            assert (df["energy_kwh"] >= 0).all()
            # Typical feeder consumption: 0.1 to 10,000 kWh per 30min
            assert df["energy_kwh"].max() < 1_000_000

    def test_interval_standardization(self):
        """Test that all records have 30-minute intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load data
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # All intervals should be 30 minutes
            assert (df["interval_mins"] == 30).all()

    def test_dataset_identifier(self):
        """Test that dataset column is correctly set to 'ssen'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load data
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # All records should be 'ssen' dataset
            assert (df["dataset"] == "ssen").all()

    def test_entity_id_format(self):
        """Test that entity IDs follow 'ssen_feeder_<id>' format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load data
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # Check entity ID format
            entity_ids = df["entity_id"].to_list()
            assert all(eid.startswith("ssen_feeder_") for eid in entity_ids)

    def test_timezone_awareness(self):
        """Test that timestamps are UTC timezone-aware."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Load with pandas to check timezone
            output_dir = Path(tmpdir) / "ssen_data"
            df_pd = pd.read_parquet(output_dir)

            # Check timezone
            assert df_pd["ts_utc"].dt.tz is not None
            assert str(df_pd["ts_utc"].dt.tz) == "UTC"

    def test_stats_tracking(self):
        """Test that ingestion statistics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Check stats
            assert ingestor.stats["processed"] > 0
            assert ingestor.stats["errors"] == 0

    def test_summary_file_creation(self):
        """Test that ingestion summary JSON is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=True,
            )
            ingestor.run()

            # Check summary exists
            summary_path = Path(tmpdir) / "ingestion_summary.json"
            assert summary_path.exists()

            # Verify contents
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary["dataset"] == "ssen"
            assert "stats" in summary
            assert summary["use_samples"] is True


@pytest.mark.skipif(
    not (
        Path(
            "data/raw/ssen/ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv"
        ).exists()
    ),
    reason="Real SSEN time-series CSV not available",
)
class TestSSENRealDataIngestion:
    """Tests that require the full SSEN time-series CSV file."""

    def test_full_csv_ingestion(self):
        """Test ingestion of full SSEN time-series CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=False,
            )
            ingestor.run()

            # Verify output
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # Should have ~100K records
            assert len(df) > 90_000
            # Should have 28 unique feeders
            assert df["entity_id"].n_unique() == 28

    def test_metadata_lookup_loading(self):
        """Test that feeder metadata lookup is correctly loaded."""
        ingestor = SSENIngestor(
            input_root=Path("data/raw"),
            output_root=Path(tempfile.gettempdir()),
            use_samples=False,
        )

        # Load metadata
        feeder_lookup = ingestor._load_feeder_lookup()

        # Should have 100K feeders
        assert len(feeder_lookup) == 100_000
        # Should have expected columns
        expected_cols = {
            "lv_feeder_id",
            "dno_name",
            "secondary_substation_id",
            "total_mpan_count",
        }
        assert expected_cols.issubset(set(feeder_lookup.columns))

    def test_customer_count_preservation(self):
        """Test that total_mpan_count is preserved in enrichment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = SSENIngestor(
                input_root=Path("data/raw"),
                output_root=Path(tmpdir),
                use_samples=False,
            )
            ingestor.run()

            # Load data
            output_dir = Path(tmpdir) / "ssen_data"
            df = pl.scan_parquet(str(output_dir / "*.parquet")).collect()

            # Parse extras to check customer counts
            df_with_counts = df.with_columns(
                [
                    pl.col("extras")
                    .map_elements(
                        lambda x: json.loads(x).get("total_mpan_count"),
                        return_dtype=pl.Float64,
                    )
                    .alias("mpan_count")
                ]
            )

            # Most records should have customer counts
            non_null_count = df_with_counts.filter(
                pl.col("mpan_count").is_not_null()
            ).height
            assert non_null_count / len(df) > 0.95  # At least 95% should have counts
