"""
Tests to validate complete ingestion.
Run after ingestion to ensure everything is correct.

Usage:
    pytest tests/test_ingestion_complete.py -v
"""

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class TestLCLIngestion:
    """Test LCL dataset ingestion quality."""

    @pytest.fixture
    def lcl_path(self):
        """LCL dataset path fixture."""
        path = DATA_PROCESSED / "dataset=lcl"
        if not path.exists():
            pytest.skip("LCL not yet ingested")
        return path

    def test_lcl_exists(self, lcl_path):
        """Test LCL processed data exists."""
        assert lcl_path.exists()
        parquet_files = list(lcl_path.rglob("*.parquet"))
        assert len(parquet_files) > 0, "No Parquet files found"

    def test_lcl_schema(self, lcl_path):
        """Test LCL has correct schema."""
        schema = pq.read_schema(list(lcl_path.rglob("*.parquet"))[0])

        required_columns = [
            "dataset",
            "entity_id",
            "ts_utc",
            "interval_mins",
            "energy_kwh",
            "source",
            "extras",
        ]

        for col in required_columns:
            assert col in schema.names, f"Missing column: {col}"

    def test_lcl_intervals(self, lcl_path):
        """Test LCL has 30-minute intervals."""
        df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(10000).collect()
        intervals = df["interval_mins"].unique().to_list()

        assert 30 in intervals, f"Expected interval 30, found {intervals}"
        assert len(intervals) == 1, f"Multiple intervals found: {intervals}"

    def test_lcl_timezone(self, lcl_path):
        """Test LCL timestamps are UTC."""
        df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(1000).collect()

        # Check timezone-aware
        assert (
            "UTC" in str(df["ts_utc"].dtype) or "utc" in str(df["ts_utc"].dtype).lower()
        ), f"Timestamps not UTC, got: {df['ts_utc'].dtype}"

    def test_lcl_energy_values(self, lcl_path):
        """Test LCL energy values are reasonable."""
        df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(100000).collect()

        # No negative values
        negative_count = (df["energy_kwh"] < 0).sum()
        assert negative_count == 0, f"Found {negative_count} negative energy values"

        # Reasonable range (0-10 kWh per 30min for household)
        max_energy = df["energy_kwh"].max()
        assert max_energy < 50, f"Unrealistic max energy: {max_energy} kWh"

    def test_lcl_completeness(self, lcl_path):
        """Test LCL ingestion is >95% complete."""
        # Estimate record count
        parquet_files = list(lcl_path.rglob("*.parquet"))

        # Sample and extrapolate
        sample_size = min(100, len(parquet_files))
        sample_files = parquet_files[:sample_size]

        total_in_sample = 0
        for f in sample_files:
            try:
                metadata = pq.ParquetFile(f).metadata
                total_in_sample += metadata.num_rows
            except Exception:
                pass

        if sample_size > 0:
            avg_per_file = total_in_sample / sample_size
            estimated_total = int(avg_per_file * len(parquet_files))
        else:
            estimated_total = 0

        expected = 167_000_000
        completion_pct = (estimated_total / expected) * 100

        assert (
            completion_pct >= 95
        ), f"LCL only {completion_pct:.1f}% complete ({estimated_total:,}/{expected:,})"

    def test_lcl_acorn_in_extras(self, lcl_path):
        """Test LCL extras contains Acorn demographic data."""
        df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(1000).collect()

        # Parse first extras
        first_extras = json.loads(df["extras"][0])

        # Should have Acorn data
        has_acorn = (
            "acorn_group" in first_extras
            or "acorn" in first_extras
            or "acorn_grouped" in first_extras
            or "tariff_type" in first_extras
        )

        assert (
            has_acorn
        ), f"Acorn/tariff data missing from extras. Keys: {first_extras.keys()}"


class TestUKDALEIngestion:
    """Test UK-DALE dataset ingestion quality."""

    @pytest.fixture
    def ukdale_path(self):
        """UK-DALE dataset path fixture."""
        path = DATA_PROCESSED / "dataset=ukdale"
        if not path.exists():
            pytest.skip("UK-DALE not yet ingested")
        return path

    def test_ukdale_exists(self, ukdale_path):
        """Test UK-DALE processed data exists."""
        assert ukdale_path.exists()
        parquet_files = list(ukdale_path.rglob("*.parquet"))
        assert len(parquet_files) > 0, "No Parquet files found"

    def test_ukdale_schema(self, ukdale_path):
        """Test UK-DALE has correct schema."""
        schema = pq.read_schema(list(ukdale_path.rglob("*.parquet"))[0])

        required_columns = [
            "dataset",
            "entity_id",
            "ts_utc",
            "interval_mins",
            "energy_kwh",
            "source",
            "extras",
        ]

        for col in required_columns:
            assert col in schema.names, f"Missing column: {col}"

    def test_ukdale_intervals(self, ukdale_path):
        """Test UK-DALE has 30-minute intervals (CRITICAL)."""
        df = pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(10000).collect()
        intervals = df["interval_mins"].unique().sort().to_list()

        assert 30 in intervals, (
            f"UK-DALE missing 30-minute data! Found intervals: {intervals}\n"
            f"Re-run: PYTHONPATH=$(pwd)/src python -m fyp.ingestion.cli ukdale"
        )

    def test_ukdale_timezone(self, ukdale_path):
        """Test UK-DALE timestamps are UTC."""
        df = pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(1000).collect()

        # Check timezone-aware
        assert (
            "UTC" in str(df["ts_utc"].dtype) or "utc" in str(df["ts_utc"].dtype).lower()
        ), f"Timestamps not UTC, got: {df['ts_utc'].dtype}"

    def test_ukdale_energy_conversion(self, ukdale_path):
        """Test UK-DALE power-to-energy conversion is correct."""
        df = pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(100000).collect()

        # Filter for 30-minute data
        df_30min = df.filter(pl.col("interval_mins") == 30)

        if len(df_30min) == 0:
            pytest.fail("No 30-minute data found!")

        # No negative values
        negative_count = (df_30min["energy_kwh"] < 0).sum()
        assert negative_count == 0, f"Found {negative_count} negative energy values"

        # Reasonable range for 30-minute intervals
        max_energy = df_30min["energy_kwh"].max()
        assert max_energy < 10, f"Unrealistic max energy for 30min: {max_energy} kWh"

    def test_ukdale_appliance_in_extras(self, ukdale_path):
        """Test UK-DALE extras contains appliance channel data."""
        df = pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(1000).collect()

        # Parse first extras
        first_extras = json.loads(df["extras"][0])

        # Should have channel data
        assert (
            "channel" in first_extras
        ), f"Appliance channel missing from extras. Keys: {first_extras.keys()}"


class TestCrossDatasetStandardization:
    """Test standardization across both datasets."""

    @pytest.fixture
    def both_datasets(self):
        """Both datasets path fixture."""
        lcl_path = DATA_PROCESSED / "dataset=lcl"
        ukdale_path = DATA_PROCESSED / "dataset=ukdale"

        if not lcl_path.exists() or not ukdale_path.exists():
            pytest.skip("Both datasets not yet ingested")

        return lcl_path, ukdale_path

    def test_matching_intervals(self, both_datasets):
        """Test both datasets have 30-minute data (CRITICAL)."""
        lcl_path, ukdale_path = both_datasets

        lcl_df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(1000).collect()
        ukdale_df = (
            pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(1000).collect()
        )

        lcl_intervals = set(lcl_df["interval_mins"].unique().to_list())
        ukdale_intervals = set(ukdale_df["interval_mins"].unique().to_list())

        # Both must have 30
        assert 30 in lcl_intervals, "LCL missing 30-minute data"
        assert 30 in ukdale_intervals, "UK-DALE missing 30-minute data"

    def test_matching_schema(self, both_datasets):
        """Test both datasets have identical schema."""
        lcl_path, ukdale_path = both_datasets

        lcl_schema = pq.read_schema(list(lcl_path.rglob("*.parquet"))[0])
        ukdale_schema = pq.read_schema(list(ukdale_path.rglob("*.parquet"))[0])

        assert (
            lcl_schema.names == ukdale_schema.names
        ), f"Schema mismatch:\nLCL: {lcl_schema.names}\nUK-DALE: {ukdale_schema.names}"

    def test_matching_timezone(self, both_datasets):
        """Test both datasets use UTC timezone."""
        lcl_path, ukdale_path = both_datasets

        lcl_df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(100).collect()
        ukdale_df = (
            pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(100).collect()
        )

        lcl_tz = str(lcl_df["ts_utc"].dtype).lower()
        ukdale_tz = str(ukdale_df["ts_utc"].dtype).lower()

        assert "utc" in lcl_tz, f"LCL not UTC: {lcl_tz}"
        assert "utc" in ukdale_tz, f"UK-DALE not UTC: {ukdale_tz}"

    def test_can_load_together(self, both_datasets):
        """Test both datasets can be loaded together for training."""
        lcl_path, ukdale_path = both_datasets

        # Load samples from both
        lcl_df = pl.scan_parquet(str(lcl_path / "**/*.parquet")).head(1000).collect()
        ukdale_df = (
            pl.scan_parquet(str(ukdale_path / "**/*.parquet")).head(1000).collect()
        )

        # Filter both to 30-minute data
        lcl_30min = lcl_df.filter(pl.col("interval_mins") == 30)
        ukdale_30min = ukdale_df.filter(pl.col("interval_mins") == 30)

        # Concatenate
        combined = pl.concat([lcl_30min, ukdale_30min])

        assert len(combined) > 0, "Cannot combine datasets"
        assert "dataset" in combined.columns, "Missing dataset column"

        # Check both datasets present
        datasets = set(combined["dataset"].unique().to_list())
        assert datasets == {"lcl", "ukdale"}, f"Expected both datasets, got {datasets}"
