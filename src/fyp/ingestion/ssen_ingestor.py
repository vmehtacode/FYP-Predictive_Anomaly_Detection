"""Ingestor for SSEN feeder metadata.

This ingestor processes the rich SSEN Low Voltage feeder lookup dataset
which contains network hierarchy and customer count information.

Note: The CKAN API does not provide time-series consumption data.
Time-series data will come from LCL household aggregations, guided by
the customer count distribution from this SSEN metadata.
"""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from .base import BaseIngestor, ensure_timezone_aware
from .schema import EnergyReading
from .utils import RateLimitedSession


class SSENIngestor(BaseIngestor):
    """Ingest SSEN feeder metadata from rich CSV lookup.

    This ingestor processes the SSEN Low Voltage feeder lookup dataset
    containing network hierarchy and customer counts (total_mpan_count).

    The dataset includes:
    - Network hierarchy (primary/secondary substations, HV/LV feeders)
    - Customer counts per LV feeder (total_mpan_count)
    - Geographic information (postcodes)
    - Complete feeder identification

    This metadata is critical for:
    1. Defining physical network constraints for validation
    2. Guiding pseudo-feeder construction from LCL aggregations
    3. Validating forecasts against realistic feeder loads
    """

    def __init__(
        self,
        *args,
        ckan_url: str = "https://data.ssen.co.uk",
        package_id: str = "low-voltage-feeder-data",
        api_key: str | None = None,
        rate_limit: float = 1.0,  # seconds between requests
        force_refresh: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ckan_url = ckan_url
        self.package_id = package_id
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.force_refresh = force_refresh
        self.feeder_lookup = None
        self.logger = self._get_logger()

        # Initialize rate-limited session with caching
        cache_dir = str(self.output_root.parent / ".cache" / "ssen_api")
        self.session = RateLimitedSession(
            rate_limit=rate_limit,
            cache_dir=cache_dir,
            max_retries=3,
            backoff_factor=0.5,
        )

    def _get_logger(self):
        import logging

        return logging.getLogger(self.__class__.__name__)

    def _load_feeder_lookup(self) -> pd.DataFrame:
        """Load rich feeder metadata from optimized CSV.

        Loads the enhanced SSEN dataset with 11 columns including:
        - dataset_id: Unique dataset identifier
        - postcode: Geographic location
        - primary_substation_id/name: Primary substation details
        - hv_feeder_id/name: High voltage feeder details
        - secondary_substation_id/name: Secondary substation details
        - lv_feeder_id/name: Low voltage feeder details
        - total_mpan_count: Number of customer meters (CRITICAL for pseudo-feeders)

        Returns:
            DataFrame with all 11 columns from the optimized dataset
        """
        # Try new optimized file first (most recent version)
        lookup_path = (
            self.input_root
            / "ssen"
            / "ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv"
        )

        # Fall back to older version
        if not lookup_path.exists():
            lookup_path = (
                self.input_root
                / "ssen"
                / "ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_20_2025.csv"
            )

        # Fall back to legacy file if optimized doesn't exist
        if not lookup_path.exists():
            lookup_path = self.input_root / "ssen" / "LV_FEEDER_LOOKUP.csv"

        if not lookup_path.exists():
            self.logger.warning(
                "No SSEN feeder lookup file found. Tried:\n"
                "  - ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv\n"
                "  - ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_20_2025.csv\n"
                "  - LV_FEEDER_LOOKUP.csv\n"
                "Continuing without metadata enrichment."
            )
            # Initialize empty dict to avoid KeyErrors later
            self.feeder_metadata_dict = {}
            return pd.DataFrame()

        self.logger.info(f"Loading SSEN feeder metadata from: {lookup_path.name}")
        df = pd.read_csv(lookup_path)

        # Standardize column names (lowercase with underscores)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Log data quality
        self.logger.info(f"Loaded {len(df):,} feeders with {len(df.columns)} columns")
        if "total_mpan_count" in df.columns:
            non_null_counts = (~df["total_mpan_count"].isna()).sum()
            self.logger.info(
                f"Customer count data available for {non_null_counts:,}/{len(df):,} feeders"
            )

        # Create fast lookup dictionary for metadata enrichment
        # This avoids scanning the entire DataFrame for each record
        self.feeder_metadata_dict = {}
        if "lv_feeder_id" in df.columns and len(df) > 0:
            for _, row in df.iterrows():
                feeder_id = str(row["lv_feeder_id"]).strip()
                self.feeder_metadata_dict[feeder_id] = row.to_dict()
            self.logger.info(
                f"Created fast lookup index for {len(self.feeder_metadata_dict):,} feeders"
            )
        else:
            self.logger.debug("No feeder metadata to index")

        return df

    # =========================================================================
    # API Methods (DISABLED - API does not provide time-series data)
    # =========================================================================
    # These methods are kept for reference but are NOT used in the current
    # implementation. Investigation confirmed that the SSEN CKAN API only
    # provides metadata and documentation, not time-series consumption data.
    #
    # Time-series data will be synthesized through:
    # 1. LCL household consumption aggregations
    # 2. Guided by 'total_mpan_count' distribution from this metadata
    # 3. Validated against network constraints from feeder metadata
    # =========================================================================

    def _get_api_headers(self) -> dict[str, str]:
        """Get headers for API requests (NOT CURRENTLY USED)."""
        headers = {"User-Agent": "FYP-Energy-Forecasting/0.1.0"}
        if self.api_key:
            headers["Authorization"] = self.api_key
        return headers

    def _api_request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make API request with rate limiting and caching."""
        url = urljoin(self.ckan_url, endpoint)

        try:
            response = self.session.get(
                url,
                params=params,
                headers=self._get_api_headers(),
                timeout=30,
                force_refresh=self.force_refresh,
            )
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def _discover_resources(self) -> list[dict[str, Any]]:
        """Discover available data resources."""
        try:
            # Get package metadata
            result = self._api_request(
                "/api/3/action/package_show", params={"id": self.package_id}
            )

            if not result.get("success"):
                raise ValueError(f"Package not found: {self.package_id}")

            resources = result["result"].get("resources", [])

            # Filter for feeder time series data
            feeder_resources = [
                r
                for r in resources
                if "feeder" in r.get("name", "").lower()
                and r.get("format", "").upper() in ["CSV", "JSON"]
            ]

            return feeder_resources

        except Exception as e:
            self.logger.warning(f"API discovery failed: {e}")
            return []

    def read_raw_data(self) -> Iterator[dict[str, Any]]:
        """Read SSEN feeder data from both metadata and time-series CSVs.

        Strategy:
        1. Always load metadata lookup (for enrichment)
        2. If use_samples: use sample data
        3. Else if time-series file exists: read actual consumption data
        4. Else: fall back to metadata-only mode (save metadata, yield nothing)

        The time-series data is the PRIMARY output when available. Metadata
        enriches the time-series records with feeder characteristics and
        customer counts.

        Returns:
            Iterator of consumption records (if time-series available) or
            empty iterator with metadata saved to parquet (if metadata-only)
        """
        if self.use_samples:
            # Use sample data for testing/CI (no metadata enrichment needed)
            self.logger.info("Using sample data mode (no metadata enrichment)")
            self.feeder_lookup = pd.DataFrame()
            self.feeder_metadata_dict = {}
            yield from self._read_sample_data()
            return

        # Load feeder lookup for metadata enrichment (full data mode only)
        self.feeder_lookup = self._load_feeder_lookup()

        # Check for time-series consumption data
        timeseries_path = (
            self.input_root
            / "ssen"
            / "ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv"
        )

        if timeseries_path.exists():
            self.logger.info(
                "REAL SSEN time-series consumption data found! "
                "This enables validation against actual distribution network loads."
            )
            self.logger.info(
                "Processing actual consumption readings with timestamps..."
            )
            yield from self._read_timeseries_data()
        else:
            self.logger.warning(
                "Time-series consumption data not found. "
                "Saving metadata only (no consumption records)."
            )
            self.logger.info(f"Expected path: {timeseries_path}")
            self.logger.info(
                "Pseudo-feeder construction will use LCL aggregations guided by metadata."
            )
            yield from self._read_metadata_only()

    def _read_sample_data(self) -> Iterator[dict[str, Any]]:
        """Read sample CSV data in production SSEN format."""
        sample_path = Path("data/samples/ssen_sample.csv")

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        df = pd.read_csv(sample_path)

        # Sample file is in production SSEN format, so convert column names
        for _, row in df.iterrows():
            # Use production column names
            if (
                pd.notna(row.get("lv_feeder_id"))
                and pd.notna(row.get("data_collection_log_timestamp"))
                and pd.notna(row.get("total_consumption_active_import"))
            ):
                yield {
                    "feeder_id": str(row["lv_feeder_id"]).strip(),
                    "timestamp": pd.to_datetime(row["data_collection_log_timestamp"]),
                    "wh_30m": float(row["total_consumption_active_import"]),
                    "device_count": float(
                        row.get("aggregated_device_count_active", None)
                    )
                    if pd.notna(row.get("aggregated_device_count_active"))
                    else None,
                    "reactive_wh": float(
                        row.get("total_consumption_reactive_import", None)
                    )
                    if pd.notna(row.get("total_consumption_reactive_import"))
                    else None,
                    "primary_consumption": float(
                        row.get("primary_consumption_active_import", None)
                    )
                    if pd.notna(row.get("primary_consumption_active_import"))
                    else None,
                    "secondary_consumption": float(
                        row.get("secondary_consumption_active_import", None)
                    )
                    if pd.notna(row.get("secondary_consumption_active_import"))
                    else None,
                    "dno_name": str(row.get("dno_name", "SSEN")),
                    "secondary_substation_id": str(
                        row.get("secondary_substation_id", None)
                    )
                    if pd.notna(row.get("secondary_substation_id"))
                    else None,
                    "source": sample_path.name,
                }

    def _read_api_data(self) -> Iterator[dict[str, Any]]:
        """Read data from CKAN API."""
        resources = self._discover_resources()

        if not resources:
            self.logger.warning("No resources found, using mock data")
            # Generate mock data for testing
            yield from self._generate_mock_data()
            return

        for resource in resources:
            resource_id = resource["id"]
            self.logger.info(f"Processing resource: {resource['name']}")

            # Paginate through data
            offset = 0
            limit = 1000

            while True:
                try:
                    result = self._api_request(
                        "/api/3/action/datastore_search",
                        params={
                            "resource_id": resource_id,
                            "limit": limit,
                            "offset": offset,
                        },
                    )

                    if not result.get("success"):
                        break

                    records = result["result"].get("records", [])
                    if not records:
                        break

                    for record in records:
                        # Parse record based on expected format
                        yield self._parse_api_record(record, resource_id)

                    offset += limit

                    # Check if more data available
                    total = result["result"].get("total", 0)
                    if offset >= total:
                        break

                except Exception as e:
                    self.logger.error(f"Error reading resource {resource_id}: {e}")
                    break

    def _parse_api_record(self, record: dict, resource_id: str) -> dict[str, Any]:
        """Parse API record to standard format."""
        # Adapt based on actual API response format
        # This is a placeholder implementation
        return {
            "feeder_id": record.get("feeder_id", "unknown"),
            "timestamp": pd.to_datetime(record.get("timestamp")),
            "wh_30m": float(record.get("energy_wh", 0))
            if "energy_wh" in record
            else float(record.get("power_kw", 0)) * 500,  # Convert kW to Wh for 30min
            "source": f"api:{resource_id}",
            "retrieved_at": datetime.utcnow(),
        }

    def _read_timeseries_data(self) -> Iterator[dict[str, Any]]:
        """Read actual feeder consumption time-series from usage CSV.

        This method processes REAL half-hourly consumption data from SSEN feeders.
        Each row represents a consumption reading at a specific timestamp from an
        operational distribution network.

        Expected CSV columns:
            - lv_feeder_id: LV feeder identifier
            - data_collection_log_timestamp: When the reading was taken
            - total_consumption_active_import: Total active energy (Wh)
            - aggregated_device_count_active: Number of smart meters reporting
            - total_consumption_reactive_import: Reactive energy (Wh)
            - [Other columns for metadata enrichment]

        Returns:
            Iterator of dicts with keys:
                - feeder_id (str): LV feeder identifier
                - timestamp (datetime): When the reading was taken
                - wh_30m (float): Total active energy consumption (Wh for 30-min period)
                - device_count (float): Number of smart meters reporting
                - reactive_wh (float): Reactive energy consumption (Wh)
                - source (str): Filename for provenance
        """
        timeseries_path = (
            self.input_root
            / "ssen"
            / "ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv"
        )

        if not timeseries_path.exists():
            self.logger.warning(
                f"Time-series usage file not found: {timeseries_path.name}. "
                "Falling back to metadata-only mode."
            )
            return

        self.logger.info(
            f"Loading SSEN time-series consumption data from {timeseries_path.name}"
        )
        self.logger.info(
            "This enables REAL feeder-level validation against operational grid data!"
        )

        # Read CSV in chunks for memory efficiency
        chunk_size = 10000
        total_records = 0
        valid_records = 0
        skipped_missing_feeder = 0
        skipped_missing_timestamp = 0
        skipped_missing_consumption = 0

        try:
            for chunk in pd.read_csv(timeseries_path, chunksize=chunk_size):
                total_records += len(chunk)

                # Filter out rows with missing required fields (vectorized)
                has_feeder = chunk["lv_feeder_id"].notna()
                has_timestamp = chunk["data_collection_log_timestamp"].notna()
                has_consumption = chunk["total_consumption_active_import"].notna()

                # Count skipped records
                skipped_missing_feeder += (~has_feeder).sum()
                skipped_missing_timestamp += (~has_timestamp).sum()
                skipped_missing_consumption += (~has_consumption).sum()

                # Keep only valid records
                valid_mask = has_feeder & has_timestamp & has_consumption
                valid_chunk = chunk[valid_mask].copy()
                valid_records += len(valid_chunk)

                # Convert timestamp column once for the entire chunk
                valid_chunk["timestamp"] = pd.to_datetime(
                    valid_chunk["data_collection_log_timestamp"], errors="coerce"
                )

                # Yield records from valid chunk
                for _, row in valid_chunk.iterrows():
                    yield {
                        "feeder_id": str(row["lv_feeder_id"]).strip(),
                        "timestamp": row["timestamp"],
                        "wh_30m": float(row["total_consumption_active_import"]),
                        "device_count": (
                            float(row["aggregated_device_count_active"])
                            if pd.notna(row.get("aggregated_device_count_active"))
                            else None
                        ),
                        "reactive_wh": (
                            float(row["total_consumption_reactive_import"])
                            if pd.notna(row.get("total_consumption_reactive_import"))
                            else None
                        ),
                        "primary_consumption": (
                            float(row["primary_consumption_active_import"])
                            if pd.notna(row.get("primary_consumption_active_import"))
                            else None
                        ),
                        "secondary_consumption": (
                            float(row["secondary_consumption_active_import"])
                            if pd.notna(row.get("secondary_consumption_active_import"))
                            else None
                        ),
                        "dno_name": (
                            str(row["dno_name"])
                            if pd.notna(row.get("dno_name"))
                            else "SSEN"
                        ),
                        "secondary_substation_id": (
                            str(row["secondary_substation_id"])
                            if pd.notna(row.get("secondary_substation_id"))
                            else None
                        ),
                        "source": timeseries_path.name,
                    }

            # Log processing statistics
            self.logger.info(
                f"Processed {valid_records:,} valid consumption records "
                f"out of {total_records:,} total rows"
            )
            if (
                skipped_missing_feeder
                + skipped_missing_timestamp
                + skipped_missing_consumption
                > 0
            ):
                self.logger.info("Skipped records:")
                self.logger.info(f"  - Missing feeder ID: {skipped_missing_feeder:,}")
                self.logger.info(
                    f"  - Missing timestamp: {skipped_missing_timestamp:,}"
                )
                self.logger.info(
                    f"  - Missing consumption: {skipped_missing_consumption:,}"
                )

            self.logger.info("SSEN time-series data ready for real-world validation!")

        except Exception as e:
            self.logger.error(f"Error reading time-series data: {e}")
            raise

    def _read_metadata_only(self) -> Iterator[dict[str, Any]]:
        """Read feeder metadata from CSV and save with statistics.

        This processes the enhanced SSEN dataset containing:
        - Network hierarchy (primary/secondary substations, HV/LV feeders)
        - Customer counts (total_mpan_count) - critical for pseudo-feeder construction
        - Geographic distribution (postcodes)

        Generates quality flags and statistics for downstream analysis.
        Does NOT produce time-series records (no consumption data available).

        Returns:
            Empty iterator (metadata saved directly to parquet)
        """
        if self.feeder_lookup is None or self.feeder_lookup.empty:
            self.logger.warning("No feeder lookup data available")
            return

        self.logger.info(
            f"Processing {len(self.feeder_lookup)} feeders from enhanced dataset"
        )

        # We don't have time-series data, so we save metadata separately
        metadata_path = self.output_root / "ssen_metadata.parquet"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Start with original data
        metadata_df = self.feeder_lookup.copy()

        # Add UK electrical constraints
        metadata_df["voltage_nominal_v"] = 230.0  # UK standard
        metadata_df["voltage_tolerance_pct"] = 10.0  # UK statutory ±10%
        metadata_df["power_factor_min"] = 0.8  # Typical minimum
        metadata_df["power_factor_max"] = 1.0

        # Add data quality flags
        if "total_mpan_count" in metadata_df.columns:
            metadata_df["has_customer_count"] = ~metadata_df["total_mpan_count"].isna()

            # Categorize feeder size based on customer count
            def categorize_feeder_size(count):
                if pd.isna(count):
                    return "unknown"
                elif count < 30:
                    return "small"
                elif count <= 100:
                    return "medium"
                else:
                    return "large"

            metadata_df["feeder_size_category"] = metadata_df["total_mpan_count"].apply(
                categorize_feeder_size
            )
        else:
            metadata_df["has_customer_count"] = False
            metadata_df["feeder_size_category"] = "unknown"

        # Calculate and log statistics
        total_feeders = len(metadata_df)

        if "total_mpan_count" in metadata_df.columns:
            customer_counts = metadata_df["total_mpan_count"].dropna()
            feeders_with_counts = len(customer_counts)

            if feeders_with_counts > 0:
                self.logger.info(
                    f"Feeders with customer data: {feeders_with_counts:,}/{total_feeders:,}"
                )
                self.logger.info(
                    f"Customer count range: {customer_counts.min():.0f} - {customer_counts.max():.0f}"
                )
                self.logger.info(
                    f"Median customers per feeder: {customer_counts.median():.0f}"
                )
                self.logger.info(
                    f"Customer count quartiles - Q1: {customer_counts.quantile(0.25):.0f}, "
                    f"Q3: {customer_counts.quantile(0.75):.0f}"
                )
                self.logger.info(
                    f"Mean customers per feeder: {customer_counts.mean():.1f}"
                )

                # Log size distribution
                size_dist = metadata_df["feeder_size_category"].value_counts()
                self.logger.info("Feeder size distribution:")
                for size, count in size_dist.items():
                    pct = (count / total_feeders) * 100
                    self.logger.info(f"  {size}: {count:,} ({pct:.1f}%)")

        # Log geographic coverage
        if "postcode" in metadata_df.columns:
            unique_postcodes = metadata_df["postcode"].nunique()
            self.logger.info(
                f"Geographic coverage: {unique_postcodes:,} unique postcodes"
            )

        # Log network hierarchy
        if "primary_substation_id" in metadata_df.columns:
            unique_primary = metadata_df["primary_substation_id"].nunique()
            self.logger.info(
                f"Network hierarchy: {unique_primary:,} primary substations"
            )

        if "secondary_substation_id" in metadata_df.columns:
            unique_secondary = metadata_df["secondary_substation_id"].nunique()
            self.logger.info(
                f"                   {unique_secondary:,} secondary substations"
            )

        if "hv_feeder_id" in metadata_df.columns:
            unique_hv = metadata_df["hv_feeder_id"].nunique()
            self.logger.info(f"                   {unique_hv:,} HV feeders")

        if "lv_feeder_id" in metadata_df.columns:
            unique_lv = metadata_df["lv_feeder_id"].nunique()
            self.logger.info(f"                   {unique_lv:,} LV feeders")

        # Save to parquet
        metadata_df.to_parquet(metadata_path, index=False)
        self.logger.info(f"Saved feeder metadata to {metadata_path}")
        self.logger.info(
            "Metadata includes: network hierarchy, customer counts, voltage constraints"
        )
        self.logger.info(
            f"Total columns: {len(metadata_df.columns)} (11 original + 6 enhanced)"
        )
        self.logger.info(
            "Ready for pseudo-feeder construction: use 'total_mpan_count' to guide LCL aggregations"
        )
        self.logger.info(
            "Note: Time-series consumption data not available - metadata only"
        )

        # Update stats for summary
        self.stats["feeders_processed"] = total_feeders
        if "total_mpan_count" in metadata_df.columns:
            self.stats["feeders_with_customer_data"] = feeders_with_counts
            if feeders_with_counts > 0:
                self.stats["customer_count_min"] = float(customer_counts.min())
                self.stats["customer_count_max"] = float(customer_counts.max())
                self.stats["customer_count_median"] = float(customer_counts.median())
                self.stats["customer_count_mean"] = float(customer_counts.mean())

        # Don't yield any records since we don't have time-series data
        # This will result in 0 records processed, which is correct
        return
        yield  # Make this a generator

    def transform_record(self, record: dict[str, Any]) -> EnergyReading | None:
        """Transform SSEN record to unified schema with metadata enrichment.

        For time-series records, enriches with feeder metadata from lookup table
        including customer counts, network hierarchy, and geographic information.

        Args:
            record: Dict with keys from _read_timeseries_data() or _read_sample_data()

        Returns:
            EnergyReading with enriched extras, or None if transformation fails
        """
        try:
            # Convert timestamp to UTC
            ts_utc = ensure_timezone_aware(record["timestamp"])

            # Convert Wh to kWh
            energy_kwh = record["wh_30m"] / 1000.0

            # Build extras dict with time-series specific fields
            extras = {
                "source_file": record["source"],
                "ingestion_version": "v2.1_timeseries",
            }

            # Add time-series specific fields if available
            if "device_count" in record and record["device_count"] is not None:
                extras["device_count"] = int(record["device_count"])

            if "reactive_wh" in record and record["reactive_wh"] is not None:
                extras["reactive_kwh"] = float(record["reactive_wh"]) / 1000.0

            if (
                "primary_consumption" in record
                and record["primary_consumption"] is not None
            ):
                extras["primary_consumption_kwh"] = (
                    float(record["primary_consumption"]) / 1000.0
                )

            if (
                "secondary_consumption" in record
                and record["secondary_consumption"] is not None
            ):
                extras["secondary_consumption_kwh"] = (
                    float(record["secondary_consumption"]) / 1000.0
                )

            if "dno_name" in record:
                extras["dno_name"] = record["dno_name"]

            if (
                "secondary_substation_id" in record
                and record["secondary_substation_id"]
            ):
                extras["secondary_substation_id"] = record["secondary_substation_id"]

            # Enrich with feeder metadata using fast dictionary lookup
            if hasattr(self, "feeder_metadata_dict"):
                feeder_id_str = str(record["feeder_id"]).strip()

                # Fast dictionary lookup instead of DataFrame scan
                if feeder_id_str in self.feeder_metadata_dict:
                    metadata_row = self.feeder_metadata_dict[feeder_id_str]

                    # Add critical metadata fields
                    metadata_fields = [
                        "lv_feeder_name",
                        "total_mpan_count",
                        "postcode",
                        "primary_substation_id",
                        "primary_substation_name",
                        "secondary_substation_name",
                        "hv_feeder_id",
                        "hv_feeder_name",
                    ]

                    for field in metadata_fields:
                        if field in metadata_row and pd.notna(metadata_row[field]):
                            # Store as appropriate type
                            value = metadata_row[field]
                            if isinstance(value, int | float):
                                extras[field] = (
                                    float(value) if pd.notna(value) else None
                                )
                            else:
                                extras[field] = str(value)

            # Create entity ID with dataset prefix
            entity_id = f"ssen_feeder_{record['feeder_id']}"

            return EnergyReading(
                dataset="ssen",
                entity_id=entity_id,
                ts_utc=ts_utc,
                interval_mins=30,
                energy_kwh=energy_kwh,
                source=f"timeseries:{record['source']}",
                extras=extras,
            )

        except Exception as e:
            self.logger.debug(
                f"Transform error for feeder {record.get('feeder_id', 'unknown')}: {e}"
            )
            return None


def main():
    """CLI entry point."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Ingest SSEN feeder data")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Input root directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root directory",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample data instead of API",
    )
    parser.add_argument(
        "--ckan-url",
        default=os.getenv("SSEN_CKAN_URL", "https://data.ssen.co.uk"),
        help="CKAN API base URL",
    )
    parser.add_argument(
        "--package-id",
        default="low-voltage-feeder-data",
        help="CKAN package ID",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SSEN_API_KEY"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of cached API responses",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )

    args = parser.parse_args()

    ingestor = SSENIngestor(
        input_root=args.input_root,
        output_root=args.output_root,
        use_samples=args.use_samples,
        ckan_url=args.ckan_url,
        package_id=args.package_id,
        api_key=args.api_key,
        force_refresh=args.force_refresh,
        dry_run=args.dry_run,
    )
    ingestor.run()


if __name__ == "__main__":
    main()
