"""Proposer agent for generating challenging energy consumption scenarios.

This module implements the scenario generation component of the self-play system,
creating physically-plausible but challenging scenarios for training robust
forecasting models.
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data

from fyp.selfplay.utils import (
    apply_scenario_transformation,
    estimate_scenario_difficulty,
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioProposal:
    """Data class for proposed consumption scenarios."""

    scenario_type: str  # One of: EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA
    magnitude: float  # Intensity multiplier (e.g., 1.5x baseline)
    duration: int  # Number of 30-min intervals
    start_time: datetime  # When scenario begins
    affected_appliances: list[str]  # Which appliances/loads affected
    baseline_context: np.ndarray  # Historical context window (7 days = 336 intervals)
    difficulty_score: float  # Estimated learnability (0=trivial, 1=unsolvable)
    physics_valid: bool  # Passes constraint pre-check
    metadata: dict = field(default_factory=dict)  # Additional scenario-specific data

    def apply_to_timeseries(self, baseline: np.ndarray) -> np.ndarray:
        """Transform baseline consumption with scenario parameters.

        Args:
            baseline: Original consumption time series

        Returns:
            Transformed time series with scenario applied
        """
        # Calculate start index relative to baseline
        start_idx = 0  # Assuming scenario starts at beginning of forecast
        if "start_offset" in self.metadata:
            start_idx = self.metadata["start_offset"]

        return apply_scenario_transformation(
            baseline=baseline,
            scenario_type=self.scenario_type,
            magnitude=self.magnitude,
            duration=self.duration,
            start_idx=start_idx,
        )

    def apply_to_graph_timeseries(self, baseline: np.ndarray) -> np.ndarray:
        """Transform multi-node baseline consumption with per-node cascade magnitudes.

        For graph-aware scenarios, each node's time series is scaled by its
        cascade magnitude from metadata["affected_nodes"]. Seed nodes get
        the full transformation, while neighbors get a blended version
        proportional to their cascade decay magnitude.

        For non-graph scenarios (no "graph_aware" in metadata), applies the
        uniform transformation to every node row (same as apply_to_timeseries
        broadcast across nodes).

        Args:
            baseline: 2-D array of shape [num_nodes, timesteps]

        Returns:
            Transformed 2-D array with per-node scenario applied
        """
        if baseline.ndim != 2:
            raise ValueError(
                f"Expected 2-D array [num_nodes, timesteps], got shape {baseline.shape}"
            )

        num_nodes, timesteps = baseline.shape
        result = np.empty_like(baseline)

        if not self.metadata.get("graph_aware", False):
            # No graph info: apply uniform transformation to every node
            for i in range(num_nodes):
                result[i] = self.apply_to_timeseries(baseline[i])
            return result

        affected_nodes = self.metadata.get("affected_nodes", {})

        for i in range(num_nodes):
            full_transform = self.apply_to_timeseries(baseline[i])
            magnitude = affected_nodes.get(i, 0.0)

            if magnitude >= 1.0:
                # Seed node: full transformation
                result[i] = full_transform
            elif magnitude > 0.0:
                # Cascade neighbor: blended transformation
                # baseline + magnitude * (transformed - baseline)
                result[i] = baseline[i] + magnitude * (full_transform - baseline[i])
            else:
                # Unaffected node: no transformation
                result[i] = baseline[i].copy()

        return result

    def get_verification_constraints(self) -> list[str]:
        """Return physics constraints this scenario must satisfy.

        Returns:
            List of constraint names to check
        """
        # All scenarios must satisfy basic physics
        constraints = ["non_negativity", "household_max"]

        # Scenario-specific constraints
        if self.scenario_type == "EV_SPIKE":
            constraints.extend(["ramp_rate", "voltage"])
        elif self.scenario_type == "COLD_SNAP":
            constraints.extend(["temporal_pattern", "power_factor"])
        elif self.scenario_type == "PEAK_SHIFT":
            constraints.append("temporal_pattern")
        elif self.scenario_type == "OUTAGE":
            constraints.append("ramp_rate")  # Check recovery ramp

        return constraints

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scenario_type": self.scenario_type,
            "magnitude": self.magnitude,
            "duration": self.duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "affected_appliances": self.affected_appliances,
            "difficulty_score": self.difficulty_score,
            "physics_valid": self.physics_valid,
            "metadata": self.metadata,
        }


class ProposerAgent:
    """Generates diverse energy consumption scenarios for self-play training."""

    # Scenario configuration
    SCENARIO_CONFIGS = {
        "EV_SPIKE": {
            "magnitude_range": (1.0, 2.0),  # Multiplier for EV power (3.5-7 kW base)
            "duration_range": (2, 16),  # 1-8 hours in 30-min intervals
            "appliances": ["electric_vehicle", "battery_charger"],
            "base_difficulty": 0.3,
        },
        "COLD_SNAP": {
            "magnitude_range": (1.5, 3.0),  # 1.5-3x baseline consumption
            "duration_range": (12, 144),  # 6-72 hours
            "appliances": ["heating", "space_heater", "heat_pump"],
            "base_difficulty": 0.4,
        },
        "PEAK_SHIFT": {
            "magnitude_range": (-2.0, 2.0),  # ±2 hours shift (negative = earlier)
            "duration_range": (4, 12),  # 2-6 hours of shifted peak
            "appliances": ["cooking", "washing_machine", "dishwasher"],
            "base_difficulty": 0.6,
        },
        "OUTAGE": {
            "magnitude_range": (0.0, 0.0),  # Zero consumption
            "duration_range": (2, 24),  # 1-12 hours
            "appliances": ["all"],
            "base_difficulty": 0.2,
        },
        "MISSING_DATA": {
            "magnitude_range": (0.0, 0.0),  # Not applicable
            "duration_range": (1, 12),  # 0.5-6 hours of gaps
            "appliances": ["meter"],
            "base_difficulty": 0.5,
        },
    }

    def __init__(
        self,
        ssen_constraints_path: str,
        difficulty_curriculum: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize proposer agent.

        Args:
            ssen_constraints_path: Path to physics constraints JSON
            difficulty_curriculum: Enable progressive difficulty scaling
            random_seed: Random seed for reproducibility
        """
        self.constraints = self._load_ssen_constraints(ssen_constraints_path)
        self.scenario_buffer = []  # Store past (scenario, reward) pairs
        self.curriculum_level = 0.0 if difficulty_curriculum else 1.0
        self.difficulty_curriculum = difficulty_curriculum
        self.episode_count = 0

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        logger.info(
            f"Initialized ProposerAgent with curriculum={difficulty_curriculum}, "
            f"level={self.curriculum_level}"
        )

    def _load_ssen_constraints(self, path: str) -> dict:
        """Load SSEN physics constraints from JSON."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Constraints file not found: {path}")
            return {
                "household_limits": {
                    "typical_max_kwh_30min": 7.5,
                    "absolute_max_kwh_30min": 50.0,
                },
                "ev_charging": {
                    "typical_power_kw": 3.5,
                    "fast_charge_power_kw": 7.0,
                    "max_duration_hours": 8,
                },
            }

    def propose_scenario(
        self,
        historical_context: np.ndarray,
        conditioning_samples: list[tuple[ScenarioProposal, float]] | None = None,
        forecast_horizon: int = 48,
        current_timestamp: datetime | None = None,
        graph_data: Data | None = None,
    ) -> ScenarioProposal:
        """Generate a new scenario conditioned on past successful scenarios.

        Inspired by AZR's learnability reward:
        - Avoid trivial scenarios (solver success rate = 100%)
        - Avoid unsolvable scenarios (solver success rate = 0%)
        - Target moderate difficulty (solver success rate ~40-60%)

        Args:
            historical_context: Shape (336,) - 7 days of 30-min data
            conditioning_samples: Recent (scenario, reward) pairs from buffer
            forecast_horizon: Number of intervals to forecast (default 48 = 24h)
            current_timestamp: Current time for scenario generation

        Returns:
            ScenarioProposal with estimated difficulty and physics validation
        """
        # Select scenario type based on curriculum and conditioning
        scenario_type = self._select_scenario_type(conditioning_samples)

        # Sample parameters within physics bounds
        params = self._sample_scenario_parameters(scenario_type, forecast_horizon)

        # Create base proposal
        proposal = ScenarioProposal(
            scenario_type=scenario_type,
            magnitude=params["magnitude"],
            duration=params["duration"],
            start_time=current_timestamp or datetime.now(),
            affected_appliances=self.SCENARIO_CONFIGS[scenario_type]["appliances"],
            baseline_context=historical_context.copy(),
            difficulty_score=0.0,  # Will be estimated
            physics_valid=False,  # Will be validated
            metadata=params.get("metadata", {}),
        )

        # Validate against physics constraints
        proposal.physics_valid = self._validate_physics_constraints(proposal)

        # Estimate difficulty using historical data and conditioning
        proposal.difficulty_score = self._estimate_difficulty(
            proposal, conditioning_samples
        )

        # Adjust difficulty based on curriculum
        if self.difficulty_curriculum:
            proposal = self._apply_curriculum_adjustment(proposal)

        # Enrich with graph topology if available
        if graph_data is not None:
            self._enrich_with_graph_topology(proposal, graph_data)

        logger.debug(
            f"Proposed {scenario_type} scenario: magnitude={params['magnitude']:.2f}, "
            f"duration={params['duration']}, difficulty={proposal.difficulty_score:.2f}"
        )

        return proposal

    def _select_scenario_type(
        self, conditioning_samples: list[tuple[ScenarioProposal, float]] | None
    ) -> str:
        """Select scenario type based on curriculum and past performance."""
        # Get scenario type distribution based on curriculum level
        if self.curriculum_level < 0.3:
            # Early curriculum: mostly easy scenarios
            weights = {
                "OUTAGE": 0.35,
                "EV_SPIKE": 0.35,
                "COLD_SNAP": 0.20,
                "PEAK_SHIFT": 0.05,
                "MISSING_DATA": 0.05,
            }
        elif self.curriculum_level < 0.7:
            # Mid curriculum: balanced mix
            weights = {
                "OUTAGE": 0.15,
                "EV_SPIKE": 0.25,
                "COLD_SNAP": 0.30,
                "PEAK_SHIFT": 0.20,
                "MISSING_DATA": 0.10,
            }
        else:
            # Late curriculum: mostly hard scenarios
            weights = {
                "OUTAGE": 0.05,
                "EV_SPIKE": 0.15,
                "COLD_SNAP": 0.25,
                "PEAK_SHIFT": 0.35,
                "MISSING_DATA": 0.20,
            }

        # Adjust weights based on recent performance
        if conditioning_samples:
            recent_rewards = {st: [] for st in self.SCENARIO_CONFIGS.keys()}

            for scenario, reward in conditioning_samples[-10:]:
                recent_rewards[scenario.scenario_type].append(reward)

            # Boost underperforming scenario types
            for st, rewards in recent_rewards.items():
                if rewards:
                    avg_reward = np.mean(rewards)
                    if avg_reward < 0.3:  # Poor learnability
                        weights[st] *= 0.7  # Reduce frequency
                    elif avg_reward > 0.7:  # Too easy
                        weights[st] *= 1.3  # Increase frequency

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Sample scenario type
        return np.random.choice(list(weights.keys()), p=list(weights.values()))

    def _sample_scenario_parameters(
        self, scenario_type: str, forecast_horizon: int
    ) -> dict:
        """Sample scenario parameters within valid ranges."""
        config = self.SCENARIO_CONFIGS[scenario_type]

        # Sample magnitude
        mag_range = config["magnitude_range"]
        magnitude = np.random.uniform(mag_range[0], mag_range[1])

        # Sample duration (constrained by forecast horizon)
        dur_range = config["duration_range"]
        max_duration = min(dur_range[1], forecast_horizon)
        duration = np.random.randint(dur_range[0], max_duration + 1)

        # Sample start offset within forecast window
        max_start = max(0, forecast_horizon - duration)
        start_offset = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        params = {
            "magnitude": magnitude,
            "duration": duration,
            "metadata": {"start_offset": start_offset},
        }

        # Scenario-specific parameters
        if scenario_type == "EV_SPIKE":
            # Add EV-specific metadata
            params["metadata"]["charge_rate_kw"] = magnitude * self.constraints.get(
                "ev_charging", {}
            ).get("typical_power_kw", 3.5)

        elif scenario_type == "PEAK_SHIFT":
            # Convert magnitude to shift hours
            params["metadata"]["shift_hours"] = int(magnitude)

        return params

    def _validate_physics_constraints(self, proposal: ScenarioProposal) -> bool:
        """Pre-validate scenario against hard physics constraints."""
        # Create synthetic forecast to test
        test_forecast = np.ones(48) * np.mean(proposal.baseline_context)
        test_forecast = proposal.apply_to_timeseries(test_forecast)

        # Check basic physics constraints
        if np.any(test_forecast < 0):
            return False

        if np.any(
            test_forecast
            > self.constraints["household_limits"]["absolute_max_kwh_30min"]
        ):
            return False

        # Check scenario-specific constraints
        if proposal.scenario_type == "EV_SPIKE":
            # Validate EV charging limits
            ev_power = proposal.metadata.get("charge_rate_kw", 0)
            max_ev_power = self.constraints.get("ev_charging", {}).get(
                "fast_charge_power_kw", 7.0
            )
            if ev_power > max_ev_power * 1.5:  # Allow some flexibility
                return False

        return True

    def _estimate_difficulty(
        self,
        proposal: ScenarioProposal,
        conditioning_samples: list[tuple[ScenarioProposal, float]] | None,
    ) -> float:
        """Estimate scenario difficulty based on type and parameters."""
        # Calculate baseline volatility from context
        if len(proposal.baseline_context) > 1:
            volatility = np.std(proposal.baseline_context) / (
                np.mean(proposal.baseline_context) + 1e-6
            )
        else:
            volatility = 0.1

        # Base difficulty estimation
        difficulty = estimate_scenario_difficulty(
            scenario_type=proposal.scenario_type,
            magnitude=proposal.magnitude,
            duration=proposal.duration,
            historical_volatility=volatility,
        )

        # Adjust based on conditioning samples
        if conditioning_samples:
            # Find similar scenarios
            similar_difficulties = []
            for past_scenario, reward in conditioning_samples[-20:]:
                if past_scenario.scenario_type == proposal.scenario_type:
                    # Weight by parameter similarity
                    mag_diff = abs(past_scenario.magnitude - proposal.magnitude)
                    dur_diff = abs(past_scenario.duration - proposal.duration) / 48
                    similarity = np.exp(-(mag_diff + dur_diff))

                    # Learnability reward indicates difficulty
                    # High reward = good difficulty, low reward = too easy/hard
                    adjusted_difficulty = past_scenario.difficulty_score
                    if reward < 0.2:  # Too hard
                        adjusted_difficulty *= 1.2
                    elif reward > 0.8:  # Too easy
                        adjusted_difficulty *= 0.8

                    similar_difficulties.append((adjusted_difficulty, similarity))

            if similar_difficulties:
                # Weighted average of similar scenarios
                weights = [s[1] for s in similar_difficulties]
                difficulties = [s[0] for s in similar_difficulties]
                weighted_avg = np.average(difficulties, weights=weights)
                difficulty = 0.7 * difficulty + 0.3 * weighted_avg

        return np.clip(difficulty, 0.0, 1.0)

    def _apply_curriculum_adjustment(
        self, proposal: ScenarioProposal
    ) -> ScenarioProposal:
        """Adjust scenario difficulty based on curriculum level."""
        # Scale difficulty by curriculum level
        target_difficulty = self.curriculum_level
        current_difficulty = proposal.difficulty_score

        if current_difficulty > target_difficulty + 0.2:
            # Scenario too hard for current level - simplify
            if proposal.scenario_type in ["COLD_SNAP", "PEAK_SHIFT"]:
                proposal.magnitude *= 0.8  # Reduce intensity
                proposal.duration = max(2, int(proposal.duration * 0.8))
            proposal.difficulty_score *= 0.9

        elif current_difficulty < target_difficulty - 0.2:
            # Scenario too easy - make harder
            if proposal.scenario_type in ["EV_SPIKE", "COLD_SNAP"]:
                proposal.magnitude *= 1.2  # Increase intensity
                proposal.duration = min(proposal.duration + 4, 48)
            proposal.difficulty_score *= 1.1

        proposal.difficulty_score = np.clip(proposal.difficulty_score, 0.0, 1.0)
        return proposal

    def update_buffer(self, scenario: ScenarioProposal, reward: float):
        """Store scenario and reward for future conditioning.

        Args:
            scenario: Completed scenario
            reward: Learnability reward received
        """
        self.scenario_buffer.append((scenario, reward))
        if len(self.scenario_buffer) > 1000:  # Keep recent 1000
            self.scenario_buffer.pop(0)

        # Update curriculum level based on performance
        if self.difficulty_curriculum and len(self.scenario_buffer) >= 10:
            recent_rewards = [r for _, r in self.scenario_buffer[-10:]]
            avg_reward = np.mean(recent_rewards)

            # Increase difficulty if doing well
            if avg_reward > 0.6 and self.curriculum_level < 1.0:
                self.curriculum_level = min(1.0, self.curriculum_level + 0.01)
            # Decrease if struggling
            elif avg_reward < 0.3 and self.curriculum_level > 0.0:
                self.curriculum_level = max(0.0, self.curriculum_level - 0.01)

        self.episode_count += 1

    def compute_learnability_reward(
        self, scenario: ScenarioProposal, solver_success_rate: float
    ) -> float:
        """Compute proposer reward based on scenario learnability.

        Following AZR Equation (4):
        r_propose = {
            0, if r_solve_avg = 0 (unsolvable)
            1 - r_solve_avg, otherwise
        }

        Intuition: Reward scenarios that are challenging but solvable.

        Args:
            scenario: Proposed scenario
            solver_success_rate: Solver's success rate on this scenario type

        Returns:
            Learnability reward in [0, 1]
        """
        if solver_success_rate == 0.0:
            return 0.0

        # Base learnability reward
        learnability = 1.0 - solver_success_rate

        # Bonus for well-calibrated difficulty
        if 0.4 <= solver_success_rate <= 0.6:
            learnability *= 1.2  # Ideal difficulty range

        # Penalty for physics violations
        if not scenario.physics_valid:
            learnability *= 0.5

        return np.clip(learnability, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Graph-aware methods (Phase 3)
    # ------------------------------------------------------------------

    def _enrich_with_graph_topology(
        self, proposal: ScenarioProposal, graph_data: Data
    ) -> None:
        """Enrich proposal with graph topology information.

        Selects seed nodes based on scenario type and node hierarchy,
        propagates cascade through graph neighbors, and stores per-node
        magnitude information in proposal metadata.

        Args:
            proposal: ScenarioProposal to enrich in-place.
            graph_data: PyG Data with edge_index, node_type, num_nodes.
        """
        seed_nodes = self._select_seed_nodes(graph_data, proposal.scenario_type)
        affected_nodes = self._propagate_through_neighbors(
            seed_nodes, graph_data, num_hops=2, decay_factor=0.7
        )

        # Cap affected nodes at 30% of graph
        max_affected = math.ceil(0.3 * graph_data.num_nodes)
        if len(affected_nodes) > max_affected:
            seed_set = set(seed_nodes.tolist())
            non_seeds = [k for k in affected_nodes if k not in seed_set]
            # Keep all seeds, randomly subsample non-seeds
            num_non_seeds_to_keep = max_affected - len(seed_set)
            if num_non_seeds_to_keep > 0:
                kept_non_seeds = random.sample(
                    non_seeds, min(num_non_seeds_to_keep, len(non_seeds))
                )
            else:
                kept_non_seeds = []
            # Rebuild affected_nodes with only kept nodes
            capped = {}
            for k in seed_set:
                if k in affected_nodes:
                    capped[k] = affected_nodes[k]
            for k in kept_non_seeds:
                capped[k] = affected_nodes[k]
            affected_nodes = capped

        # Compute actual max cascade depth reached
        seed_set = set(seed_nodes.tolist())
        cascade_depth = 0
        for node, mag in affected_nodes.items():
            if node not in seed_set:
                if mag >= 0.7 - 1e-9:
                    cascade_depth = max(cascade_depth, 1)
                else:
                    cascade_depth = max(cascade_depth, 2)

        proposal.metadata["graph_aware"] = True
        proposal.metadata["seed_nodes"] = seed_nodes.tolist()
        proposal.metadata["affected_nodes"] = affected_nodes
        proposal.metadata["num_hops"] = 2
        proposal.metadata["decay_factor"] = 0.7
        proposal.metadata["cascade_depth"] = cascade_depth

    def _select_seed_nodes(
        self, graph_data: Data, scenario_type: str, num_seeds: int = 3
    ) -> torch.Tensor:
        """Select seed nodes for cascade based on scenario type and node hierarchy.

        For COLD_SNAP, OUTAGE, and EV_SPIKE: prefer LV feeder nodes (type=2).
        For PEAK_SHIFT, MISSING_DATA: use all nodes as candidates.

        Args:
            graph_data: PyG Data with node_type tensor.
            scenario_type: Scenario type string.
            num_seeds: Maximum number of seed nodes to select.

        Returns:
            Tensor of selected node indices.
        """
        node_type = graph_data.node_type

        # Filter candidates based on scenario type
        lv_preferred = {"COLD_SNAP", "OUTAGE", "EV_SPIKE"}
        if scenario_type in lv_preferred:
            # Prefer LV feeders (type == 2)
            candidates = torch.where(node_type == 2)[0]
            if len(candidates) == 0:
                # Fallback to all nodes
                candidates = torch.arange(graph_data.num_nodes)
        else:
            candidates = torch.arange(graph_data.num_nodes)

        # Randomly select seeds
        n_select = min(num_seeds, len(candidates))
        perm = torch.randperm(len(candidates))[:n_select]
        return candidates[perm]

    @staticmethod
    def _build_adjacency(
        edge_index: torch.Tensor, num_nodes: int
    ) -> dict[int, list[int]]:
        """Build adjacency dict from COO edge_index.

        Follows the exact pattern from CascadeLogicLayer._build_adjacency.

        Args:
            edge_index: Shape [2, num_edges] COO format tensor.
            num_nodes: Total number of nodes in the graph.

        Returns:
            Dict mapping each node index to a list of neighbor indices.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        return adj

    def _propagate_through_neighbors(
        self,
        seed_nodes: torch.Tensor,
        graph_data: Data,
        num_hops: int = 2,
        decay_factor: float = 0.7,
    ) -> dict[int, float]:
        """Propagate cascade through graph neighbors with decay.

        BFS from seed nodes through num_hops levels. Each hop applies
        decay_factor to the magnitude.

        Args:
            seed_nodes: Tensor of seed node indices.
            graph_data: PyG Data with edge_index and num_nodes.
            num_hops: Maximum number of hops to propagate.
            decay_factor: Magnitude decay per hop (default 0.7).

        Returns:
            Dict mapping node index to cascade magnitude (seed=1.0,
            hop1=decay, hop2=decay^2, etc.).
        """
        adj = self._build_adjacency(graph_data.edge_index, graph_data.num_nodes)

        # Initialize seeds at magnitude 1.0
        affected_nodes: dict[int, float] = {}
        current_frontier = set()
        for s in seed_nodes.tolist():
            affected_nodes[s] = 1.0
            current_frontier.add(s)

        visited = set(current_frontier)
        current_decay = 1.0

        for _hop in range(num_hops):
            current_decay *= decay_factor
            next_frontier: set[int] = set()
            for node in current_frontier:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
                        affected_nodes[neighbor] = current_decay
            current_frontier = next_frontier

        return affected_nodes

    def get_scenario_statistics(self) -> dict:
        """Get statistics about generated scenarios.

        Returns:
            Dictionary with scenario generation statistics
        """
        if not self.scenario_buffer:
            return {
                "total_scenarios": 0,
                "curriculum_level": self.curriculum_level,
                "scenario_types": {},
                "avg_difficulty": 0.0,
                "avg_reward": 0.0,
            }

        scenario_types = {}
        difficulties = []
        rewards = []

        for scenario, reward in self.scenario_buffer:
            st = scenario.scenario_type
            scenario_types[st] = scenario_types.get(st, 0) + 1
            difficulties.append(scenario.difficulty_score)
            rewards.append(reward)

        return {
            "total_scenarios": len(self.scenario_buffer),
            "curriculum_level": self.curriculum_level,
            "scenario_types": scenario_types,
            "avg_difficulty": np.mean(difficulties),
            "avg_reward": np.mean(rewards),
            "difficulty_std": np.std(difficulties),
            "reward_std": np.std(rewards),
        }
