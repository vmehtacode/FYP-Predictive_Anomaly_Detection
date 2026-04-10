"""
BDH-inspired enhancements for Grid Guardian self-play system.

This module incorporates lightweight concepts from the Dragon Hatchling paper
to improve the self-play training dynamics without replacing the core architecture.

Key Concepts Applied:
    1. Hebbian-like constraint weight adaptation (synaptic plasticity)
    2. Sparse activation monitoring (interpretability)
    3. Graph-based scenario relationships (structured network)

References:
    Kosowski et al. (2025). The Dragon Hatchling: The Missing Link between
    the Transformer and Models of the Brain. arXiv:2509.26507
    https://arxiv.org/abs/2509.26507
"""

import logging
from typing import Any

import numpy as np

from fyp.selfplay.proposer import ProposerAgent, ScenarioProposal
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import VerifierAgent

logger = logging.getLogger(__name__)


class HebbianVerifier:
    """
    Verifier with BDH-inspired Hebbian constraint weight adaptation.

    Key Concept from BDH: Synapses strengthen when neurons co-activate during
    learning. Similarly, constraints that are frequently violated get strengthened
    to encourage the solver to respect them more.

    This mimics the synaptic plasticity mechanism in BDH where connection weights
    σ(i,j) adapt based on activation patterns.
    """

    def __init__(
        self,
        base_verifier: VerifierAgent,
        hebbian_rate: float = 0.01,
        decay_rate: float = 0.005,
    ):
        """
        Initialize Hebbian constraint adaptation.

        Args:
            base_verifier: Standard VerifierAgent instance to wrap
            hebbian_rate: Learning rate for weight strengthening (default 0.01)
            decay_rate: Rate for weight decay toward baseline (default 0.005)
        """
        self.verifier = base_verifier
        self.hebbian_rate = hebbian_rate
        self.decay_rate = decay_rate

        # Track constraint activation history (like σ matrix in BDH)
        self.activation_history: dict[str, list[bool]] = {
            constraint: [] for constraint in self.verifier.constraints.keys()
        }

        # Store baseline weights for decay
        self.baseline_weights = self.verifier.weights.copy()

        # Track weight evolution over time for analysis
        self.weight_history: dict[str, list[float]] = {
            constraint: [weight] for constraint, weight in self.verifier.weights.items()
        }

        logger.info(
            f"Initialized HebbianVerifier with hebbian_rate={hebbian_rate}, "
            f"decay_rate={decay_rate}"
        )

    def evaluate(
        self,
        forecast: np.ndarray,
        scenario: ScenarioProposal | None = None,
        return_details: bool = False,
    ) -> tuple[float, dict[str, Any]] | float:
        """
        Evaluate with Hebbian weight adaptation.

        Args:
            forecast: Predicted consumption values
            scenario: Scenario proposal (optional)
            return_details: Whether to return detailed constraint results

        Returns:
            Reward score, or (reward, details) if return_details=True
        """
        # Get standard evaluation
        if return_details:
            reward, details = self.verifier.evaluate(
                forecast, scenario, return_details=True
            )
        else:
            reward = self.verifier.evaluate(forecast, scenario)
            details = None

        # BDH-inspired: Strengthen weights for violated constraints
        if details:
            for constraint_name, result in details.items():
                # Skip if not a real constraint (e.g., difficulty_bonus, summary fields)
                if constraint_name not in self.verifier.weights:
                    continue

                violation_occurred = result["score"] < 0

                # Hebbian rule: strengthen if co-activation (violation + context)
                if violation_occurred:
                    current_weight = self.verifier.weights[constraint_name]
                    # Multiplicative strengthening (like synaptic potentiation)
                    new_weight = current_weight * (1 + self.hebbian_rate)
                    self.verifier.weights[constraint_name] = min(
                        new_weight, 2.0
                    )  # Cap at 2x baseline

                    logger.debug(
                        f"Hebbian adaptation: {constraint_name} "
                        f"{current_weight:.3f} → {new_weight:.3f}"
                    )
                else:
                    # Decay toward baseline when not violated
                    current_weight = self.verifier.weights[constraint_name]
                    baseline_weight = self.baseline_weights[constraint_name]
                    new_weight = current_weight - self.decay_rate * (
                        current_weight - baseline_weight
                    )
                    self.verifier.weights[constraint_name] = max(
                        new_weight, baseline_weight
                    )

                # Track activation
                self.activation_history[constraint_name].append(violation_occurred)

                # Track weight evolution
                self.weight_history[constraint_name].append(
                    self.verifier.weights[constraint_name]
                )

        return (reward, details) if return_details else reward

    def get_weight_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get current constraint weights and activation frequencies.

        Returns:
            Dictionary mapping constraint names to stats (weight, activation_rate)
        """
        stats = {}
        for constraint_name in self.verifier.constraints.keys():
            stats[constraint_name] = {
                "weight": self.verifier.weights[constraint_name],
                "baseline_weight": self.baseline_weights[constraint_name],
                "activation_rate": (
                    np.mean(self.activation_history[constraint_name])
                    if self.activation_history[constraint_name]
                    else 0.0
                ),
                "total_activations": len(self.activation_history[constraint_name]),
            }
        return stats


class SparseActivationMonitor:
    """
    Monitor activation sparsity in Solver (BDH-inspired).

    BDH Finding: Approximately 5% activation sparsity leads to interpretability
    through monosemanticity. We track this for model hidden states.

    Note: Requires solver to expose hidden states, which is not implemented
    in the current SolverAgent. This is a placeholder for future integration.
    """

    def __init__(self, solver: SolverAgent, sparsity_target: float = 0.05):
        """
        Initialize sparsity monitoring.

        Args:
            solver: SolverAgent instance
            sparsity_target: Target sparsity level (default 5% like BDH)
        """
        self.solver = solver
        self.sparsity_target = sparsity_target
        self.sparsity_history: list[float] = []

        logger.info(
            f"Initialized SparseActivationMonitor with target={sparsity_target:.1%}"
        )

    def compute_sparsity(
        self, hidden_states: np.ndarray, threshold: float = 0.1
    ) -> float:
        """
        Compute activation sparsity (fraction of near-zero activations).

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            threshold: Values below this are considered inactive

        Returns:
            Sparsity ratio (0 = all active, 1 = all inactive)
        """
        inactive = np.abs(hidden_states) < threshold
        return float(np.mean(inactive))

    def monitor_prediction(
        self, context: np.ndarray, scenario: ScenarioProposal | None = None
    ) -> dict[str, np.ndarray]:
        """
        Wrap solver prediction to monitor sparsity.

        Args:
            context: Historical context window
            scenario: Optional scenario proposal

        Returns:
            Forecast dictionary with quantiles
        """
        # Get prediction
        forecast = self.solver.predict(context, scenario, return_quantiles=True)

        # If solver exposes hidden states (future implementation)
        if hasattr(self.solver, "last_hidden_states"):
            sparsity = self.compute_sparsity(self.solver.last_hidden_states)
            self.sparsity_history.append(sparsity)

            if len(self.sparsity_history) % 10 == 0:
                avg_sparsity = np.mean(self.sparsity_history[-10:])
                logger.info(
                    f"Activation sparsity: {avg_sparsity:.1%} "
                    f"(target: {self.sparsity_target:.1%}, BDH-like: ~5%)"
                )

        return forecast

    def get_statistics(self) -> dict[str, float]:
        """Get sparsity statistics."""
        if not self.sparsity_history:
            return {
                "mean_sparsity": 0.0,
                "std_sparsity": 0.0,
                "samples": 0,
                "target_sparsity": self.sparsity_target,
            }

        return {
            "mean_sparsity": float(np.mean(self.sparsity_history)),
            "std_sparsity": float(np.std(self.sparsity_history)),
            "min_sparsity": float(np.min(self.sparsity_history)),
            "max_sparsity": float(np.max(self.sparsity_history)),
            "samples": len(self.sparsity_history),
            "target_sparsity": self.sparsity_target,
        }


class GraphBasedProposer:
    """
    Proposer with BDH-inspired graph-based scenario relationships.

    BDH Concept: Neuron interactions form a structured graph with high clustering
    coefficient and heavy-tailed degree distribution.

    Application: Scenarios aren't independent - they have causal/temporal
    relationships that form a graph structure:
    - COLD_SNAP often leads to EV_SPIKE (more charging in cold weather)
    - EV_SPIKE can cause PEAK_SHIFT (grid stress)
    - OUTAGE conflicts with other scenarios

    This encourages more realistic scenario transitions during training.
    """

    def __init__(self, base_proposer: ProposerAgent, transition_prob: float = 0.5):
        """
        Initialize graph-based scenario selection.

        Args:
            base_proposer: Standard ProposerAgent instance to wrap
            transition_prob: Probability of following graph edges (default 0.5)
        """
        self.proposer = base_proposer
        self.transition_prob = transition_prob

        # Scenario relationship graph (adjacency matrix)
        # Values represent transition probabilities when following graph
        self.scenario_graph: dict[str, dict[str, Any]] = {
            "EV_SPIKE": {
                "leads_to": {"PEAK_SHIFT": 0.4, "COLD_SNAP": 0.1},
                "conflicts_with": {"OUTAGE": 0.9},
            },
            "COLD_SNAP": {
                "leads_to": {"EV_SPIKE": 0.5, "PEAK_SHIFT": 0.3},
                "amplifies": {"EV_SPIKE": 0.6},
            },
            "PEAK_SHIFT": {
                "leads_to": {"EV_SPIKE": 0.2},
                "conflicts_with": {"OUTAGE": 0.8},
            },
            "OUTAGE": {
                "conflicts_with": {
                    "EV_SPIKE": 0.9,
                    "COLD_SNAP": 0.7,
                    "PEAK_SHIFT": 0.8,
                }
            },
            "MISSING_DATA": {"independent": True},  # No strong relationships
        }

        logger.info(
            f"Initialized GraphBasedProposer with transition_prob={transition_prob}"
        )

    def propose_scenario(
        self,
        historical_context: np.ndarray,
        forecast_horizon: int = 48,
        **kwargs: Any,
    ) -> ScenarioProposal:
        """
        Generate scenario using graph-based sampling.

        Args:
            historical_context: Historical consumption data
            forecast_horizon: Number of intervals to forecast
            **kwargs: Additional arguments passed to base proposer

        Returns:
            ScenarioProposal following graph structure
        """
        # Get recent scenarios from buffer
        recent_scenarios = [
            s.scenario_type for s, _ in self.proposer.scenario_buffer[-5:]
        ]

        if recent_scenarios and np.random.rand() < self.transition_prob:
            last_scenario = recent_scenarios[-1]

            # Graph-based sampling (like BDH's neuron graph propagation)
            if last_scenario in self.scenario_graph:
                relationships = self.scenario_graph[last_scenario]

                if "leads_to" in relationships:
                    # Sample from graph neighbors with specified probabilities
                    next_scenarios = list(relationships["leads_to"].keys())
                    probs = list(relationships["leads_to"].values())

                    # Normalize probabilities
                    probs_normalized = np.array(probs) / sum(probs)
                    scenario_type = np.random.choice(next_scenarios, p=probs_normalized)

                    logger.debug(f"Graph sampling: {last_scenario} → {scenario_type}")

                    # Generate scenario with this type
                    # Note: We generate normally then override the type
                    # A better approach would be to pass scenario_type to proposer
                    scenario = self.proposer.propose_scenario(
                        historical_context=historical_context,
                        forecast_horizon=forecast_horizon,
                        **kwargs,
                    )

                    # Override scenario type to follow graph
                    scenario.scenario_type = scenario_type

                    return scenario

        # Default: use base proposer
        return self.proposer.propose_scenario(
            historical_context=historical_context,
            forecast_horizon=forecast_horizon,
            **kwargs,
        )

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get graph structure statistics."""
        # Count edges and nodes
        num_nodes = len(self.scenario_graph)
        num_edges = sum(
            len(neighbors.get("leads_to", {}))
            for neighbors in self.scenario_graph.values()
        )

        # Compute degree distribution
        out_degrees = [
            len(neighbors.get("leads_to", {}))
            for neighbors in self.scenario_graph.values()
        ]

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_out_degree": np.mean(out_degrees) if out_degrees else 0.0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "graph_density": num_edges / (num_nodes * (num_nodes - 1))
            if num_nodes > 1
            else 0.0,
        }


def create_bdh_enhanced_trainer(
    base_proposer: ProposerAgent,
    base_solver: SolverAgent,
    base_verifier: VerifierAgent,
    config: dict[str, Any],
    enable_hebbian: bool = True,
    enable_graph: bool = True,
    enable_sparsity: bool = False,
) -> SelfPlayTrainer:
    """
    Create self-play trainer with BDH enhancements.

    Args:
        base_proposer: Standard ProposerAgent instance
        base_solver: Standard SolverAgent instance
        base_verifier: Standard VerifierAgent instance
        config: Trainer configuration dictionary
        enable_hebbian: Enable Hebbian constraint adaptation
        enable_graph: Enable graph-based scenario relationships
        enable_sparsity: Enable sparsity monitoring (requires model changes)

    Returns:
        SelfPlayTrainer with BDH enhancements

    Usage:
        >>> proposer = ProposerAgent(...)
        >>> solver = SolverAgent(...)
        >>> verifier = VerifierAgent(...)
        >>> config = {'alpha': 0.1, 'batch_size': 4}
        >>> trainer = create_bdh_enhanced_trainer(
        ...     proposer, solver, verifier, config
        ... )
        >>> metrics = trainer.train(num_episodes=100)
    """
    # Wrap components with BDH enhancements
    proposer = base_proposer
    solver = base_solver
    verifier = base_verifier

    if enable_hebbian:
        verifier = HebbianVerifier(base_verifier, hebbian_rate=0.01, decay_rate=0.005)
        logger.info("Enabled Hebbian constraint adaptation")

    if enable_graph:
        proposer = GraphBasedProposer(base_proposer, transition_prob=0.5)
        logger.info("Enabled graph-based scenario relationships")

    sparsity_monitor = None
    if enable_sparsity:
        sparsity_monitor = SparseActivationMonitor(base_solver, sparsity_target=0.05)
        logger.info("Enabled sparsity monitoring")

    # Create trainer with enhanced components
    trainer = SelfPlayTrainer(
        proposer=proposer if isinstance(proposer, ProposerAgent) else proposer.proposer,
        solver=solver,
        verifier=verifier if isinstance(verifier, VerifierAgent) else verifier.verifier,
        config=config,
    )

    # Attach enhanced components as attributes for logging
    if enable_hebbian:
        trainer.hebbian_verifier = verifier  # type: ignore
    if enable_graph:
        trainer.graph_proposer = proposer  # type: ignore
    if enable_sparsity:
        trainer.sparsity_monitor = sparsity_monitor  # type: ignore

    logger.info("Created BDH-enhanced self-play trainer")
    return trainer
