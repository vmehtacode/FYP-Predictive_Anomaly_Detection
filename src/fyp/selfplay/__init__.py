"""Self-play reinforcement learning system for energy forecasting.

Components:
    - ProposerAgent: Generates challenging but realistic scenarios
    - SolverAgent: Forecasts consumption under proposed scenarios
    - VerifierAgent: Validates forecasts using physics constraints
    - SelfPlayTrainer: Runs the propose→solve→verify training loop

BDH Enhancements (Optional):
    - HebbianVerifier: Constraint adaptation via synaptic-like plasticity
    - GraphBasedProposer: Scenario sampling from causal relationship graph
    - SparseActivationMonitor: Track activation sparsity
    - create_bdh_enhanced_trainer: Helper to create BDH-enhanced trainer
"""

from fyp.selfplay.proposer import ProposerAgent, ScenarioProposal
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import (
    Constraint,
    HouseholdMaxConstraint,
    NonNegativityConstraint,
    RampRateConstraint,
    VerifierAgent,
)

# BDH enhancements (optional imports - won't break if not used)
try:
    from fyp.selfplay.bdh_enhancements import (
        GraphBasedProposer,
        HebbianVerifier,
        SparseActivationMonitor,
        create_bdh_enhanced_trainer,
    )

    _bdh_available = True
except ImportError:
    _bdh_available = False

__all__ = [
    # Core components
    "ProposerAgent",
    "ScenarioProposal",
    "SolverAgent",
    "VerifierAgent",
    "SelfPlayTrainer",
    "Constraint",
    "NonNegativityConstraint",
    "HouseholdMaxConstraint",
    "RampRateConstraint",
]

# Add BDH enhancements if available
if _bdh_available:
    __all__.extend(
        [
            "HebbianVerifier",
            "GraphBasedProposer",
            "SparseActivationMonitor",
            "create_bdh_enhanced_trainer",
        ]
    )
