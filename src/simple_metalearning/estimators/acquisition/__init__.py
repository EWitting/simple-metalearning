"""Acquisition functions for Bayesian Optimisation over discrete flow sets.

Available implementations
-------------------------
- :class:`GreedyAcquisition`          — pure exploitation (argmax μ)
- :class:`EIGaussianAcquisition`      — standard EI under Gaussian posterior
- :class:`LogEIGaussianAcquisition`   — log EI (numerically stable, same ranking)
- :class:`EIForestAcquisition`        — SMAC-style per-tree EI for Random Forests
- :class:`UCBAcquisition`             — Upper Confidence Bound (μ + κσ)
- :class:`MMRAcquisition`             — Maximal Marginal Relevance (diversity-aware)
- :class:`EmbeddingMMRAcquisition`    — MMR with similarity in learned embedding space
- :class:`CostCooledEIAcquisition`    — EI / cost^alpha (Snoek 2012)
"""

from .base import AcquisitionFunction
from .greedy import GreedyAcquisition
from .ei_gaussian import EIGaussianAcquisition
from .log_ei_gaussian import LogEIGaussianAcquisition
from .ei_forest import EIForestAcquisition
from .ucb import UCBAcquisition
from .mmr import MMRAcquisition, EmbeddingMMRAcquisition
from .ei_cost_cooled import CostCooledEIAcquisition

__all__ = [
    "AcquisitionFunction",
    "GreedyAcquisition",
    "EIGaussianAcquisition",
    "LogEIGaussianAcquisition",
    "EIForestAcquisition",
    "UCBAcquisition",
    "MMRAcquisition",
    "EmbeddingMMRAcquisition",
    "CostCooledEIAcquisition",
]
