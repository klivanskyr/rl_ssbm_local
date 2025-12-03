"""
SSBM Reinforcement Learning Package

A modular framework for training RL agents to play Super Smash Bros Melee.
"""

__version__ = "0.1.0"

from ssbm_rl.envs.melee_env import MeleeEnv
from ssbm_rl.policies.base_policy import BasePolicy
from ssbm_rl.policies.random_policy import RandomPolicy

__all__ = [
    "MeleeEnv",
    "BasePolicy", 
    "RandomPolicy",
]
