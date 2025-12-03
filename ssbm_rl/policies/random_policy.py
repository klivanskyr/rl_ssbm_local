"""
Random policy for baseline testing
"""

import random
import melee
from ssbm_rl.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """
    Random policy that samples actions uniformly
    
    Useful for:
    - Baseline performance comparison
    - Environment testing
    - Initial exploration
    """
    
    def __init__(self, action_space=None):
        """
        Args:
            action_space: List of possible actions (default: basic buttons)
        """
        if action_space is None:
            self.action_space = [
                melee.Button.BUTTON_A,
                melee.Button.BUTTON_B,
                melee.Button.BUTTON_L,
                melee.Button.BUTTON_R,
                None,  # No action
            ]
        else:
            self.action_space = action_space
    
    def get_action(self, state):
        """
        Sample random action
        
        Args:
            state: Current state (unused for random policy)
            
        Returns:
            Random action from action_space
        """
        return random.choice(self.action_space)
    
    def update(self, trajectory):
        """
        No-op for random policy (doesn't learn)
        
        Args:
            trajectory: List of (state, action, reward) tuples (unused)
        """
        pass
