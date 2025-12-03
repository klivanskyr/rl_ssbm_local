"""
Base policy interface
"""

from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Abstract base class for all policies"""
    
    @abstractmethod
    def get_action(self, state):
        """
        Get action from policy given current state
        
        Args:
            state: numpy array of state features
            
        Returns:
            action: Action to execute
        """
        pass
    
    @abstractmethod
    def update(self, trajectory):
        """
        Update policy parameters given trajectory
        
        Args:
            trajectory: List of (state, action, reward) tuples
        """
        pass
    
    def save(self, path):
        """Save policy to disk"""
        raise NotImplementedError("save() not implemented")
    
    def load(self, path):
        """Load policy from disk"""
        raise NotImplementedError("load() not implemented")
