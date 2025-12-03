"""
State extraction from libmelee GameState
"""

import numpy as np
import melee


class StateExtractor:
    """Extract state features from libmelee GameState"""
    
    def __init__(self, state_dim=14, normalize=True):
        """
        Args:
            state_dim: Dimensionality of state vector
            normalize: Whether to normalize features to [-1, 1] range
        """
        self.state_dim = state_dim
        self.normalize = normalize
    
    def extract(self, gamestate, player_port=1, opponent_port=2):
        """
        Extract state features from current gamestate
        
        Args:
            gamestate: libmelee GameState object
            player_port: Port number for the agent (default 1)
            opponent_port: Port number for opponent (default 2)
            
        Returns:
            numpy array of shape (state_dim,) or None if invalid state
        """
        if not gamestate or gamestate.menu_state != melee.Menu.IN_GAME:
            return None
        
        player = gamestate.players.get(player_port)
        opponent = gamestate.players.get(opponent_port)
        
        if not player or not opponent:
            return None
        
        # Extract raw features
        features = [
            # Player state
            player.position.x,
            player.position.y,
            player.percent,
            player.stock,
            float(player.facing),
            float(player.action.value),
            player.speed_air_x_self,
            player.speed_y_self,
            player.speed_x_attack,
            player.speed_y_attack,
            
            # Opponent state
            opponent.position.x,
            opponent.position.y,
            opponent.percent,
            opponent.stock,
            float(opponent.facing),
            float(opponent.action.value),
            opponent.speed_air_x_self,
            opponent.speed_y_self,
            
            # Relative state
            player.position.x - opponent.position.x,
            player.position.y - opponent.position.y,
            
            # Stage info
            float(gamestate.stage.value if hasattr(gamestate, 'stage') else 0),
        ]
        
        state = np.array(features[:self.state_dim], dtype=np.float32)
        
        if self.normalize:
            state = self._normalize(state)
        
        return state
    
    def _normalize(self, state):
        """Normalize state features to reasonable ranges"""
        # Simple normalization - can be improved with running statistics
        normalization_factors = np.array([
            100.0,  # x position
            100.0,  # y position
            200.0,  # percent
            4.0,    # stocks
            1.0,    # facing
            400.0,  # action
            10.0,   # speed_x
            10.0,   # speed_y
            10.0,   # attack_speed_x
            10.0,   # attack_speed_y
            100.0,  # opponent x
            100.0,  # opponent y
            200.0,  # opponent percent
            4.0,    # opponent stocks
        ], dtype=np.float32)
        
        # Ensure we don't divide by zero
        normalization_factors = np.maximum(normalization_factors[:len(state)], 1e-8)
        return state / normalization_factors
    
    def get_info(self, gamestate, player_port=1, opponent_port=2):
        """
        Extract additional info (not part of state, but useful for logging)
        
        Returns:
            dict with additional information
        """
        if not gamestate or gamestate.menu_state != melee.Menu.IN_GAME:
            return {}
        
        player = gamestate.players.get(player_port)
        opponent = gamestate.players.get(opponent_port)
        
        if not player or not opponent:
            return {}
        
        return {
            'player_stock': player.stock,
            'player_percent': player.percent,
            'opponent_stock': opponent.stock,
            'opponent_percent': opponent.percent,
            'player_action': player.action.name if hasattr(player.action, 'name') else str(player.action),
            'opponent_action': opponent.action.name if hasattr(opponent.action, 'name') else str(opponent.action),
        }
