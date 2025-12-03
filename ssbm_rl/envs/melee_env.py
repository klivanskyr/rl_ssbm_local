"""
Main environment wrapper for SSBM
"""

import melee
from ssbm_rl.envs.state_extractor import StateExtractor
from ssbm_rl.envs.reward_function import RewardFunction


class MeleeEnv:
    """
    Environment wrapper for Super Smash Bros Melee
    
    Provides a clean interface for RL training:
    - step(): Returns (state, reward, done, info)
    - reset(): Resets to character select
    - close(): Closes the emulator
    """
    
    def __init__(self, 
                 console,
                 player_controller,
                 opponent_controller,
                 player_port=1,
                 opponent_port=2,
                 state_extractor=None,
                 reward_function=None):
        """
        Args:
            console: libmelee Console object
            player_controller: libmelee Controller for the agent
            opponent_controller: libmelee Controller for opponent
            player_port: Port number for agent (default 1)
            opponent_port: Port number for opponent (default 2)
            state_extractor: StateExtractor instance (optional)
            reward_function: RewardFunction instance (optional)
        """
        self.console = console
        self.player_controller = player_controller
        self.opponent_controller = opponent_controller
        self.player_port = player_port
        self.opponent_port = opponent_port
        
        # Initialize state extractor and reward function
        self.state_extractor = state_extractor or StateExtractor()
        self.reward_function = reward_function or RewardFunction(
            player_port=player_port,
            opponent_port=opponent_port
        )
        
        # Track previous state for reward computation
        self.previous_gamestate = None
        self.current_gamestate = None
        
        # Episode statistics
        self.episode_steps = 0
        self.episode_reward = 0.0
    
    def step(self):
        """
        Step the environment
        
        Returns:
            tuple: (state, reward, done, info)
                - state: numpy array of state features
                - reward: float reward value
                - done: boolean indicating episode end
                - info: dict with additional information
        """
        # Get next gamestate from emulator
        self.previous_gamestate = self.current_gamestate
        self.current_gamestate = self.console.step()
        
        if self.current_gamestate is None:
            return None, 0.0, False, {}
        
        # Extract state
        state = self.state_extractor.extract(
            self.current_gamestate,
            self.player_port,
            self.opponent_port
        )
        
        # Compute reward
        reward = 0.0
        if self.previous_gamestate is not None:
            reward = self.reward_function.compute(
                self.current_gamestate,
                self.previous_gamestate
            )
        
        # Check if episode is done
        done = self._is_done()
        
        # Get additional info
        info = self.state_extractor.get_info(
            self.current_gamestate,
            self.player_port,
            self.opponent_port
        )
        info['episode_steps'] = self.episode_steps
        info['episode_reward'] = self.episode_reward
        
        # Update statistics
        self.episode_steps += 1
        self.episode_reward += reward
        
        return state, reward, done, info
    
    def _is_done(self):
        """Check if episode is complete"""
        if not self.current_gamestate:
            return False
        
        # Episode ends when not in game anymore
        if self.current_gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # Check if we just finished a game
            if self.previous_gamestate and self.previous_gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return True
        
        # Also check if someone lost all stocks
        if self.current_gamestate.menu_state == melee.Menu.IN_GAME:
            player = self.current_gamestate.players.get(self.player_port)
            opponent = self.current_gamestate.players.get(self.opponent_port)
            
            if player and opponent:
                if player.stock == 0 or opponent.stock == 0:
                    return True
        
        return False
    
    def reset(self):
        """
        Reset environment to start new episode
        
        Returns:
            Initial state or None
        """
        self.previous_gamestate = None
        self.current_gamestate = None
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        return None
    
    def get_current_state(self):
        """Get current state without stepping"""
        if self.current_gamestate:
            return self.state_extractor.extract(
                self.current_gamestate,
                self.player_port,
                self.opponent_port
            )
        return None
    
    def close(self):
        """Close the environment"""
        if self.console:
            self.console.stop()
