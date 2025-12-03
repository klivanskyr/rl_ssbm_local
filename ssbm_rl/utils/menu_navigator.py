"""
Menu navigation utilities
"""

import melee
import random


class MenuNavigator:
    """
    Handles menu navigation in SSBM
    
    Navigates to character select and starts games automatically
    """
    
    def __init__(self,
                 player_character=melee.Character.MARTH,
                 player_costume=0,
                 opponent_character=None,
                 opponent_costume=0,
                 stage=melee.Stage.YOSHIS_STORY,
                 cpu_level=9):
        """
        Args:
            player_character: Character for player 1
            player_costume: Costume for player 1
            opponent_character: Character for player 2 (None for random)
            opponent_costume: Costume for player 2
            stage: Stage to play on
            cpu_level: CPU difficulty (1-9)
        """
        self.player_character = player_character
        self.player_costume = player_costume
        self.opponent_character = opponent_character
        self.opponent_costume = opponent_costume
        self.stage = stage
        self.cpu_level = cpu_level
        
        # Available characters for random selection
        self.available_characters = [
            melee.Character.FOX,
            melee.Character.FALCO,
            melee.Character.MARTH,
            melee.Character.SHEIK,
            melee.Character.JIGGLYPUFF,
            melee.Character.PEACH,
            melee.Character.CPTFALCON,
            melee.Character.SAMUS,
        ]
    
    def navigate(self, gamestate, player_controller, opponent_controller, autostart=True):
        """
        Navigate menus to start game
        
        Args:
            gamestate: Current GameState
            player_controller: Player controller
            opponent_controller: Opponent controller
            autostart: Whether to auto-start when both players ready
        """
        if not gamestate:
            return
        
        # Randomize opponent if not set
        if self.opponent_character is None:
            self.opponent_character = random.choice(self.available_characters)
            self.opponent_costume = random.randint(0, 3)
        
        # Navigate player
        melee.MenuHelper.menu_helper_simple(
            gamestate,
            player_controller,
            self.player_character,
            self.stage,
            "",
            costume=self.player_costume,
            autostart=autostart,
            swag=False
        )
        
        # Navigate opponent (CPU)
        melee.MenuHelper.menu_helper_simple(
            gamestate,
            opponent_controller,
            self.opponent_character,
            self.stage,
            "",
            costume=self.opponent_costume,
            autostart=False,
            swag=False,
            cpu_level=self.cpu_level
        )
    
    def randomize_opponent(self):
        """Randomize opponent character and costume"""
        self.opponent_character = random.choice(self.available_characters)
        self.opponent_costume = random.randint(0, 3)
    
    def check_ready(self, gamestate, player_port=1, opponent_port=2):
        """
        Check if both players are ready to start
        
        Returns:
            bool: True if both players are ready
        """
        if not gamestate or gamestate.menu_state != melee.Menu.CHARACTER_SELECT:
            return False
        
        if player_port not in gamestate.players or opponent_port not in gamestate.players:
            return False
        
        player = gamestate.players[player_port]
        opponent = gamestate.players[opponent_port]
        
        # Check if both have valid characters
        return (
            player.character == self.player_character and
            opponent.controller_status == melee.ControllerStatus.CONTROLLER_CPU and
            opponent.cpu_level == self.cpu_level and
            opponent.character is not None and
            opponent.character != melee.Character.UNKNOWN_CHARACTER
        )
