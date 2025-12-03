"""
Reward function for SSBM RL training
"""

import melee


class RewardFunction:
    """Compute rewards for SSBM gameplay"""
    
    def __init__(self, 
                 damage_dealt_weight=0.01,
                 damage_taken_weight=-0.01,
                 stock_won_reward=1.0,
                 stock_lost_penalty=-1.0,
                 distance_penalty_weight=-0.0001,
                 win_bonus=5.0,
                 loss_penalty=-5.0,
                 player_port=1,
                 opponent_port=2):
        """
        Args:
            damage_dealt_weight: Reward per % damage dealt to opponent
            damage_taken_weight: Penalty per % damage taken
            stock_won_reward: Reward for taking opponent's stock
            stock_lost_penalty: Penalty for losing own stock
            distance_penalty_weight: Penalty for being far from opponent
            win_bonus: Bonus for winning the game
            loss_penalty: Penalty for losing the game
            player_port: Port number for the agent
            opponent_port: Port number for opponent
        """
        self.damage_dealt_weight = damage_dealt_weight
        self.damage_taken_weight = damage_taken_weight
        self.stock_won_reward = stock_won_reward
        self.stock_lost_penalty = stock_lost_penalty
        self.distance_penalty_weight = distance_penalty_weight
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.player_port = player_port
        self.opponent_port = opponent_port
    
    def compute(self, current_state, previous_state):
        """
        Compute reward for current transition
        
        Args:
            current_state: Current GameState
            previous_state: Previous GameState
            
        Returns:
            float: Reward value
        """
        if not current_state or not previous_state:
            return 0.0
        
        if current_state.menu_state != melee.Menu.IN_GAME:
            return 0.0
        
        player = current_state.players.get(self.player_port)
        opponent = current_state.players.get(self.opponent_port)
        player_prev = previous_state.players.get(self.player_port)
        opponent_prev = previous_state.players.get(self.opponent_port)
        
        if not all([player, opponent, player_prev, opponent_prev]):
            return 0.0
        
        reward = 0.0
        
        # Reward for damaging opponent
        damage_dealt = opponent.percent - opponent_prev.percent
        reward += damage_dealt * self.damage_dealt_weight
        
        # Reward for taking opponent's stock
        if opponent.stock < opponent_prev.stock:
            reward += self.stock_won_reward
        
        # Penalty for taking damage
        damage_taken = player.percent - player_prev.percent
        reward += damage_taken * self.damage_taken_weight
        
        # Penalty for losing stock
        if player.stock < player_prev.stock:
            reward += self.stock_lost_penalty
        
        # Distance penalty (encourage engagement)
        distance = abs(player.position.x - opponent.position.x)
        reward += distance * self.distance_penalty_weight
        
        # Terminal rewards
        if opponent.stock == 0 and player.stock > 0:
            reward += self.win_bonus
        elif player.stock == 0 and opponent.stock > 0:
            reward += self.loss_penalty
        
        return reward
    
    def compute_sparse(self, current_state, previous_state):
        """
        Sparse reward: only at end of game
        
        Returns:
            float: +1 for win, -1 for loss, 0 otherwise
        """
        if not current_state or current_state.menu_state != melee.Menu.IN_GAME:
            return 0.0
        
        player = current_state.players.get(self.player_port)
        opponent = current_state.players.get(self.opponent_port)
        
        if not player or not opponent:
            return 0.0
        
        if opponent.stock == 0 and player.stock > 0:
            return 1.0
        elif player.stock == 0 and opponent.stock > 0:
            return -1.0
        
        return 0.0
