"""
PPO (Proximal Policy Optimization) policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ssbm_rl.policies.base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    """
    Proximal Policy Optimization policy
    
    Reference: https://arxiv.org/abs/1707.06347
    """
    
    def __init__(self,
                 state_dim=14,
                 action_dim=5,
                 hidden_dim=128,
                 lr=3e-4,
                 gamma=0.99,
                 epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 device='cpu'):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Size of hidden layers
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Build actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def get_action(self, state, deterministic=False):
        """
        Get action from policy
        
        Args:
            state: numpy array of state features
            deterministic: If True, return mode instead of sampling
            
        Returns:
            tuple: (action_index, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, trajectories, epochs=10):
        """
        Update policy using PPO algorithm
        
        Args:
            trajectories: List of trajectories, where each trajectory is a list of
                         (state, action, reward, log_prob, value) tuples
            epochs: Number of optimization epochs
            
        Returns:
            dict: Training statistics
        """
        # Convert trajectories to tensors
        states, actions, rewards, old_log_probs, values = self._process_trajectories(trajectories)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
        }
        
        for _ in range(epochs):
            # Forward pass
            action_probs, new_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            # Record stats
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
            stats['total_loss'].append(loss.item())
        
        # Average stats
        for key in stats:
            stats[key] = np.mean(stats[key])
        
        return stats
    
    def _process_trajectories(self, trajectories):
        """Convert trajectories to tensors"""
        states_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []
        
        for traj in trajectories:
            for state, action, reward, log_prob, value in traj:
                states_list.append(state)
                actions_list.append(action)
                rewards_list.append(reward)
                log_probs_list.append(log_prob)
                values_list.append(value)
        
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        rewards = torch.FloatTensor(rewards_list).to(self.device)
        log_probs = torch.FloatTensor(log_probs_list).to(self.device)
        values = torch.FloatTensor(values_list).to(self.device)
        
        return states, actions, rewards, log_probs, values
    
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save(self, path):
        """Save policy to disk"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load policy from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ActorCritic(nn.Module):
    """Actor-Critic network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass
        
        Returns:
            tuple: (action_probs, value)
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
