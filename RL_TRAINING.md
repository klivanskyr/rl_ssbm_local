# Reinforcement Learning Training for SSBM Bot

## Overview

This setup trains a Marth bot to beat CPU Level 9 opponents using **Proximal Policy Optimization (PPO)**. Unlike imitation learning from replay files, RL learns by **playing live games** and optimizing for reward.

## How It Works

### No .slp Files Needed!

```
┌──────────────────────────────────────────────────────────────┐
│                    RL Training Loop                           │
│                                                               │
│  1. Dolphin Emulator (live game)                             │
│          ↓                                                    │
│  2. Get GameState (every frame)                              │
│          ↓                                                    │
│  3. Extract State Features                                   │
│          ↓                                                    │
│  4. Policy Network → Sample Action                           │
│          ↓                                                    │
│  5. Execute Action in Emulator                               │
│          ↓                                                    │
│  6. Compute Reward (damage dealt, stocks won, etc.)          │
│          ↓                                                    │
│  7. Store (state, action, reward) in trajectory              │
│          ↓                                                    │
│  8. After episode: Update Policy with PPO                    │
│          ↓                                                    │
│  9. Repeat for next episode                                  │
└──────────────────────────────────────────────────────────────┘
```

### Current Implementation

**File**: `train_ppo.py`

**Features**:
- ✅ Environment wrapper (`MeleeEnv`) that extracts state from live GameState
- ✅ Reward function (damage dealt, stocks won, positioning)
- ✅ Trajectory collection (state, action, reward tuples)
- ✅ Training loop structure
- ⚠️  Random policy (placeholder - needs PPO implementation)

## State Representation

Each frame's state is a 14-dimensional vector:

```python
state = [
    # Bot (P1)
    x_position,      # Normalized position
    y_position,
    damage_percent,  # Current damage
    stocks,          # Remaining stocks
    facing_right,    # Boolean
    action_id,       # Current animation
    
    # Opponent (P2)
    x_position,
    y_position,
    damage_percent,
    stocks,
    facing_right,
    action_id,
    
    # Relative
    relative_x,      # Distance between players
    relative_y,
]
```

## Reward Function

```python
reward = (
    + 0.01 × damage_dealt_to_opponent
    + 1.0  × opponent_stock_taken
    - 0.01 × damage_taken
    - 1.0  × own_stock_lost
    - 0.0001 × distance_from_opponent  # Encourage engagement
    + 5.0  × win_bonus
    - 5.0  × loss_penalty
)
```

## Next Steps: Implement PPO

### 1. Install PyTorch

```bash
pip install torch numpy
```

### 2. Create PPO Policy Network

Replace `RandomPolicy` with actual PPO:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PPOPolicy(nn.Module):
    def __init__(self, state_dim=14, action_dim=4, hidden_dim=128):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.forward(state_tensor)
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def update(self, trajectories, optimizer, epsilon=0.2, epochs=10):
        """PPO update"""
        states, actions, rewards, old_log_probs = zip(*trajectories)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Compute advantages (simple version - use GAE for better results)
        _, values = self.forward(states)
        advantages = rewards - values.detach().squeeze()
        
        for _ in range(epochs):
            # Get current policy
            action_probs, values = self.forward(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 3. Modify Action Space

Current actions are too simple. Consider:
- Directional inputs (left/right stick)
- Button combinations (short hop, fast fall, etc.)
- Character-specific tech (wavedash, L-cancel)

### 4. Training Optimizations

- **Curriculum learning**: Start with CPU level 1, gradually increase
- **Self-play**: Train against copies of itself
- **Experience replay**: Store and reuse good trajectories
- **Parallel environments**: Run multiple Dolphin instances

## Running Training

### Current (Random Policy)

```bash
# Build Docker image
docker build -t melee-bot .

# Run training
docker run --rm -it \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/train_ppo.py:/opt/melee/train_ppo.py:ro \
    melee-bot \
    python3 /opt/melee/train_ppo.py --episodes 10
```

### With PPO (After Implementation)

```bash
# Mount model checkpoint directory
docker run --rm -it \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/train_ppo.py:/opt/melee/train_ppo.py:ro \
    -v $(pwd)/checkpoints:/opt/melee/checkpoints:rw \
    melee-bot \
    python3 /opt/melee/train_ppo.py --episodes 1000
```

## Advanced: Use Existing RL Frameworks

Consider using established RL implementations:

### Option 1: Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap MeleeEnv in gym interface
env = DummyVecEnv([lambda: GymWrapper(MeleeEnv(...))])

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("marth_ppo")
```

### Option 2: RLlib (Ray)

```python
from ray.rllib.agents.ppo import PPOTrainer

trainer = PPOTrainer(config={
    "env": MeleeEnv,
    "num_workers": 4,  # Parallel envs
    "framework": "torch",
})

for i in range(1000):
    result = trainer.train()
    if i % 100 == 0:
        checkpoint = trainer.save()
```

## Monitoring Training

Track these metrics:
- Average episode reward
- Win rate vs CPU
- Average damage dealt/taken per episode
- Policy entropy (exploration vs exploitation)

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/

