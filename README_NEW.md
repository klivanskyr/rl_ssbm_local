# SSBM Reinforcement Learning Framework

A modular, production-ready framework for training reinforcement learning agents to play Super Smash Bros Melee.

## 🎯 Project Structure

```
.
├── ssbm_rl/                    # Main package
│   ├── envs/                   # Environment wrappers
│   │   ├── melee_env.py       # Main RL environment
│   │   ├── state_extractor.py # State feature extraction
│   │   └── reward_function.py # Reward computation
│   ├── policies/               # Policy implementations
│   │   ├── base_policy.py     # Abstract policy interface
│   │   ├── random_policy.py   # Random baseline
│   │   └── ppo_policy.py      # PPO implementation
│   └── utils/                  # Utilities
│       ├── console_manager.py # Dolphin management
│       ├── menu_navigator.py  # Menu automation
│       └── logger.py          # Training logger
├── configs/                    # Configuration files
│   └── train_config.yaml      # Training configuration
├── train.py                    # Generic training script
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Build Docker Image

```bash
docker build -f Dockerfile.new -t melee-rl .
```

### 2. Run Training with Random Policy

```bash
docker run --rm -it \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/ssbm_rl:/opt/melee/ssbm_rl:ro \
    -v $(pwd)/configs:/opt/melee/configs:ro \
    -v $(pwd)/train.py:/opt/melee/train.py:ro \
    -v $(pwd)/checkpoints:/opt/melee/checkpoints:rw \
    -v $(pwd)/logs:/opt/melee/logs:rw \
    melee-rl \
    python3 train.py --policy random --episodes 10
```

### 3. Run Training with PPO

```bash
docker run --rm -it \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/ssbm_rl:/opt/melee/ssbm_rl:ro \
    -v $(pwd)/configs:/opt/melee/configs:ro \
    -v $(pwd)/train.py:/opt/melee/train.py:ro \
    -v $(pwd)/checkpoints:/opt/melee/checkpoints:rw \
    -v $(pwd)/logs:/opt/melee/logs:rw \
    melee-rl \
    python3 train.py --policy ppo --episodes 1000 --save-freq 100
```

## 📝 Configuration

Edit `configs/train_config.yaml` to customize:

```yaml
# Game Settings
player_character: "MARTH"
cpu_level: 9
stage: "YOSHIS_STORY"

# Reward Function
damage_dealt_weight: 0.01
stock_won_reward: 1.0
win_bonus: 5.0

# Training
total_episodes: 1000
policy_type: "ppo"

# PPO Hyperparameters
ppo:
  learning_rate: 0.0003
  gamma: 0.99
  epsilon: 0.2
```

## 🏗️ Architecture

### Environment (`MeleeEnv`)

Wraps libmelee to provide a clean RL interface:

```python
from ssbm_rl.envs import MeleeEnv

env = MeleeEnv(console, player_controller, opponent_controller)
state, reward, done, info = env.step()
```

**State**: 14-dimensional feature vector (positions, damage, stocks, etc.)  
**Reward**: Configurable reward function (damage dealt, stocks won, etc.)  
**Done**: Episode ends when game finishes

### Policies

All policies implement the `BasePolicy` interface:

```python
class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, state):
        """Get action from policy"""
        pass
    
    @abstractmethod
    def update(self, trajectory):
        """Update policy from trajectory"""
        pass
```

**Available Policies:**
- `RandomPolicy`: Random baseline
- `PPOPolicy`: Proximal Policy Optimization

### Utilities

- **ConsoleManager**: Manages Dolphin lifecycle
- **MenuNavigator**: Automates menu navigation
- **TrainingLogger**: Logs training statistics

## 📊 Monitoring Training

### View Logs

```bash
# Real-time progress
docker logs -f <container_id>

# Training statistics (JSONL format)
cat logs/experiment_*.jsonl | jq '.'
```

### Checkpoints

Models are saved to `checkpoints/` every N episodes (configurable):

```
checkpoints/
├── policy_ep100.pt
├── policy_ep200.pt
└── policy_final.pt
```

### Load Checkpoint

```python
from ssbm_rl.policies.ppo_policy import PPOPolicy

policy = PPOPolicy()
policy.load("checkpoints/policy_final.pt")
```

## 🔧 Extending the Framework

### Add New Policy

```python
# ssbm_rl/policies/my_policy.py
from ssbm_rl.policies.base_policy import BasePolicy

class MyPolicy(BasePolicy):
    def get_action(self, state):
        # Your policy logic
        return action
    
    def update(self, trajectory):
        # Your learning algorithm
        pass
```

### Customize Reward Function

```python
from ssbm_rl.envs import RewardFunction

reward_fn = RewardFunction(
    damage_dealt_weight=0.02,  # Emphasize damage
    stock_won_reward=2.0,       # Big reward for stocks
    win_bonus=10.0              # Huge win bonus
)
```

### Modify State Representation

```python
from ssbm_rl.envs import StateExtractor

class MyStateExtractor(StateExtractor):
    def extract(self, gamestate, player_port, opponent_port):
        # Custom state features
        return custom_state_vector
```

## 🎮 Action Space

Current implementation uses 5 discrete actions:
- A button
- B button
- L button
- R button
- No action

### Expand Action Space

To add directional inputs, modify the action execution in `train.py`:

```python
# Example: Add stick positions
actions = [
    (melee.Button.BUTTON_A, 0.5, 0.5),   # A + neutral
    (melee.Button.BUTTON_B, 1.0, 0.5),   # B + right
    (melee.Button.BUTTON_L, 0.5, 0.0),   # L + down
    # ... more combinations
]
```

## 🐛 Troubleshooting

### Dolphin won't start
```bash
# Check if Dolphin is already running
docker ps -a | grep melee

# Remove stale containers
docker rm -f $(docker ps -a -q --filter ancestor=melee-rl)
```

### Training is slow
- Reduce `max_steps_per_episode` in config
- Use CPU-only PyTorch (already configured)
- Run on faster hardware

### Out of memory
- Reduce `hidden_dim` in PPO config
- Use smaller batches (modify PPO update)
- Close other applications

## 📚 References

- **LibMelee**: https://github.com/altf4/libmelee
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Slippi**: https://slippi.gg/

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Happy Training!** 🎮🤖
