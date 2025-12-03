# SSBM RL Framework - Quick Reference

## 🚀 Quick Start

```bash
# 1. Setup
chmod +x setup.sh run_training.sh
./setup.sh

# 2. Run training
./run_training.sh random 10          # Random policy, 10 episodes
./run_training.sh ppo 1000           # PPO policy, 1000 episodes
```

## 📁 File Organization

### Core Modules
- `ssbm_rl/envs/melee_env.py` - Main RL environment
- `ssbm_rl/policies/ppo_policy.py` - PPO implementation  
- `ssbm_rl/utils/console_manager.py` - Dolphin management

### Configuration
- `configs/train_config.yaml` - All training settings
- `train.py` - Generic training script

### Outputs
- `checkpoints/` - Saved models (*.pt files)
- `logs/` - Training logs (*.jsonl files)

## 🎮 Common Tasks

### Change Character
```yaml
# configs/train_config.yaml
player_character: "FOX"  # or FALCO, SHEIK, etc.
```

### Adjust Difficulty
```yaml
cpu_level: 5  # 1-9
```

### Tune Rewards
```yaml
damage_dealt_weight: 0.02  # More emphasis on damage
win_bonus: 10.0            # Bigger win reward
```

### Modify PPO Hyperparameters
```yaml
ppo:
  learning_rate: 0.001     # Higher learning rate
  gamma: 0.95              # Lower discount
  hidden_dim: 256          # Bigger network
```

## 📊 Monitor Training

### View Progress
```bash
# Real-time logs
docker logs -f ssbm-training

# View training stats
cat logs/experiment_*.jsonl | tail -20
```

### Load Checkpoint
```python
from ssbm_rl.policies.ppo_policy import PPOPolicy

policy = PPOPolicy()
policy.load("checkpoints/policy_ep1000.pt")
```

## 🔧 Troubleshooting

### Container won't start
```bash
docker stop ssbm-training
docker rm ssbm-training
```

### Reset everything
```bash
rm -rf checkpoints/* logs/*
./setup.sh
```

## 📚 Module Documentation

### MeleeEnv
```python
env = MeleeEnv(console, player_controller, opponent_controller)
state, reward, done, info = env.step()
```

### StateExtractor
```python
extractor = StateExtractor(state_dim=14, normalize=True)
state = extractor.extract(gamestate, player_port=1, opponent_port=2)
```

### RewardFunction
```python
reward_fn = RewardFunction(
    damage_dealt_weight=0.01,
    stock_won_reward=1.0,
    win_bonus=5.0
)
reward = reward_fn.compute(current_state, previous_state)
```

### PPOPolicy
```python
policy = PPOPolicy(state_dim=14, action_dim=5, hidden_dim=128)
action, log_prob, value = policy.get_action(state)
stats = policy.update([trajectory])
policy.save("checkpoint.pt")
```

## 💡 Tips

1. **Start Small**: Train with random policy first to test setup
2. **Monitor Early**: Check first few episodes look reasonable
3. **Save Often**: Set `save_frequency` to save checkpoints
4. **Tune Rewards**: The reward function is critical for learning
5. **Be Patient**: RL takes time - expect 1000+ episodes

## 🎯 Next Steps

1. Train baseline with random policy
2. Train PPO for 1000 episodes
3. Tune hyperparameters based on logs
4. Expand action space for better control
5. Implement curriculum learning (start vs easy CPU)
