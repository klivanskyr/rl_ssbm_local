#!/usr/bin/env python3
"""
Generic training script for SSBM RL

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --policy random --episodes 10
    python train.py --policy ppo --episodes 1000 --save-freq 100
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ssbm_rl.envs import MeleeEnv, StateExtractor, RewardFunction
from ssbm_rl.policies import RandomPolicy
from ssbm_rl.policies.ppo_policy import PPOPolicy
from ssbm_rl.utils import ConsoleManager, MenuNavigator, TrainingLogger
import melee


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_character_enum(char_name):
    """Convert character name string to melee.Character enum"""
    char_map = {
        'FOX': melee.Character.FOX,
        'FALCO': melee.Character.FALCO,
        'MARTH': melee.Character.MARTH,
        'SHEIK': melee.Character.SHEIK,
        'JIGGLYPUFF': melee.Character.JIGGLYPUFF,
        'PEACH': melee.Character.PEACH,
        'CPTFALCON': melee.Character.CPTFALCON,
        'SAMUS': melee.Character.SAMUS,
        'PIKACHU': melee.Character.PIKACHU,
    }
    return char_map.get(char_name.upper(), melee.Character.MARTH)


def get_stage_enum(stage_name):
    """Convert stage name string to melee.Stage enum"""
    stage_map = {
        'YOSHIS_STORY': melee.Stage.YOSHIS_STORY,
        'BATTLEFIELD': melee.Stage.BATTLEFIELD,
        'FINAL_DESTINATION': melee.Stage.FINAL_DESTINATION,
        'FOUNTAIN_OF_DREAMS': melee.Stage.FOUNTAIN_OF_DREAMS,
        'POKEMON_STADIUM': melee.Stage.POKEMON_STADIUM,
    }
    return stage_map.get(stage_name.upper(), melee.Stage.YOSHIS_STORY)


def create_policy(config):
    """Create policy based on configuration"""
    policy_type = config.get('policy_type', 'random').lower()
    
    if policy_type == 'random':
        return RandomPolicy()
    elif policy_type == 'ppo':
        ppo_config = config.get('ppo', {})
        return PPOPolicy(**ppo_config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def train(config):
    """Main training loop"""
    
    print("="*80)
    print("SSBM RL Training")
    print("="*80)
    print(f"Policy: {config['policy_type']}")
    print(f"Episodes: {config['total_episodes']}")
    print(f"Player: {config['player_character']} (Port {config['player_port']})")
    print(f"Opponent: CPU Level {config['cpu_level']} (Port {config['opponent_port']})")
    print("="*80)
    
    # Initialize console manager
    console_mgr = ConsoleManager(
        dolphin_path=config['dolphin_path'],
        iso_path=config['iso_path'],
        save_replays=config['save_replays'],
        enable_logging=config['enable_logging']
    )
    
    console = console_mgr.create_console()
    player_controller = console_mgr.add_controller(config['player_port'])
    opponent_controller = console_mgr.add_controller(config['opponent_port'])
    console_mgr.setup_signal_handler()
    
    # Start console
    console_mgr.start()
    
    # Initialize environment components
    state_extractor = StateExtractor(
        state_dim=config['state_dim'],
        normalize=config['normalize_state']
    )
    
    reward_function = RewardFunction(
        damage_dealt_weight=config['damage_dealt_weight'],
        damage_taken_weight=config['damage_taken_weight'],
        stock_won_reward=config['stock_won_reward'],
        stock_lost_penalty=config['stock_lost_penalty'],
        distance_penalty_weight=config['distance_penalty_weight'],
        win_bonus=config['win_bonus'],
        loss_penalty=config['loss_penalty'],
        player_port=config['player_port'],
        opponent_port=config['opponent_port']
    )
    
    env = MeleeEnv(
        console=console,
        player_controller=player_controller,
        opponent_controller=opponent_controller,
        player_port=config['player_port'],
        opponent_port=config['opponent_port'],
        state_extractor=state_extractor,
        reward_function=reward_function
    )
    
    # Initialize menu navigator
    menu_nav = MenuNavigator(
        player_character=get_character_enum(config['player_character']),
        player_costume=config['player_costume'],
        opponent_character=None,  # Random
        stage=get_stage_enum(config['stage']),
        cpu_level=config['cpu_level']
    )
    
    # Initialize policy
    policy = create_policy(config)
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=config['log_dir'],
        experiment_name=config.get('experiment_name')
    )
    
    # Training loop
    print("\nStarting training...")
    print("="*80)
    
    episode = 0
    frame_count = 0
    in_game = False
    episode_trajectory = []
    
    while episode < config['total_episodes']:
        gamestate = console.step()
        frame_count += 1
        
        if gamestate is None:
            continue
        
        # In-game: collect experience
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            if not in_game:
                in_game = True
                env.reset()
                episode_trajectory = []
                menu_nav.randomize_opponent()
            
            # Get state, action, reward
            state, reward, done, info = env.step()
            
            if state is not None:
                # Get action from policy
                if hasattr(policy, 'get_action'):
                    if config['policy_type'] == 'ppo':
                        action_idx, log_prob, value = policy.get_action(state)
                        episode_trajectory.append((state, action_idx, reward, log_prob, value))
                    else:
                        action = policy.get_action(state)
                        episode_trajectory.append((state, action, reward))
                    
                    # Execute action (simplified - expand for full action space)
                    action_buttons = [
                        melee.Button.BUTTON_A,
                        melee.Button.BUTTON_B,
                        melee.Button.BUTTON_L,
                        melee.Button.BUTTON_R,
                        None  # No action
                    ]
                    
                    if config['policy_type'] == 'ppo':
                        button = action_buttons[action_idx] if action_idx < len(action_buttons) else None
                    else:
                        button = action
                    
                    if button:
                        player_controller.press_button(button)
                    else:
                        player_controller.release_all()
            
            # Check if episode done
            if done:
                in_game = False
                episode += 1
                
                # Update policy
                if config['policy_type'] == 'ppo' and len(episode_trajectory) > 0:
                    stats = policy.update([episode_trajectory])
                else:
                    stats = {}
                
                # Log episode
                p1 = gamestate.players.get(config['player_port'])
                p2 = gamestate.players.get(config['opponent_port'])
                won = p1 and p2 and p1.stock > 0 and p2.stock == 0
                
                episode_stats = {
                    'episode_reward': info.get('episode_reward', 0),
                    'episode_steps': info.get('episode_steps', 0),
                    'won': won,
                    **stats
                }
                
                logger.log_episode(episode, episode_stats)
                
                if episode % config['print_frequency'] == 0:
                    logger.print_training_progress(
                        episode,
                        config['total_episodes'],
                        episode_stats
                    )
                
                # Save checkpoint
                if episode % config['save_frequency'] == 0 and hasattr(policy, 'save'):
                    checkpoint_path = Path(config['checkpoint_dir']) / f"policy_ep{episode}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    policy.save(str(checkpoint_path))
                    print(f"Checkpoint saved: {checkpoint_path}")
        
        else:
            # Menu navigation
            if frame_count % 60 == 0 and not in_game:
                print(f"Navigating menus... (frame {frame_count})")
            
            menu_nav.navigate(
                gamestate,
                player_controller,
                opponent_controller,
                autostart=menu_nav.check_ready(gamestate, config['player_port'], config['opponent_port'])
            )
    
    # Training complete
    print("\n" + "="*80)
    print(f"Training complete! Completed {episode} episodes.")
    print("="*80)
    
    # Save final model
    if hasattr(policy, 'save'):
        final_path = Path(config['checkpoint_dir']) / "policy_final.pt"
        policy.save(str(final_path))
        print(f"Final model saved: {final_path}")
    
    # Print summary
    summary = logger.get_summary_stats()
    print("\nTraining Summary (last 100 episodes):")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    logger.save_summary()
    console_mgr.stop()


def main():
    parser = argparse.ArgumentParser(description='Train SSBM RL agent')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--policy', type=str, choices=['random', 'ppo'],
                        help='Policy type (overrides config)')
    parser.add_argument('--episodes', type=int,
                        help='Number of episodes (overrides config)')
    parser.add_argument('--save-freq', type=int,
                        help='Save frequency (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.policy:
        config['policy_type'] = args.policy
    if args.episodes:
        config['total_episodes'] = args.episodes
    if args.save_freq:
        config['save_frequency'] = args.save_freq
    
    # Run training
    try:
        train(config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
