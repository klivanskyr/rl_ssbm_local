#!/usr/bin/python3
"""
PPO Training Script for Marth Bot vs CPU Level 9

This script trains a bot using Proximal Policy Optimization (PPO) by:
1. Running live games in Dolphin emulator
2. Collecting state/action/reward trajectories each frame
3. Training the policy network to maximize cumulative reward
"""

import melee
import sys
import signal
import argparse
import random
import numpy as np
from collections import deque

print("PPO Training Script Starting...")
sys.stdout.flush()

parser = argparse.ArgumentParser(description='PPO Training: Marth Bot vs Level 9 CPU')
parser.add_argument('--dolphin_executable_path', '-e', 
                    default="/opt/slippi-extracted/AppRun",
                    help='Path to Dolphin executable')
parser.add_argument('--iso', default="/opt/melee/Melee.iso", type=str,
                    help='Path to melee iso')
parser.add_argument('--episodes', type=int, default=100,
                    help='Number of episodes to train')
parser.add_argument('--max_steps', type=int, default=3600,
                    help='Max steps per episode (60 steps = 1 second)')

args = parser.parse_args()
print("Arguments parsed")
sys.stdout.flush()

# Environment wrapper for RL
class MeleeEnv:
    """Environment wrapper for SSBM with libmelee"""
    
    def __init__(self, console, controller_bot, controller_cpu):
        self.console = console
        self.controller_bot = controller_bot
        self.controller_cpu = controller_cpu
        self.prev_state = None
        
    def get_state(self, gamestate):
        """Extract state features from gamestate for policy input"""
        if not gamestate or gamestate.menu_state != melee.Menu.IN_GAME:
            return None
            
        p1 = gamestate.players.get(1)  # Our bot
        p2 = gamestate.players.get(2)  # CPU opponent
        
        if not p1 or not p2:
            return None
        
        # State representation (normalized to roughly [-1, 1])
        state = np.array([
            # Player 1 (bot) state
            p1.position.x / 100.0,
            p1.position.y / 100.0,
            p1.percent / 200.0,
            p1.stock / 4.0,
            p1.facing == 1,  # Boolean: facing right
            p1.action.value / 400.0,  # Action ID normalized
            
            # Player 2 (opponent) state
            p2.position.x / 100.0,
            p2.position.y / 100.0,
            p2.percent / 200.0,
            p2.stock / 4.0,
            p2.facing == 1,
            p2.action.value / 400.0,
            
            # Relative state
            (p1.position.x - p2.position.x) / 100.0,
            (p1.position.y - p2.position.y) / 100.0,
        ], dtype=np.float32)
        
        return state
    
    def compute_reward(self, gamestate, prev_gamestate):
        """Compute reward for current transition"""
        if not gamestate or not prev_gamestate:
            return 0.0
            
        if gamestate.menu_state != melee.Menu.IN_GAME:
            return 0.0
            
        p1 = gamestate.players.get(1)
        p2 = gamestate.players.get(2)
        p1_prev = prev_gamestate.players.get(1)
        p2_prev = prev_gamestate.players.get(2)
        
        if not all([p1, p2, p1_prev, p2_prev]):
            return 0.0
        
        reward = 0.0
        
        # Reward for damaging opponent
        damage_dealt = p2.percent - p2_prev.percent
        reward += damage_dealt * 0.01
        
        # Reward for taking opponent's stock
        if p2.stock < p2_prev.stock:
            reward += 1.0
        
        # Penalty for taking damage
        damage_taken = p1.percent - p1_prev.percent
        reward -= damage_taken * 0.01
        
        # Penalty for losing stock
        if p1.stock < p1_prev.stock:
            reward -= 1.0
        
        # Small penalty for distance (encourage engagement)
        distance = abs(p1.position.x - p2.position.x)
        reward -= distance * 0.0001
        
        # Bonus for winning
        if p2.stock == 0 and p1.stock > 0:
            reward += 5.0
        
        # Penalty for losing
        if p1.stock == 0 and p2.stock > 0:
            reward -= 5.0
        
        return reward
    
    def step(self):
        """Step environment and return (state, reward, done)"""
        gamestate = self.console.step()
        
        if gamestate is None:
            return None, 0.0, False
        
        state = self.get_state(gamestate)
        reward = self.compute_reward(gamestate, self.prev_state) if self.prev_state else 0.0
        
        # Episode ends when someone loses all stocks
        done = False
        if gamestate.menu_state == melee.Menu.IN_GAME:
            p1 = gamestate.players.get(1)
            p2 = gamestate.players.get(2)
            if p1 and p2:
                done = (p1.stock == 0 or p2.stock == 0)
        
        self.prev_state = gamestate
        return state, reward, done


# Simple random policy (placeholder - replace with PPO later)
class RandomPolicy:
    """Random policy for baseline testing"""
    
    def __init__(self):
        self.actions = [
            melee.Button.BUTTON_A,
            melee.Button.BUTTON_B,
            melee.Button.BUTTON_L,
            None  # No button
        ]
    
    def get_action(self, state):
        """Sample random action"""
        return random.choice(self.actions)
    
    def update(self, trajectory):
        """Update policy (placeholder for PPO)"""
        pass


print("Creating console...")
sys.stdout.flush()

# Create Console - no replay saving needed for RL
console = melee.Console(path=args.dolphin_executable_path,
                        slippi_address='127.0.0.1',
                        fullscreen=False,
                        gfx_backend='Null',
                        disable_audio=True,
                        save_replays=False)  # No replays for RL

print("Console created")
sys.stdout.flush()

# Create controllers
controller_bot = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
controller_cpu = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)

print("Controllers created")
sys.stdout.flush()

# Signal handler
def signal_handler(sig, frame):
    print("\n\nShutting down...")
    console.stop()
    print("Shutdown complete!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Run console
console.run(iso_path=args.iso)
print("Console running")
sys.stdout.flush()

# Connect
if not console.connect():
    print("ERROR: Failed to connect to console")
    sys.exit(-1)
if not controller_bot.connect():
    print("ERROR: Failed to connect controller 1")
    sys.exit(-1)
if not controller_cpu.connect():
    print("ERROR: Failed to connect controller 2")
    sys.exit(-1)

print("All controllers connected")
sys.stdout.flush()

# Create environment and policy
env = MeleeEnv(console, controller_bot, controller_cpu)
policy = RandomPolicy()  # Replace with PPO later

# Training loop
print(f"\nStarting training for {args.episodes} episodes...")
print("=" * 80)
sys.stdout.flush()

episode = 0
frame_count = 0
cpu_character = random.choice([
    melee.Character.FOX, melee.Character.FALCO, melee.Character.MARTH,
    melee.Character.SHEIK, melee.Character.JIGGLYPUFF
])

in_game = False
episode_reward = 0.0
episode_trajectory = []  # Store (state, action, reward) tuples

while episode < args.episodes:
    gamestate = console.step()
    frame_count += 1
    
    if gamestate is None:
        continue
    
    # In-game: collect experience
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if not in_game:
            in_game = True
            episode_reward = 0.0
            episode_trajectory = []
            print(f"\nEpisode {episode + 1} started")
            sys.stdout.flush()
        
        # Get state
        state = env.get_state(gamestate)
        if state is None:
            continue
        
        # Get action from policy
        action = policy.get_action(state)
        
        # Execute action
        if action == melee.Button.BUTTON_A:
            controller_bot.press_button(melee.Button.BUTTON_A)
        elif action == melee.Button.BUTTON_B:
            controller_bot.press_button(melee.Button.BUTTON_B)
        elif action == melee.Button.BUTTON_L:
            controller_bot.press_button(melee.Button.BUTTON_L)
        else:
            controller_bot.release_all()
        
        # Get reward
        _, reward, done = env.step()
        episode_reward += reward
        
        # Store transition
        episode_trajectory.append((state, action, reward))
        
        # Print status every 3 seconds
        if frame_count % 180 == 0:
            p1 = gamestate.players.get(1)
            p2 = gamestate.players.get(2)
            if p1 and p2:
                print(f"  Frame {frame_count} | Reward: {episode_reward:.2f} | "
                      f"P1: {p1.stock}stocks {p1.percent:.0f}% | "
                      f"P2: {p2.stock}stocks {p2.percent:.0f}%")
                sys.stdout.flush()
        
        # Check if episode done
        if done:
            in_game = False
            episode += 1
            
            # Update policy with collected trajectory
            policy.update(episode_trajectory)
            
            p1 = gamestate.players.get(1)
            p2 = gamestate.players.get(2)
            won = p1.stock > 0 and p2.stock == 0
            
            print(f"\n{'✅ WON' if won else '❌ LOST'} Episode {episode} | "
                  f"Reward: {episode_reward:.2f} | Steps: {len(episode_trajectory)}")
            print("=" * 80)
            sys.stdout.flush()
            
            # Pick new opponent
            cpu_character = random.choice([
                melee.Character.FOX, melee.Character.FALCO, melee.Character.MARTH,
                melee.Character.SHEIK, melee.Character.JIGGLYPUFF
            ])
    
    else:
        # Menu navigation
        if frame_count % 60 == 0:
            print(f"Frame {frame_count}, menu: {gamestate.menu_state}")
            sys.stdout.flush()
        
        # Navigate to character select and start game
        melee.MenuHelper.menu_helper_simple(
            gamestate, controller_bot,
            melee.Character.MARTH,
            melee.Stage.YOSHIS_STORY,
            "", autostart=True, swag=False
        )
        
        melee.MenuHelper.menu_helper_simple(
            gamestate, controller_cpu,
            cpu_character,
            melee.Stage.YOSHIS_STORY,
            "", autostart=False, swag=False, cpu_level=9
        )

print(f"\nTraining complete! Ran {episode} episodes.")
console.stop()
