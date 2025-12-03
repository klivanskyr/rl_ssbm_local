#!/usr/bin/python3
import melee
import sys
import signal
import argparse
import random
import os

print("Script starting...")
sys.stdout.flush()

parser = argparse.ArgumentParser(description='Headless Melee: Marth Bot vs Level 9 CPU')
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game states',
                    default=False)
parser.add_argument('--dolphin_executable_path', '-e', 
                    default="/opt/slippi-extracted/AppRun",
                    help='The directory where dolphin is')
parser.add_argument('--iso', default="/opt/melee/Melee.iso", type=str,
                    help='Path to melee iso.')

args = parser.parse_args()
print("Arguments parsed")
sys.stdout.flush()

# Logger
log = None
if args.debug:
    log = melee.Logger()

print("Creating console...")
sys.stdout.flush()

# Set Slippi directory to the mounted volume path
# This is where Docker mounts our local ./replays directory
dolphin_home = "/root/.local/share"
slippi_dir = os.path.join(dolphin_home, "Slippi")
os.makedirs(slippi_dir, exist_ok=True)

# Create Console - use dolphin_home_path to control where replays are saved
console = melee.Console(path=args.dolphin_executable_path,
                        dolphin_home_path=dolphin_home,  # Explicitly set Dolphin home directory
                        slippi_address='127.0.0.1',
                        logger=log,
                        fullscreen=False,
                        gfx_backend='Null',
                        disable_audio=True,
                        save_replays=True)

print("Console created")
print(f"Dolphin home path: {console.dolphin_home_path}")
print(f"Replays will be saved to: {slippi_dir} (mounted as ./replays on host)")
sys.stdout.flush()

# Create controller for Player 1 (Marth bot)
controller_1 = melee.Controller(console=console,
                                port=1,
                                type=melee.ControllerType.STANDARD)

# Create a temporary controller for Player 2 to set it up as CPU
controller_2 = melee.Controller(console=console,
                                port=2,
                                type=melee.ControllerType.STANDARD)

print("Controllers created (Player 1 = Marth Bot, Player 2 = Setting up Level 9 CPU)")
sys.stdout.flush()

# Signal handler
def signal_handler(sig, frame):
    print("\n\nShutting down...")
    
    if log:
        log.writelog()
        print("")
        print("Log file created: " + log.filename)
    console.stop()
    print("Shutdown complete!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Running console with ISO...")
sys.stdout.flush()

# Run console
console.run(iso_path=args.iso)

print("Console run() completed")
sys.stdout.flush()

# Connect to console
print("Connecting to console...")
sys.stdout.flush()

if not console.connect():
    print("ERROR: Failed to connect to the console.")
    sys.exit(-1)
print("Console connected")
sys.stdout.flush()

# Connect controller 1 (the bot)
print("Connecting controller 1 (Marth Bot)...")
sys.stdout.flush()

if not controller_1.connect():
    print("ERROR: Failed to connect controller 1.")
    sys.exit(-1)
print("Controller 1 connected")
sys.stdout.flush()

# Connect controller 2 temporarily to set up CPU
print("Connecting controller 2 (for CPU setup)...")
sys.stdout.flush()

if not controller_2.connect():
    print("ERROR: Failed to connect controller 2.")
    sys.exit(-1)
print("Controller 2 connected")
sys.stdout.flush()

costume_1 = 0

# Pick a random character for the CPU
cpu_characters = [
    melee.Character.FOX,
    melee.Character.FALCO,
    melee.Character.MARTH,
    melee.Character.SHEIK,
    melee.Character.JIGGLYPUFF,
    melee.Character.PEACH,
    melee.Character.CPTFALCON,
    melee.Character.PIKACHU,
    melee.Character.SAMUS,
    melee.Character.YOSHI,
    melee.Character.LUIGI,
    melee.Character.GANONDORF,
    melee.Character.DOC,
    melee.Character.MARIO,
    melee.Character.ROY,
    melee.Character.LINK,
    melee.Character.YLINK,
    melee.Character.DK,
    melee.Character.ZELDA
]

cpu_character = random.choice(cpu_characters)
cpu_costume = random.randint(0, 3)

print(f"CPU will be: {cpu_character.name} (costume {cpu_costume})")
sys.stdout.flush()

print("Entering main loop...")
sys.stdout.flush()

# Main loop
frame_count = 0
last_print_frame = 0
game_count = 0
was_in_game = False
game_just_ended = False

while True:
    # Step to next frame
    gamestate = console.step()
    
    frame_count += 1
    
    if gamestate is None:
        continue
    
    # Detect game end by transition from IN_GAME to any menu state
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        was_in_game = True
        game_just_ended = False
        
        # We're in game - print status every 3 seconds
        if frame_count - last_print_frame >= 180:
            last_print_frame = frame_count
            
            p1 = gamestate.players.get(1)
            p2 = gamestate.players.get(2)
            
            if p1 and p2:
                print(f"\n=== Game {game_count + 1} - Frame {frame_count} ({frame_count/60:.1f}s) ===")
                print(f"Player 1 MARTH BOT: {p1.stock} stocks, {p1.percent:.1f}% damage")
                print(f"  Position: ({p1.position.x:.1f}, {p1.position.y:.1f})")
                print(f"  Action: {p1.action.name}")
                print(f"Player 2 CPU Level {p2.cpu_level} ({p2.character.name}): {p2.stock} stocks, {p2.percent:.1f}% damage")
                print(f"  Position: ({p2.position.x:.1f}, {p2.position.y:.1f})")
                print(f"  Action: {p2.action.name}")
                sys.stdout.flush()
                
                # Check if game is ending soon (someone at 0 stocks)
                if p1.stock == 0 or p2.stock == 0:
                    winner = "MARTH BOT (P1)" if p2.stock == 0 else f"CPU Level {p2.cpu_level} (P2)"
                    print(f"\n🎉 GAME {game_count + 1} ENDING! {winner} is winning!")
                    sys.stdout.flush()
        
        # Control the bot (Player 1)
        if 1 in gamestate.players:
            ai_state = gamestate.players[1]
            melee.techskill.multishine(ai_state=ai_state, controller=controller_1)
        
        if log:
            log.logframe(gamestate)
            log.writeframe()
    else:
        # Not in game anymore
        if was_in_game and not game_just_ended:
            # Game just ended!
            game_just_ended = True
            was_in_game = False
            game_count += 1
            
            print(f"\n✅ GAME {game_count} COMPLETE!")
            print(f"Replay should be saved automatically to /root/.local/share/Slippi (mounted as ./replays)")
            sys.stdout.flush()
            
            # Exit after 1 game
            print("\nExiting after 1 game as requested...")
            if log:
                log.writelog()
                print("Log file created: " + log.filename)
            console.stop()
            print("Shutdown complete!")
            sys.exit(0)
        
        # In menus (character select, stage select, etc.)
        if frame_count % 60 == 0:
            print(f"Frame {frame_count}, menu_state: {gamestate.menu_state}")
            if gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                if 2 in gamestate.players:
                    p2 = gamestate.players[2]
                    print(f"  P2: {p2.controller_status}, Level {p2.cpu_level}, Character: {p2.character.name if p2.character else 'None'}")
            sys.stdout.flush()
        
        # Navigate menus - always try to progress
        # Check if both players have valid characters selected
        both_ready = False
        if gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
            if 1 in gamestate.players and 2 in gamestate.players:
                p1 = gamestate.players[1]
                p2 = gamestate.players[2]
                if (p1.character == melee.Character.MARTH and 
                    p2.controller_status == melee.ControllerStatus.CONTROLLER_CPU and 
                    p2.cpu_level == 9 and
                    p2.character is not None and p2.character != melee.Character.UNKNOWN_CHARACTER):
                    both_ready = True
        
        # Player 1 - Marth bot
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller_1,
                                            melee.Character.MARTH,
                                            melee.Stage.YOSHIS_STORY,
                                            "",
                                            costume=costume_1,
                                            autostart=both_ready,
                                            swag=False)
        
        # Player 2 - CPU setup
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller_2,
                                            cpu_character,
                                            melee.Stage.YOSHIS_STORY,
                                            "",
                                            costume=cpu_costume,
                                            autostart=False,
                                            swag=False,
                                            cpu_level=9)
        
        if log:
            log.skipframe()