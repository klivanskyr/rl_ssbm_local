#!/usr/bin/env python3
"""
Parse Slippi replay files (.slp) using peppi-py
"""

import os
import sys
from pathlib import Path
import json
import peppi_py as peppi

def parse_replay(slp_path):
    """Parse a single .slp replay file and extract key information"""
    try:
        game = peppi.read_slippi(slp_path)
        
        # Basic game info
        info = {
            "file": os.path.basename(slp_path),
            "stage": game.start.stage.name if hasattr(game.start, 'stage') else "Unknown",
            "duration_frames": game.metadata.duration if hasattr(game.metadata, 'duration') else 0,
            "duration_seconds": round(game.metadata.duration / 60, 2) if hasattr(game.metadata, 'duration') else 0,
            "players": []
        }
        
        # Player information
        for port, player_data in game.start.players.items():
            if player_data is None:
                continue
                
            player_info = {
                "port": port,
                "character": player_data.character.name if hasattr(player_data, 'character') else "Unknown",
                "costume": player_data.costume if hasattr(player_data, 'costume') else 0,
                "type": player_data.type.name if hasattr(player_data, 'type') else "Unknown",
            }
            
            # Get final stats from last frame if available
            if hasattr(game, 'frames') and len(game.frames) > 0:
                last_frame = game.frames[-1]
                if hasattr(last_frame, 'ports') and port in last_frame.ports:
                    port_data = last_frame.ports[port]
                    if hasattr(port_data, 'leader'):
                        leader = port_data.leader
                        if hasattr(leader, 'post'):
                            player_info["final_stocks"] = leader.post.stocks if hasattr(leader.post, 'stocks') else 0
                            player_info["final_damage"] = round(leader.post.damage, 2) if hasattr(leader.post, 'damage') else 0
            
            info["players"].append(player_info)
        
        # Determine winner (player with stocks remaining)
        if info["players"]:
            stocks = [(p.get("final_stocks", 0), p) for p in info["players"]]
            max_stocks = max(s[0] for s in stocks)
            winner = [p for s, p in stocks if s == max_stocks and s > 0]
            if winner:
                info["winner"] = {
                    "port": winner[0]["port"],
                    "character": winner[0]["character"]
                }
        
        return info
        
    except Exception as e:
        return {
            "file": os.path.basename(slp_path),
            "error": str(e)
        }

def analyze_replays(replay_dir="./replays", output_json=None):
    """Analyze all .slp files in a directory"""
    replay_dir = Path(replay_dir)
    
    if not replay_dir.exists():
        print(f"Error: Directory {replay_dir} does not exist")
        return
    
    slp_files = list(replay_dir.glob("*.slp"))
    
    if not slp_files:
        print(f"No .slp files found in {replay_dir}")
        return
    
    print(f"Found {len(slp_files)} replay file(s)")
    print("-" * 80)
    
    results = []
    
    for slp_file in sorted(slp_files):
        print(f"\nParsing: {slp_file.name}")
        info = parse_replay(str(slp_file))
        results.append(info)
        
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        
        print(f"  Stage: {info['stage']}")
        print(f"  Duration: {info['duration_seconds']}s ({info['duration_frames']} frames)")
        
        for player in info["players"]:
            stocks = player.get("final_stocks", "?")
            damage = player.get("final_damage", "?")
            print(f"  Player {player['port']} ({player['character']}): {stocks} stocks, {damage}% damage")
        
        if "winner" in info:
            print(f"  Winner: Port {info['winner']['port']} ({info['winner']['character']})")
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_games = [r for r in results if "error" not in r]
    
    if valid_games:
        total_duration = sum(r["duration_seconds"] for r in valid_games)
        avg_duration = total_duration / len(valid_games)
        
        print(f"Total games: {len(valid_games)}")
        print(f"Total playtime: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"Average game duration: {avg_duration:.1f} seconds")
        
        # Character usage
        char_usage = {}
        for game in valid_games:
            for player in game["players"]:
                char = player["character"]
                char_usage[char] = char_usage.get(char, 0) + 1
        
        print("\nCharacter usage:")
        for char, count in sorted(char_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {char}: {count} games")
        
        # Win rates by port
        port_wins = {}
        port_games = {}
        for game in valid_games:
            if "winner" in game:
                winner_port = game["winner"]["port"]
                port_wins[winner_port] = port_wins.get(winner_port, 0) + 1
            
            for player in game["players"]:
                port = player["port"]
                port_games[port] = port_games.get(port, 0) + 1
        
        print("\nWin rates by port:")
        for port in sorted(port_games.keys()):
            wins = port_wins.get(port, 0)
            games = port_games[port]
            win_rate = (wins / games * 100) if games > 0 else 0
            print(f"  Port {port}: {wins}/{games} ({win_rate:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Slippi replay files")
    parser.add_argument("--replay-dir", "-d", default="./replays",
                        help="Directory containing .slp files (default: ./replays)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--file", "-f", help="Parse a single .slp file")
    
    args = parser.parse_args()
    
    if args.file:
        # Parse single file
        print(f"Parsing: {args.file}")
        info = parse_replay(args.file)
        print(json.dumps(info, indent=2))
    else:
        # Parse directory
        analyze_replays(args.replay_dir, args.output)