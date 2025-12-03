"""
Training logger
"""

import json
import time
from pathlib import Path
import numpy as np


def convert_to_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class TrainingLogger:
    """
    Logger for training statistics and progress
    """
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{int(time.time())}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        
        # Statistics
        self.episode_stats = []
        self.current_episode = 0
    
    def log_episode(self, episode_num, stats):
        """
        Log episode statistics
        
        Args:
            episode_num: Episode number
            stats: Dict of statistics
        """
        log_entry = {
            'episode': episode_num,
            'timestamp': time.time(),
            **stats
        }
        
        self.episode_stats.append(log_entry)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_entry = convert_to_serializable(log_entry)
        
        # Write to file (JSONL format)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(serializable_entry) + '\n')
    
    def print_episode_summary(self, episode_num, stats):
        """Print episode summary to console"""
        print(f"\n{'='*80}")
        print(f"Episode {episode_num} Summary")
        print(f"{'='*80}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*80}\n")
    
    def print_training_progress(self, episode_num, total_episodes, stats):
        """Print compact training progress"""
        progress = episode_num / total_episodes * 100
        print(f"[{progress:5.1f}%] Episode {episode_num}/{total_episodes} | " + 
              " | ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                         for k, v in stats.items()]))
    
    def get_summary_stats(self, last_n=100):
        """
        Get summary statistics over last N episodes
        
        Args:
            last_n: Number of recent episodes to summarize
            
        Returns:
            dict: Summary statistics
        """
        if not self.episode_stats:
            return {}
        
        recent_stats = self.episode_stats[-last_n:]
        
        # Compute averages
        summary = {}
        keys = recent_stats[0].keys()
        
        for key in keys:
            if key in ['episode', 'timestamp']:
                continue
            
            values = [s[key] for s in recent_stats if key in s and isinstance(s[key], (int, float))]
            if values:
                summary[f'avg_{key}'] = sum(values) / len(values)
                summary[f'min_{key}'] = min(values)
                summary[f'max_{key}'] = max(values)
        
        return summary
    
    def save_summary(self):
        """Save summary statistics to JSON file"""
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        summary = self.get_summary_stats(last_n=len(self.episode_stats))
        
        with open(summary_file, 'w') as f:
            json.dumps(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
