#!/bin/bash
# Setup script for SSBM RL framework

set -e

echo "======================================"
echo "SSBM RL Framework Setup"
echo "======================================"

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p replays

# Check for Melee ISO
if [ ! -f "Melee.iso" ]; then
    echo "ERROR: Melee.iso not found!"
    echo "Please place your Melee ISO file in this directory as 'Melee.iso'"
    exit 1
fi

echo "Found Melee.iso ✓"

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -f Dockerfile.new -t melee-rl .

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Setup complete! ✓"
    echo "======================================"
    echo ""
    echo "Quick start:"
    echo "  ./run_training.sh random 10      # Random policy, 10 episodes"
    echo "  ./run_training.sh ppo 1000       # PPO policy, 1000 episodes"
    echo ""
else
    echo "ERROR: Docker build failed"
    exit 1
fi
