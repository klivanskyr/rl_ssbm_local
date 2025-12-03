#!/bin/bash
# Convenient wrapper for running training

POLICY=${1:-random}
EPISODES=${2:-10}
SAVE_FREQ=${3:-100}

echo "Running training with:"
echo "  Policy: $POLICY"
echo "  Episodes: $EPISODES"
echo "  Save frequency: $SAVE_FREQ"

docker run --rm -it \
    --name ssbm-training \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/ssbm_rl:/opt/melee/ssbm_rl:ro \
    -v $(pwd)/configs:/opt/melee/configs:ro \
    -v $(pwd)/train.py:/opt/melee/train.py:ro \
    -v $(pwd)/checkpoints:/opt/melee/checkpoints:rw \
    -v $(pwd)/logs:/opt/melee/logs:rw \
    melee-rl \
    python3 train.py --policy $POLICY --episodes $EPISODES --save-freq $SAVE_FREQ
