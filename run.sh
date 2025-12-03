#!/bin/bash

# Stop any existing containers
docker stop melee-bot 2>/dev/null
docker rm melee-bot 2>/dev/null

# Create replays directory
mkdir -p ./replays

# Run the container WITHOUT GPU
docker run \
    --name melee-bot \
    -v $(pwd)/Melee.iso:/opt/melee/Melee.iso:ro \
    -v $(pwd)/main.py:/opt/melee/main.py:ro \
    -v $(pwd)/replays:/root/.local/share/Slippi:rw \
    melee-bot \
    /bin/bash -c "cd /opt/melee && python3 ./main.py"

echo "Container started (CPU-only). Replays will be saved to ./replays/"
echo "View logs with: docker logs -f melee-bot"
echo "Stop with: docker stop melee-bot"