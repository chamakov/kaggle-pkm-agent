#!/bin/bash
echo "Iniciando entrenamiento en Docker..."
docker run --rm -v $(pwd):/workspace cabt-env:latest bash -c "
cd /workspace
pip install --no-cache-dir stable-baselines3 shimmy --quiet
python -u src/rl/train.py
"
