#!/bin/sh
# Launch from any directory; both SQLite and artifacts live next to this script.
set -eu
MLFLOW_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(dirname "$MLFLOW_DIR")
cd "$MLFLOW_DIR"
exec "$PROJECT_DIR/.venv_heart_disease/bin/mlflow" server \
    --backend-store-uri sqlite:///mlruns.db \
    --artifacts-destination "$MLFLOW_DIR/mlartifacts" \
    --serve-artifacts \
    --host 127.0.0.1 \
    --port 5000 \
    --workers 2
