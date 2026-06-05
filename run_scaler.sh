#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

run_prepare() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_data.py" "$@"
}

run_train() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_scaler.py" "$@"
}

run_ablation() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_ablation.py" "$@"
}

run_evaluate() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluate_scaler.py" "$@"
}

run_summarize() {
  "$PYTHON_BIN" "$ROOT_DIR/summarize_results.py" "$@"
}

run_smoke() {
  "$PYTHON_BIN" -m pytest "$ROOT_DIR/tests" -q
}

main() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: ./run_scaler.sh {prepare-data|train|ablation|evaluate|summarize|smoke} ..."
    exit 1
  fi
  ensure_venv
  cmd="$1"
  shift
  case "$cmd" in
    prepare-data) run_prepare "$@" ;;
    train) run_train "$@" ;;
    ablation) run_ablation "$@" ;;
    evaluate) run_evaluate "$@" ;;
    summarize) run_summarize "$@" ;;
    smoke) run_smoke "$@" ;;
    *)
      echo "Unknown command: $cmd"
      exit 1
      ;;
  esac
}

main "$@"
