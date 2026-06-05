#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if [[ -x /root/miniconda3/bin/python ]]; then
    echo /root/miniconda3/bin/python
    return
  fi
  echo "No usable Python interpreter found. Expected python3, python, or /root/miniconda3/bin/python." >&2
  exit 1
}

ensure_venv() {
  local system_python
  system_python="$(find_system_python)"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$system_python" -m venv "$VENV_DIR"
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
