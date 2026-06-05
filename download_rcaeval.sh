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
  local venv_args=()
  if [[ "${SCALER_USE_SYSTEM_SITE_PACKAGES:-0}" == "1" ]]; then
    venv_args+=(--system-site-packages)
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    "$system_python" -m venv "${venv_args[@]}" "$VENV_DIR"
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

main() {
  ensure_venv
  "$PYTHON_BIN" "$ROOT_DIR/scripts/download_rcaeval.py" "$@"
}

main "$@"
