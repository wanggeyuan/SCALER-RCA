# Contributing

Thanks for considering a contribution to SCALER-RCA.

## Development Setup

Use Python 3.10 or newer. The helper script creates a local virtual environment automatically:

```bash
./run_scaler.sh smoke
```

For a manually managed environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a pinned environment close to the tested local setup:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## Before Opening a Pull Request

Run the test suite:

```bash
source .venv/bin/activate
python -m pytest tests/ -x -v
```

For changes that affect training, data loading, metrics, or model behavior, also run at least a short training job and include the command and result summary in the pull request.

## Contribution Guidelines

- Keep train, validation, and test data isolated.
- Do not modify labels, result files, or metrics to make a run look better.
- Prefer small, reviewable changes with clear commit messages.
- Add or update tests when changing shared behavior.
- Document new configuration files and command-line behavior.

## Reporting Issues

When reporting a bug, include:

- operating system, Python version, PyTorch version, and device type
- the exact command you ran
- the configuration file used
- the relevant log excerpt or traceback
- whether you are using downloaded RCAEval data or an existing local copy
