# Release Checklist

Use this checklist before publishing a new release.

## Code and Tests

- Run `python -m pytest tests/ -x -v`.
- Run `./run_scaler.sh smoke` from a clean checkout.
- Confirm `README.md` and `README.zh-CN.md` commands match existing files.
- Confirm new configuration files are documented.

## Data and Artifacts

- Confirm no private data is tracked by Git.
- Confirm checkpoints, logs, and generated result files remain under `outputs/`.
- Confirm example metrics are produced by repository commands, not manually edited files.

## Packaging

- Check `pyproject.toml` version.
- Check `CITATION.cff` version and date.
- Check `CHANGELOG.md` for the release entry.
- Create and push a Git tag, for example `v0.1.0`.

## Public Repository

- Confirm `LICENSE`, `CONTRIBUTING.md`, and `SECURITY.md` exist.
- Confirm issue templates and pull request template are present.
- Confirm the default branch is up to date on GitHub.
