# 001 - Dependency management, tests and CI

## Context

The repository had no dependency file, tests or CI. Several bugs (inverted AlphaZero value
targets, arena results credited to the wrong player, a biased PUCT selection, Rainbow masking,
`NameError`s in the training scripts) had gone unnoticed because nothing exercised the code
outside full training runs. The Kaggle agents must be single, self-contained files, so the game
logic and the pyplAI algorithms are duplicated across `agents/*.py`.

## Decision

- **uv + `pyproject.toml`** (no installable package), Python pinned to 3.12: the oldest version
  supported by numpy 2.5 and Tianshou 2.0, and supported by kaggle-environments and torch.
  - Base dependencies: what the agents, the notebook and AlphaZero training need (numpy, torch,
    kaggle-environments, pyplAI).
  - `train` group: Tianshou, Gymnasium, TensorBoard, tqdm and matplotlib for PPO/Rainbow.
  - `dev` group: pytest and ruff.
- **PyTorch index per platform** via `[tool.uv.sources]`: CUDA 13.0 wheels on Windows, where
  training runs on the GPU, and CPU wheels on Linux, which keeps the CI install small.
- **pyplAI from the GitHub tag `v1.1.0`**: that release fixes the Minimax inefficiencies but is
  not on PyPI (which still has 1.0.3). The Kaggle agents vendor the same version.
- **Parity tests instead of a refactor**: the duplicated code stays in the single-file agents
  (a requirement of Kaggle), and tests check every copy against the reference implementation
  and against the installed pyplAI.
- **CI**: one GitHub Actions job on Ubuntu running `uv sync --locked --all-groups`,
  `ruff check` (rules F, E9 and B, no reformatting) and `pytest`.

## Alternatives considered

- *`requirements.txt` + pip*: no lock file, and no per-platform PyTorch index.
- *Everything in the base dependencies*: simpler, but installing Tianshou is unnecessary to run
  the agents or train AlphaZero.
- *A shared module for the game logic*: removes the duplication but breaks the Kaggle
  single-file submissions.
- *`ruff format` on the whole repository*: a large style-only diff on a project about to be
  archived.
