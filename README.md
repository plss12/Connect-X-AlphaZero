# Connect-X-AlphaZero

[![CI](https://github.com/plss12/Connect-X-AlphaZero/actions/workflows/ci.yml/badge.svg)](https://github.com/plss12/Connect-X-AlphaZero/actions/workflows/ci.yml)

This repository presents a comprehensive suite of Artificial Intelligence agents designed to master the game of *Connect 4*. The project was developed within the context of *Kaggle's Connect X* competition, aiming to study, implement, and benchmark different families of algorithms, ranging from classical search methods to state-of-the-art Deep Reinforcement Learning.

All developed agents were submitted to the official Kaggle competition, yielding revealing insights into the effectiveness of each approach. Most notably, AlphaZero's performance stands out, achieving a ranking within the global top 10. Furthermore, the implementations of classical search algorithms, MCTS and Minimax, demonstrated surprising effectiveness, significantly outperforming modern RL architectures like PPO and Rainbow, and securing solid positions on the global leaderboard.

The classical search algorithms, MCTS and Minimax, leverage *pyplai*, a library I previously developed, which is available on my *GitHub* repository and installable via *pip*. These agents were integrated with custom heuristics for *Connect 4*, demonstrating that in perfect information games with manageable state spaces, raw computational power and strategic planning offer spectacular performance, proving extremely competitive without the need for neural networks.

On the other hand, the exploration of model-free Deep Reinforcement Learning was conducted using the PPO and Rainbow algorithms. For Rainbow, an advanced variant of DQN, enhancements such as Double Q-Learning, Dueling Networks, Multi-Step Learning, Noisy Nets, Distributional Deep Reinforcement Learning, and Prioritized Experience Replay were included, utilizing the *Tianshou* framework for implementation. While the results were acceptable, training times proved excessively long relative to their final competitive performance. This highlighted a key limitation: as reactive algorithms that rely on "intuition" without explicit future planning, they remain vulnerable to precise long-term tactics, where a single error leads to defeat.

Finally, the AlphaZero implementation was built entirely from scratch to ensure a deep understanding and optimization of its inner workings. This approach bridges the best of both previous worlds: it uses a deep neural network to generate the "intuition" characteristic of Reinforcement Learning, while employing MCTS to validate and plan those moves. This synergy makes this agent the undisputed champion of the repository, significantly outperforming the other implemented algorithms and competing at the highest level against the most complex opponents in the competition.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`).

```bash
uv sync                    # agents, AlphaZero training and the notebook
uv sync --group train      # adds Tianshou, Gymnasium and TensorBoard for PPO/Rainbow training
uv sync --all-groups       # everything, including pytest and ruff
```

PyTorch is resolved per platform in `pyproject.toml`: CUDA 13.0 wheels on Windows (GPU training) and CPU wheels on Linux (CI). To use another CUDA version, change the `pytorch-cu130` index URL and run `uv lock`.

pyplAI is installed from its [v1.1.0 tag on GitHub](https://github.com/plss12/pyplAI/releases/tag/v1.1.0). The Minimax and MCTS Kaggle agents carry a vendored copy of it, because Kaggle submissions cannot install packages.

## Training and agents

```bash
uv run python training_algorithms/alphazero_algorithm.py
uv run --group train python training_algorithms/ppo_algorithm.py
uv run --group train python training_algorithms/rainbow_algorithm.py
```

The neural agents load their weights from `models/` (`AlphaZero.pth`, `PPO.pth`, `Rainbow.pth`), which is not versioned. AlphaZero training writes `{'state_dict': ...}` checkpoints from a `torch.compile`-d network; export the best one to the format the agent loads with:

```bash
uv run python scripts/export_alphazero.py alphazero/files/checkpoints/best.pth.tar models/AlphaZero.pth
```

Set `'compile': False` in the AlphaZero hyperparameters on machines without a C++ compiler.

## Tests and lint

```bash
uv run pytest
uv run ruff check .
```

The tests cover the game rules, the AlphaZero MCTS, self-play labels, arena and checkpoints, the Gymnasium wrapper, Rainbow masking and every Kaggle agent (with randomly initialised networks, since the trained weights are not in the repository). The game logic is duplicated across the single-file Kaggle agents, so parity tests check that every copy agrees with `env/connectXalphazero.py` and with the installed pyplAI. CI runs both commands on every push to `main` and on pull requests.
