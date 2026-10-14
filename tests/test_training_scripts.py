"""Training scripts import cleanly and their pyplAI opponents return legal moves."""

import importlib

import pytest
from helpers import kaggle_config, kaggle_obs

from training_algorithms import pyplAI_algorithms


@pytest.mark.parametrize(
    "module_name", ["training_algorithms.ppo_algorithm", "training_algorithms.rainbow_algorithm"]
)
def test_training_module_imports(module_name: str) -> None:
    """Regression: ``sys`` was used without being imported, and ``pyplAI_algorithms`` was
    imported as a top-level module, which only worked when run from its own folder."""
    pytest.importorskip("tianshou")
    importlib.import_module(module_name)


def test_alphazero_training_module_imports() -> None:
    pytest.importorskip("tensorboard")
    module = importlib.import_module("training_algorithms.alphazero_algorithm")
    assert module.args.num_iters > 0


@pytest.mark.parametrize(
    "agent",
    [pyplAI_algorithms.minimax_agent, pyplAI_algorithms.mcts_agent,
     pyplAI_algorithms.minimax_lite_agent],
    ids=["minimax", "mcts", "minimax_lite"],
)
@pytest.mark.parametrize("moves", [[], [3, 3, 2], [0, 0, 0, 0, 0, 0]])
def test_pyplai_opponents_return_legal_moves(agent: object, moves: list[int]) -> None:
    """Regression: ``minimax_agent`` referenced an undefined ``total_limit_time``."""
    obs = kaggle_obs(moves)
    move = agent(obs, kaggle_config(timeout=0.4))
    assert isinstance(move, int) and obs.board[move] == 0


@pytest.mark.parametrize(
    "agent", [pyplAI_algorithms.minimax_agent, pyplAI_algorithms.minimax_lite_agent],
    ids=["minimax", "minimax_lite"],
)
def test_pyplai_minimax_wins_and_blocks(agent: object) -> None:
    assert agent(kaggle_obs([0, 6, 0, 6, 0, 5]), kaggle_config(timeout=0.4)) == 0
    assert agent(kaggle_obs([0, 4, 1, 4, 6, 4]), kaggle_config(timeout=0.4)) == 4
