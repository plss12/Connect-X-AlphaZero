"""Kaggle agents (agents/*.py): legal moves, basic tactics and full games.

The trained weights are not in the repository, so the neural agents are tested with randomly
initialised networks saved where ``load_model`` looks for them (``models/`` in the cwd).
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from helpers import dotdict, kaggle_config, kaggle_obs
from kaggle_environments import make

import agents.alphazero_agent as alphazero
import agents.mcts_agent as mcts
import agents.minimax_agent as minimax
import agents.ppo_agent as ppo
import agents.rainbow_agent as rainbow

WIN_IN_COLUMN_0 = [0, 6, 0, 6, 0, 5]  # player 1 to move with three in column 0
BLOCK_COLUMN_4 = [0, 4, 1, 4, 6, 4]  # player 1 to move, the opponent threatens column 4
POSITIONS = [[], [3], [3, 3, 2], [0] * 6 + [1] * 6, WIN_IN_COLUMN_0]


def _save(model: torch.nn.Module, folder: Path, filename: str) -> None:
    (folder / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), folder / "models" / filename)


@pytest.fixture
def alphazero_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable:
    net = alphazero.Connect4NNet(alphazero.Connect4Game(), dotdict(num_channels=32))
    _save(net, tmp_path, alphazero.MODEL_FILENAME)
    monkeypatch.chdir(tmp_path)
    for name in ("GLOBAL_NET", "GLOBAL_GAME", "GLOBAL_ARGS", "GLOBAL_MCTS"):
        monkeypatch.setattr(alphazero, name, None)
    monkeypatch.setattr(alphazero, "TIME_LIMIT", 0.05)
    return alphazero.alphazero_agent


@pytest.fixture
def ppo_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable:
    _save(ppo.Actor(ppo.FeatureExtractor((3, 6, 7)), 7), tmp_path, ppo.MODEL_FILENAME)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ppo, "TRAINED_MODEL", None)
    return ppo.ppo_agent


@pytest.fixture
def rainbow_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable:
    net = rainbow.RainbowCNN((3, 6, 7), 7, num_atoms=rainbow.NUM_ATOMS)
    _save(net, tmp_path, rainbow.MODEL_FILENAME)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rainbow, "TRAINED_MODEL", None)
    return rainbow.rainbow_agent


@pytest.fixture(params=["alphazero", "ppo", "rainbow", "mcts", "minimax"])
def agent(request: pytest.FixtureRequest) -> Callable:
    if request.param == "mcts":
        return mcts.mcts_agent
    if request.param == "minimax":
        return minimax.minimax_agent
    return request.getfixturevalue(f"{request.param}_agent")


@pytest.mark.parametrize("moves", POSITIONS)
def test_agents_return_a_legal_column(agent: Callable, moves: list[int]) -> None:
    obs = kaggle_obs(moves)
    move = agent(obs, kaggle_config(timeout=0.4))
    assert type(move) is int and obs.board[move] == 0


def test_agents_take_an_immediate_win(agent: Callable) -> None:
    assert agent(kaggle_obs(WIN_IN_COLUMN_0), kaggle_config(timeout=0.4)) == 0


@pytest.mark.parametrize("name", ["alphazero", "ppo", "rainbow", "minimax"])
def test_agents_block_an_immediate_loss(name: str, request: pytest.FixtureRequest) -> None:
    # Pure MCTS only finds the block statistically, so it is not asserted here.
    agent = minimax.minimax_agent if name == "minimax" else request.getfixturevalue(f"{name}_agent")
    assert agent(kaggle_obs(BLOCK_COLUMN_4), kaggle_config(timeout=0.4)) == 4


def test_alphazero_opens_in_the_centre(alphazero_agent: Callable) -> None:
    assert alphazero_agent(kaggle_obs([]), kaggle_config()) == 3


@pytest.mark.parametrize("name", ["alphazero", "ppo", "rainbow"])
def test_neural_agents_finish_a_kaggle_game(name: str, request: pytest.FixtureRequest) -> None:
    agent = request.getfixturevalue(f"{name}_agent")
    env = make("connectx", debug=True)
    env.run([agent, "random"])
    assert [s.status for s in env.state] == ["DONE", "DONE"]


@pytest.mark.parametrize(
    ("module", "fixture_name"), [(ppo, "ppo_agent"), (rainbow, "rainbow_agent")]
)
def test_mismatched_weights_are_rejected(
    module: object, fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Regression: weights were loaded with ``strict=False`` and errors returned None, so a
    wrong file silently produced an agent with random weights (or crashed later)."""
    request.getfixturevalue(fixture_name)
    path = tmp_path / "models" / module.MODEL_FILENAME
    state = torch.load(path)
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_model()


@pytest.mark.parametrize("module", [alphazero, ppo, rainbow])
def test_missing_weights_raise(
    module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        module.load_model()
