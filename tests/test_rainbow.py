"""Rainbow (C51) action masking and greedy action selection.

Regressions: the network already returns probabilities but the agents applied a second softmax,
and invalid actions were masked on the logits, which leaves them with a uniform distribution
(Q = 0 for the symmetric support) that beats any valid action with a negative expected return.
"""

import importlib

import numpy as np
import pytest
import torch
from helpers import kaggle_config, to_kaggle_board
from kaggle_environments.utils import Struct

V_MIN, V_MAX, ATOMS = -30, 30, 101
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOMS)


def _module(name: str) -> object:
    if name == "training":
        pytest.importorskip("tianshou")
        return importlib.import_module("training_algorithms.rainbow_algorithm")
    return importlib.import_module("agents.rainbow_agent")


def _distribution_helpers(name: str) -> object:
    if name == "training":
        pytest.importorskip("gymnasium")
        return importlib.import_module("env.connectXgym")
    return importlib.import_module("agents.rainbow_agent")


def _one_hot_distribution(values: list[float]) -> torch.Tensor:
    """(1, 7, ATOMS) distribution with all the mass of action a on the atom closest to values[a]."""
    probs = torch.zeros(1, len(values), ATOMS)
    for a, value in enumerate(values):
        probs[0, a, int(torch.argmin((SUPPORT - value).abs()))] = 1.0
    return probs


@pytest.mark.parametrize("name", ["agent", "training"])
def test_masked_actions_get_the_lowest_value(name: str) -> None:
    net = _module(name).RainbowCNN((3, 6, 7), 7, num_atoms=ATOMS).eval()
    mask = torch.tensor([[True, True, True, False, True, True, False]])
    with torch.no_grad():
        probs, _ = net(torch.rand(1, 3, 6, 7), info={"action_mask": mask})
    assert torch.allclose(probs.sum(dim=2), torch.ones(1, 7))
    q = (probs * SUPPORT).sum(dim=2)[0]
    assert q[3] == V_MIN and q[6] == V_MIN
    assert (q[mask[0]] > V_MIN).all()


@pytest.mark.parametrize("name", ["agent", "training"])
def test_mask_accepts_numpy_batches(name: str) -> None:
    """Tianshou passes the env info masks as a numpy array of shape (batch, actions)."""
    net = _module(name).RainbowCNN((3, 6, 7), 7, num_atoms=ATOMS).eval()
    mask = np.ones((2, 7), dtype=bool)
    mask[1, 0] = False
    with torch.no_grad():
        probs, _ = net(np.random.rand(2, 3, 6, 7).astype(np.float32), info={"action_mask": mask})
    assert probs[1, 0, 0] == 1 and probs[0, 0, 0] < 1


@pytest.mark.parametrize("name", ["agent", "training"])
def test_expected_q_values_use_the_probabilities_directly(name: str) -> None:
    expected_q_values = _distribution_helpers(name).expected_q_values
    probs = _one_hot_distribution([-6, -12, 0, 18, -24, 6, 12])
    mask = torch.tensor([[True, True, True, False, True, True, True]])
    q = expected_q_values(probs, SUPPORT, mask)[0]
    assert q.tolist() == pytest.approx([-6, -12, 0, float("-inf"), -24, 6, 12], abs=1e-5)


class _FakeModel:
    """Returns a fixed distribution and ignores the mask, as an unmasked network would."""

    def __init__(self, probs: torch.Tensor) -> None:
        self.probs = probs

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, obs: torch.Tensor, state: object = None, info: object = None) -> tuple:
        return self.probs, state


def _board_with_two_free_columns() -> Struct:
    # Columns 0-4 come from a full board with no four in a row; columns 5 and 6 are empty.
    board = np.array([[1, 1, -1, -1, 1, 0, 0]] * 6, dtype=np.int8)
    board[3] *= -1
    return Struct(board=to_kaggle_board(board), mark=1, step=30, remainingOverageTime=60)


@pytest.mark.parametrize("name", ["agent", "training"])
def test_agent_picks_the_best_valid_column(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Full columns look attractive (Q = 0) if they are not masked; valid ones are negative.
    probs = _one_hot_distribution([0, 0, 0, 0, 0, -6, -12])
    module = _module(name)
    if name == "agent":
        monkeypatch.setattr(module, "TRAINED_MODEL", _FakeModel(probs))
        agent = module.rainbow_agent
    else:
        agent = module.RainbowAgent(
            _FakeModel(probs), num_atoms=ATOMS, v_min=V_MIN, v_max=V_MAX, device="cpu"
        )
    for _ in range(5):  # the old fallback picked a random valid column
        assert agent(_board_with_two_free_columns(), kaggle_config()) == 5
