"""AlphaZero network and its wrapper (alphazero/NNet.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import board_from_moves, dotdict

from alphazero.NNet import Connect4NNet, NNetWrapper
from env.connectXalphazero import Connect4Game

GAME = Connect4Game()


def _args(**overrides: object) -> dotdict:
    args = dotdict({
        "num_iters": 3, "max_lr": 0.005, "min_lr": 0.0001, "weight_decay": 1e-4, "epochs": 1,
        "batch_size": 4, "cuda": False, "num_channels": 8, "compile": False,
    })
    args.update(overrides)
    return args


def _examples(n: int) -> list[tuple[np.ndarray, list[float], float]]:
    board = board_from_moves([3, 3, 2])
    return [(board, [1 / 7] * 7, 1.0 if i % 2 else -1.0) for i in range(n)]


def test_forward_shapes() -> None:
    net = Connect4NNet(GAME, _args()).eval()
    log_pi, v = net(torch.zeros(5, 3, 6, 7))
    assert log_pi.shape == (5, 7) and v.shape == (5, 1)
    assert torch.allclose(log_pi.exp().sum(dim=1), torch.ones(5))


def test_predict_returns_a_distribution_and_a_bounded_value() -> None:
    pi, v = NNetWrapper(GAME, _args()).predict(board_from_moves([3, 2, 3]))
    assert pi.shape == (7,) and pi.sum() == pytest.approx(1, abs=1e-5)
    assert -1 <= v <= 1


def test_train_updates_the_weights() -> None:
    wrapper = NNetWrapper(GAME, _args())
    before = [p.detach().clone() for p in wrapper.nnet.parameters()]
    wrapper.train(_examples(16), writer=None, iteration=1)
    after = list(wrapper.nnet.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_train_with_a_single_iteration_schedule() -> None:
    """Regression: the cosine schedule divided by ``num_iters - 1``."""
    NNetWrapper(GAME, _args(num_iters=1)).train(_examples(8), writer=None, iteration=1)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    board = board_from_moves([3, 4, 3])
    source = NNetWrapper(GAME, _args())
    source.save_checkpoint(folder=str(tmp_path / "ckpt"), filename="best.pth.tar")
    target = NNetWrapper(GAME, _args())
    target.load_checkpoint(folder=str(tmp_path / "ckpt"), filename="best.pth.tar")
    np.testing.assert_allclose(source.predict(board)[0], target.predict(board)[0], rtol=1e-6)


def test_loading_a_missing_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No model"):
        NNetWrapper(GAME, _args()).load_checkpoint(folder=str(tmp_path), filename="nope.pth.tar")


def test_latest_checkpoint_iteration(tmp_path: Path) -> None:
    wrapper = NNetWrapper(GAME, _args())
    assert wrapper.get_latest_checkpoint_iteration(str(tmp_path / "missing")) == 1
    assert wrapper.get_latest_checkpoint_iteration(str(tmp_path)) == 1
    for name in ("checkpoint_3.examples", "checkpoint_12.examples", "checkpoint_x.examples"):
        (tmp_path / name).touch()
    assert wrapper.get_latest_checkpoint_iteration(str(tmp_path)) == 13
