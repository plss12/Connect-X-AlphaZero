"""Export of AlphaZero training checkpoints to the Kaggle agent format."""

from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import board_from_moves, dotdict

import agents.alphazero_agent as agent
from alphazero.NNet import NNetWrapper
from env.connectXalphazero import Connect4Game
from scripts.export_alphazero import COMPILED_PREFIX, export_checkpoint, main


def _trained_checkpoint(tmp_path: Path, compiled: bool) -> tuple[NNetWrapper, Path]:
    wrapper = NNetWrapper(
        Connect4Game(), dotdict({"num_channels": agent.NUM_CHANNELS, "cuda": False, "compile": False})
    )
    path = tmp_path / "checkpoints" / "best.pth.tar"
    wrapper.save_checkpoint(folder=str(path.parent), filename=path.name)
    if compiled:  # what torch.compile-d training writes, without needing a C++ compiler here
        state = torch.load(path)["state_dict"]
        torch.save({"state_dict": {COMPILED_PREFIX + k: v for k, v in state.items()}}, path)
    return wrapper, path


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_exported_weights_load_in_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compiled: bool
) -> None:
    wrapper, source = _trained_checkpoint(tmp_path, compiled)
    target = tmp_path / "models" / agent.MODEL_FILENAME
    export_checkpoint(source, target)

    exported = torch.load(target)
    for key, value in wrapper.nnet.state_dict().items():
        assert torch.equal(exported[key], value)

    monkeypatch.chdir(tmp_path)
    net, _, _ = agent.load_model()
    board = board_from_moves([3, 3, 2, 4])
    # The agent quantizes its linear layers to int8, so outputs only match approximately.
    np.testing.assert_allclose(net.predict(board)[0], wrapper.predict(board)[0], atol=0.05)


def test_wrong_network_width_is_rejected(tmp_path: Path) -> None:
    _, source = _trained_checkpoint(tmp_path, compiled=False)
    with pytest.raises(ValueError, match="num_channels=16"):
        export_checkpoint(source, tmp_path / "out.pth", num_channels=16)
    assert not (tmp_path / "out.pth").exists()


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, source = _trained_checkpoint(tmp_path, compiled=True)
    target = tmp_path / "nested" / "AlphaZero.pth"
    main([str(source), str(target)])
    assert target.exists()
    assert "Exported" in capsys.readouterr().out
