"""AlphaZero self-play, arena and checkpointing (alphazero/Trainer.py)."""

from pathlib import Path

import numpy as np
import pytest
from helpers import ScriptedMCTS, StubNet, dotdict

from alphazero.Trainer import Trainer
from env.connectXalphazero import Connect4Game

GAME = Connect4Game()


def _args(tmp_path: Path, **overrides: object) -> dotdict:
    args = dotdict({
        "num_iters": 2, "num_eps": 1, "tempThreshold": 4, "updateThreshold": 0.55,
        "maxlenOfQueue": 1000, "numItersForTrainExamplesHistory": 5, "arenaCompare": 2,
        "num_mcts_sims": 4, "cpuct": 1.0, "gamma": 0.99,
        "max_lr": 0.005, "min_lr": 0.0001, "weight_decay": 1e-4, "epochs": 1, "batch_size": 8,
        "cuda": False, "num_channels": 8, "compile": False,
        "load_model": False, "checkpoint_folder": str(tmp_path / "checkpoints"),
        "checkpoint_file": "best.pth.tar",
    })
    args.update(overrides)
    return args


def _cycle(columns: list[int]) -> ScriptedMCTS:
    """Plays the same column sequence for both players, regardless of the board."""
    step = {"n": 0}

    def policy(_: np.ndarray) -> int:
        col = columns[step["n"] % len(columns)]
        step["n"] += 1
        return col

    return ScriptedMCTS(policy)


def _tactical(board: np.ndarray) -> int:
    """Win if possible, else block, else play column 0 (or the first valid column)."""
    valid = np.flatnonzero(GAME.get_valid_moves(board))
    for player in (1, -1):
        for col in valid:
            if GAME.check_win(GAME.get_next_state(board, player, col)[0], player):
                return int(col)
    return 0 if 0 in valid else int(valid[0])


def _always_column_6(board: np.ndarray) -> int:
    valid = np.flatnonzero(GAME.get_valid_moves(board))
    return 6 if 6 in valid else int(valid[0])


def _player_to_move(canonical_board: np.ndarray) -> int:
    # Canonical boards show the player to move as +1; P1 moves when the piece count is even.
    return 1 if np.count_nonzero(canonical_board) % 2 == 0 else -1


@pytest.mark.parametrize(
    ("columns", "winner"),
    [
        ([0, 1], 1),  # P1 completes column 0 on move 7
        ([0, 1, 2, 1, 0, 1, 6, 1], -1),  # P2 completes column 1 on move 8
    ],
)
def test_self_play_labels_each_position_with_its_mover_outcome(
    tmp_path: Path, columns: list[int], winner: int
) -> None:
    """Regression: the result from P1's view was signed with alpha-zero-general's formula
    (which expects it from the next player's view), so P1 wins were stored as losses."""
    trainer = Trainer(GAME, StubNet(), _args(tmp_path))
    trainer.mcts = _cycle(columns)
    examples = trainer.executeEpisode()
    assert examples
    for board, _, value in examples:
        assert value == winner * _player_to_move(board)


def test_arena_credits_the_actual_winner(tmp_path: Path) -> None:
    """Regression: ``r * curPlayer`` credited every decisive game to player 2."""
    trainer = Trainer(GAME, StubNet(), _args(tmp_path, arenaCompare=2))
    strong, weak = ScriptedMCTS(_tactical), ScriptedMCTS(_always_column_6)
    pwins, nwins, draws = trainer.arena(pmcts=weak, nmcts=strong)
    assert (pwins, nwins, draws) == (0, 2, 0)
    pwins, nwins, draws = trainer.arena(pmcts=strong, nmcts=weak)
    assert (pwins, nwins, draws) == (2, 0, 0)


def test_train_examples_round_trip(tmp_path: Path) -> None:
    args = _args(tmp_path)
    trainer = Trainer(GAME, StubNet(), args)
    trainer.trainExamplesHistory = [[(np.zeros((6, 7)), [1 / 7] * 7, 1.0)]]
    trainer.saveTrainExamples(3)
    trainer.saveTrainExamples(4)
    trainer.saveTrainExamples(5)
    assert sorted(p.name for p in Path(args.checkpoint_folder).iterdir()) == [
        "checkpoint_4.examples", "checkpoint_5.examples",
    ]

    restored = Trainer(GAME, StubNet(), args)
    restored.loadTrainExamples()
    assert len(restored.trainExamplesHistory) == 1
    assert restored.trainExamplesHistory[0][0][2] == 1.0


def test_loading_without_example_files_exits_cleanly(tmp_path: Path) -> None:
    """Regression: ``sys`` was not imported, so this raised NameError instead of exiting."""
    args = _args(tmp_path)
    Path(args.checkpoint_folder).mkdir(parents=True)
    with pytest.raises(SystemExit):
        Trainer(GAME, StubNet(), args).loadTrainExamples()


def test_learn_runs_end_to_end_and_resumes_after_the_last_iteration(tmp_path: Path) -> None:
    """Smoke test of the full loop with a tiny real network and no TensorBoard writer.

    Regressions: ``writer=None`` crashed on the 'Arena/Updates' scalar, and examples were saved
    as iteration ``i - 1`` so a resumed run repeated the last finished iteration.
    """
    from alphazero.NNet import NNetWrapper

    args = _args(tmp_path)
    nnet = NNetWrapper(GAME, args)
    trainer = Trainer(GAME, nnet, args, writer=None)
    trainer.learn()

    folder = Path(args.checkpoint_folder)
    assert (folder / "temp.pth.tar").exists()
    assert nnet.get_latest_checkpoint_iteration(str(folder)) == args.num_iters + 1
