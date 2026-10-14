"""Builders for boards, Kaggle observations and stub networks shared by the tests."""

from collections.abc import Callable, Sequence

import numpy as np
from kaggle_environments.utils import Struct

ROWS, COLS = 6, 7


class dotdict(dict):
    """Same attribute-access dict the training script and the AlphaZero agent use for args."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def board_from_moves(moves: Sequence[int]) -> np.ndarray:
    """Board in AlphaZero format (P1 = 1, P2 = -1) after playing ``moves`` alternately from P1."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    player = 1
    for col in moves:
        row = max(r for r in range(ROWS) if board[r, col] == 0)
        board[row, col] = player
        player = -player
    return board


def to_kaggle_board(board: np.ndarray) -> list[int]:
    """Flat Kaggle board (P1 = 1, P2 = 2, empty = 0) from an AlphaZero-format board."""
    kaggle = np.where(board == -1, 2, board)
    return [int(x) for x in kaggle.flatten()]


def kaggle_obs(moves: Sequence[int]) -> Struct:
    """Kaggle observation for the player to move after ``moves``."""
    return Struct(
        board=to_kaggle_board(board_from_moves(moves)),
        mark=1 if len(moves) % 2 == 0 else 2,
        step=len(moves),
        remainingOverageTime=60,
    )


def kaggle_config(timeout: float = 2.0) -> Struct:
    """Kaggle Connect X configuration with a custom per-move timeout."""
    return Struct(
        rows=ROWS, columns=COLS, inarow=4, timeout=timeout, actTimeout=timeout,
        episodeSteps=1000, agentTimeout=60, runTimeout=1200,
    )


def random_positions(
    count: int, seed: int, stop_before_win: bool = True
) -> list[list[int]]:
    """Move lists of random legal games cut at a random length.

    With ``stop_before_win`` no position contains a finished line, so "does this move win?"
    has a single correct answer for every implementation.
    """
    from env.connectXalphazero import Connect4Game

    game = Connect4Game()
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        board = game.get_init_board()
        player = 1
        moves: list[int] = []
        target = int(rng.integers(0, ROWS * COLS))
        while len(moves) < target:
            valid = np.flatnonzero(game.get_valid_moves(board))
            if len(valid) == 0:
                break
            col = int(rng.choice(valid))
            next_board, next_player = game.get_next_state(board, player, col)
            if stop_before_win and game.check_win(next_board, player):
                break
            moves.append(col)
            board, player = next_board, next_player
        positions.append(moves)
    return positions


class StubNet:
    """Network stand-in for MCTS: fixed prior and value, counts calls.

    It also takes ``(game, args)`` so ``Trainer`` can build its ``pnet`` from the class.
    """

    def __init__(self, game: object = None, args: object = None,
                 prior: Sequence[float] | None = None, value: float = 0.0) -> None:
        self.prior = np.full(COLS, 1 / COLS) if prior is None else np.asarray(prior, float)
        self.value = value
        self.calls = 0

    def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        self.calls += 1
        return self.prior.copy(), self.value


class ScriptedMCTS:
    """MCTS stand-in that returns a one-hot policy chosen by ``policy(canonical_board)``."""

    def __init__(self, policy: Callable[[np.ndarray], int]) -> None:
        self.policy = policy

    def getActionProb(self, canonicalBoard: np.ndarray, temp: int = 1) -> list[float]:
        probs = [0.0] * COLS
        probs[self.policy(canonicalBoard)] = 1.0
        return probs
