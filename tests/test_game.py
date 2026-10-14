"""Rules of the AlphaZero Connect 4 game (env/connectXalphazero.py)."""

import numpy as np
import pytest
from helpers import board_from_moves

from env.connectXalphazero import Connect4Game


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


def test_initial_board_is_empty(game: Connect4Game) -> None:
    board = game.get_init_board()
    assert board.shape == game.get_board_size() == (6, 7)
    assert not board.any()
    assert game.get_action_size() == 7
    assert game.get_valid_moves(board).tolist() == [1] * 7


def test_pieces_fall_to_the_lowest_empty_row(game: Connect4Game) -> None:
    board = game.get_init_board()
    board, player = game.get_next_state(board, 1, 3)
    board, player = game.get_next_state(board, player, 3)
    assert player == 1
    assert board[5, 3] == 1 and board[4, 3] == -1
    assert np.count_nonzero(board) == 2


def test_next_state_does_not_mutate_the_input(game: Connect4Game) -> None:
    board = board_from_moves([3, 3, 2])
    before = board.copy()
    game.get_next_state(board, 1, 4)
    np.testing.assert_array_equal(board, before)


def test_full_column_is_not_a_valid_move(game: Connect4Game) -> None:
    board = board_from_moves([0] * 6)
    assert game.get_valid_moves(board).tolist() == [0, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "cells",
    [
        [(5, 0), (5, 1), (5, 2), (5, 3)],  # horizontal, left edge
        [(0, 3), (0, 4), (0, 5), (0, 6)],  # horizontal, top-right corner
        [(2, 6), (3, 6), (4, 6), (5, 6)],  # vertical, right edge
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # vertical, top-left corner
        [(2, 0), (3, 1), (4, 2), (5, 3)],  # diagonal "\"
        [(0, 3), (1, 4), (2, 5), (3, 6)],  # diagonal "\" touching the top-right corner
        [(5, 0), (4, 1), (3, 2), (2, 3)],  # anti-diagonal "/"
        [(3, 3), (2, 4), (1, 5), (0, 6)],  # anti-diagonal "/" touching the top-right corner
    ],
)
def test_check_win_detects_every_direction(game: Connect4Game, cells: list) -> None:
    board = np.zeros((6, 7), dtype=np.int8)
    for r, c in cells:
        board[r, c] = -1
    assert game.check_win(board, -1)
    assert not game.check_win(board, 1)


def test_three_in_a_row_or_broken_lines_are_not_wins(game: Connect4Game) -> None:
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, [0, 1, 2, 4]] = 1  # gap in column 3
    board[[3, 4, 5], 6] = 1  # three vertical
    assert not game.check_win(board, 1)


def test_game_ended_values(game: Connect4Game) -> None:
    p1_wins = board_from_moves([0, 1, 0, 1, 0, 1, 0])
    assert game.get_game_ended(p1_wins, 1) == 1
    assert game.get_game_ended(p1_wins, -1) == -1
    assert game.get_game_ended(board_from_moves([0, 1, 0]), 1) == 0


def test_full_board_without_lines_is_a_small_nonzero_draw(game: Connect4Game) -> None:
    # Pairs of columns with alternating colours and one inverted row: no four in any direction.
    pattern = np.array([[1, 1, -1, -1, 1, 1, -1]] * 6, dtype=np.int8)
    pattern[3] *= -1
    assert not game.check_win(pattern, 1) and not game.check_win(pattern, -1)
    result = game.get_game_ended(pattern, 1)
    assert result != 0 and abs(result) < 1e-3


def test_canonical_form_flips_colours_for_player_two(game: Connect4Game) -> None:
    board = board_from_moves([3, 4])
    np.testing.assert_array_equal(game.get_canonical_form(board, 1), board)
    np.testing.assert_array_equal(game.get_canonical_form(board, -1), -board)


def test_symmetries_mirror_board_and_policy(game: Connect4Game) -> None:
    board = board_from_moves([0, 1])
    pi = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0]
    (b0, p0), (b1, p1) = game.get_symmetries(board, pi)
    np.testing.assert_array_equal(b0, board)
    np.testing.assert_array_equal(b1, np.fliplr(board))
    assert list(p1) == pi[::-1]
