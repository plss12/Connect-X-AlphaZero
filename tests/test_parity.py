"""Duplicated game logic stays in sync.

Kaggle agents must be single, self-contained files, so the game rules are copied into several
modules. Instead of refactoring them into a shared module (which Kaggle cannot import), these
tests check that every copy agrees with the reference ``env.connectXalphazero.Connect4Game`` on
seeded random positions.
"""

import importlib

import numpy as np
import pytest
from helpers import board_from_moves, random_positions, to_kaggle_board

from env.connectXalphazero import Connect4Game

GAME = Connect4Game()
POSITIONS = random_positions(150, seed=1)
FINISHED_POSITIONS = random_positions(150, seed=2, stop_before_win=False)

WINNING_MOVE_MODULES = ["agents.ppo_agent", "agents.rainbow_agent", "env.connectXgym"]
STATE_MODULES = ["agents.mcts_agent", "agents.minimax_agent", "training_algorithms.pyplAI_algorithms"]


def _import(name: str) -> object:
    if name == "env.connectXgym":
        pytest.importorskip("gymnasium")
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", WINNING_MOVE_MODULES)
def test_check_winning_move_matches_reference(module_name: str) -> None:
    check_winning_move = _import(module_name).check_winning_move
    for moves in POSITIONS:
        board = board_from_moves(moves)
        kaggle_board = np.array(to_kaggle_board(board)).reshape(6, 7)
        for col in np.flatnonzero(GAME.get_valid_moves(board)):
            for player, mark in ((1, 1), (-1, 2)):
                after, _ = GAME.get_next_state(board, player, col)
                expected = GAME.check_win(after, player)
                assert check_winning_move(kaggle_board, col, mark) == expected, (moves, col, mark)


@pytest.mark.parametrize("module_name", STATE_MODULES)
def test_connectx_state_matches_reference(module_name: str) -> None:
    state_cls = _import(module_name).ConnectXState
    for moves in FINISHED_POSITIONS:
        board = board_from_moves(moves)
        mark = 1 if len(moves) % 2 == 0 else 2
        state = state_cls(to_kaggle_board(board), mark)
        assert bool(state.wins_player(1)) == GAME.check_win(board, 1), moves
        assert bool(state.wins_player(2)) == GAME.check_win(board, -1), moves
        assert state.is_final_state() == (GAME.get_game_ended(board, 1) != 0), moves
        assert state.get_moves() == list(np.flatnonzero(GAME.get_valid_moves(board))), moves


@pytest.mark.parametrize("module_name", STATE_MODULES)
def test_connectx_state_apply_move_matches_reference(module_name: str) -> None:
    state_cls = _import(module_name).ConnectXState
    for moves in POSITIONS[:50]:
        board = board_from_moves(moves)
        mark = 1 if len(moves) % 2 == 0 else 2
        player = 1 if mark == 1 else -1
        for col in np.flatnonzero(GAME.get_valid_moves(board)):
            state = state_cls(to_kaggle_board(board), mark).apply_move(int(col))
            expected, _ = GAME.get_next_state(board, player, col)
            assert state.board == to_kaggle_board(expected)
            assert state.jugadorActual == 3 - mark


def test_alphazero_agent_game_matches_reference() -> None:
    agent_game = importlib.import_module("agents.alphazero_agent").Connect4Game()
    for moves in FINISHED_POSITIONS:
        board = board_from_moves(moves)
        for player in (1, -1):
            assert agent_game.get_game_ended(board, player) == GAME.get_game_ended(board, player)
            np.testing.assert_array_equal(
                agent_game.get_canonical_form(board, player),
                GAME.get_canonical_form(board, player),
            )
        np.testing.assert_array_equal(agent_game.get_valid_moves(board), GAME.get_valid_moves(board))
        for col in np.flatnonzero(GAME.get_valid_moves(board)):
            np.testing.assert_array_equal(
                agent_game.get_next_state(board, 1, col)[0], GAME.get_next_state(board, 1, col)[0]
            )
