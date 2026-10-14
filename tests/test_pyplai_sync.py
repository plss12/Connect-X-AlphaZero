"""The Kaggle agents' vendored pyplAI algorithms match the installed pyplAI (v1.1.0)."""

import pyplAI
import pytest
from helpers import board_from_moves, random_positions, to_kaggle_board

import agents.mcts_agent as mcts_agent
import agents.minimax_agent as minimax_agent

POSITIONS = random_positions(12, seed=3)


def _state(module: object, moves: list[int]) -> object:
    mark = 1 if len(moves) % 2 == 0 else 2
    return module.ConnectXState(to_kaggle_board(board_from_moves(moves)), mark)


def _summary(result: object) -> tuple:
    return result.movimiento, [(m.movimiento, m.visitas, m.valor) for m in result.movimientos]


@pytest.mark.parametrize("moves", POSITIONS)
def test_vendored_mcts_matches_pyplai(moves: list[int]) -> None:
    S = mcts_agent.ConnectXState
    rules = (S.apply_move, S.get_moves, S.is_final_state, S.wins_player, 2, 0)
    vendored = mcts_agent.MCTS(*rules, iteraciones=150, semilla=7)
    library = pyplAI.MCTS(*rules, iteraciones=150, semilla=7)
    state = _state(mcts_agent, moves)
    assert _summary(vendored.analiza(state)) == _summary(library.analiza(state))


@pytest.mark.parametrize("moves", POSITIONS)
def test_vendored_minimax_matches_pyplai(moves: list[int]) -> None:
    S = minimax_agent.ConnectXState
    rules = (S.apply_move, S.get_moves, S.is_final_state, S.wins_player, S.heuristic, 2, 3)
    state = _state(minimax_agent, moves)
    vendored = minimax_agent.Minimax(*rules).analiza(state)
    library = pyplAI.Minimax(*rules).analiza(state)
    assert _summary(vendored) == _summary(library)
    assert minimax_agent.Minimax(*rules).ejecuta(state) == pyplAI.Minimax(*rules).ejecuta(state)


def test_minimax_prefers_the_fastest_win() -> None:
    # Player 2 wins now in column 6. pyplAI 1.0.3 scored every win as +inf, so a slower win
    # could tie with it; 1.1.0 scores a win at ply n as VICTORIA - n.
    S = minimax_agent.ConnectXState
    state = _state(minimax_agent, [0, 6, 0, 6, 0, 6, 1, 5, 2])
    solver = minimax_agent.Minimax(
        S.apply_move, S.get_moves, S.is_final_state, S.wins_player, S.heuristic, 2, 3
    )
    best = solver.analiza(state).movimientos[0]
    assert best.movimiento == 6
    assert best.valor == pytest.approx(minimax_agent.VICTORIA - 1)
