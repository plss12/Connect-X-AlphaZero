"""AlphaZero MCTS: the training copy (alphazero/MCTS.py) and the Kaggle agent copy."""

import importlib

import numpy as np
import pytest
from helpers import StubNet, board_from_moves, dotdict

from env.connectXalphazero import Connect4Game

MCTS_CLASSES = {
    "training": lambda: importlib.import_module("alphazero.MCTS").MCTS,
    "agent": lambda: importlib.import_module("agents.alphazero_agent").MCTS,
}
ARGS = dotdict({"num_mcts_sims": 25, "cpuct": 1.0, "gamma": 0.99})


@pytest.fixture(params=sorted(MCTS_CLASSES))
def mcts_cls(request: pytest.FixtureRequest) -> type:
    return MCTS_CLASSES[request.param]()


def _root_visits(mcts: object, board: np.ndarray) -> list[int]:
    s = board.tobytes()
    return [mcts.Nsa.get((s, a), 0) for a in range(7)]


@pytest.mark.parametrize("favoured", [0, 2, 6])
def test_search_follows_the_prior_of_unvisited_actions(mcts_cls: type, favoured: int) -> None:
    """Regression: unvisited actions used their column index as Q (``Qsa.get((s, a), a)``),
    so the search visited every column right-to-left before trusting the prior. The first
    selection after expanding a node also ignored the prior because sqrt(N(s)) was 0."""
    prior = np.full(7, 0.01)
    prior[favoured] = 0.94
    board = Connect4Game().get_init_board()
    mcts = mcts_cls(Connect4Game(), StubNet(prior=prior), ARGS)
    for _ in range(8):  # 1 expansion + 7 selections
        mcts.search(board)
    visits = _root_visits(mcts, board)
    assert visits[favoured] == 7, visits


def test_takes_an_immediate_win(mcts_cls: type) -> None:
    board = board_from_moves([0, 6, 0, 6, 0, 5])  # player 1 to move with three in column 0
    mcts = mcts_cls(Connect4Game(), StubNet(), ARGS)
    for _ in range(10):
        mcts.search(board)
    visits = _root_visits(mcts, board)
    assert int(np.argmax(visits)) == 0 and sum(visits) == visits[0]


def test_blocks_an_immediate_loss(mcts_cls: type) -> None:
    board = board_from_moves([0, 4, 1, 4, 6, 4])  # opponent threatens column 4
    mcts = mcts_cls(Connect4Game(), StubNet(), ARGS)
    for _ in range(10):
        mcts.search(board)
    assert int(np.argmax(_root_visits(mcts, board))) == 4


def test_check_win_inplace_restores_the_board(mcts_cls: type) -> None:
    board = board_from_moves([0, 6, 0, 6, 0, 5])
    before = board.copy()
    mcts = mcts_cls(Connect4Game(), StubNet(), ARGS)
    assert mcts._check_win_inplace(board, 1, Connect4Game().get_valid_moves(board)) == 0
    assert mcts._check_win_inplace(board, -1, Connect4Game().get_valid_moves(board)) is None
    np.testing.assert_array_equal(board, before)


def test_finished_game_value_is_from_the_previous_player_view(mcts_cls: type) -> None:
    # The opponent (-1 in canonical form) just completed four: good for whoever moved last.
    board = -board_from_moves([0, 1, 0, 1, 0, 1, 0])
    mcts = mcts_cls(Connect4Game(), StubNet(), ARGS)
    assert mcts.search(board) == 1


def test_leaf_returns_the_negated_network_value(mcts_cls: type) -> None:
    board = Connect4Game().get_init_board()
    mcts = mcts_cls(Connect4Game(), StubNet(value=0.3), ARGS)
    assert mcts.search(board) == pytest.approx(-0.3)


def test_invalid_moves_get_no_prior(mcts_cls: type) -> None:
    board = board_from_moves([3] * 6)
    mcts = mcts_cls(Connect4Game(), StubNet(), ARGS)
    mcts.search(board)
    prior = mcts.Ps[board.tobytes()]
    assert prior[3] == 0 and prior.sum() == pytest.approx(1)


def test_training_action_probabilities() -> None:
    from alphazero.MCTS import MCTS

    board = board_from_moves([3] * 6)
    mcts = MCTS(Connect4Game(), StubNet(), ARGS)
    probs = mcts.getActionProb(board, temp=1)
    assert sum(probs) == pytest.approx(1) and probs[3] == 0

    greedy = MCTS(Connect4Game(), StubNet(), ARGS).getActionProb(board, temp=0)
    assert sorted(greedy) == [0] * 6 + [1] and greedy[3] == 0


def test_agent_action_probabilities_respect_the_time_limit() -> None:
    from agents.alphazero_agent import MCTS

    board = board_from_moves([3] * 6)
    probs, sims = MCTS(Connect4Game(), StubNet(), ARGS).getActionProb(
        board, temp=1, time_limit_sec=0.05
    )
    assert sims > 0 and sum(probs) == pytest.approx(1) and probs[3] == 0


def test_agent_action_probabilities_always_return_the_simulation_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``time_limit_sec`` defaulted to None (TypeError on comparison) and the
    no-visits branch returned a bare list instead of ``(probs, sims)``."""
    import agents.alphazero_agent as agent

    monkeypatch.setattr(agent, "TIME_LIMIT", 0.05)
    board = board_from_moves([3] * 6)
    probs, sims = agent.MCTS(Connect4Game(), StubNet(), ARGS).getActionProb(board, temp=1)
    assert sims > 0 and sum(probs) == pytest.approx(1)

    probs, sims = agent.MCTS(Connect4Game(), StubNet(), ARGS).getActionProb(
        board, temp=1, time_limit_sec=-1
    )
    assert sims == 0 and probs[3] == 0 and sum(probs) == pytest.approx(1)
