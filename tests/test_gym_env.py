"""Gymnasium wrapper used to train PPO and Rainbow (env/connectXgym.py)."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from env.connectXgym import ConnectXGym  # noqa: E402


def _constant(col: int) -> object:
    return lambda observation, configuration: col


def _play(env: ConnectXGym, actions: list[int]) -> tuple:
    result = None
    for action in actions:
        result = env.step(action)
        if result[2]:
            break
    return result


def test_reset_observation_matches_the_declared_space() -> None:
    env = ConnectXGym(opponent="random", switch_prob=0.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 6, 7)
    assert obs.dtype == env.observation_space.dtype == np.float32
    assert obs[2].all() and not obs[:2].any()
    assert info["action_mask"].tolist() == [True] * 7


def test_step_rewards_moves_and_centre_bonus() -> None:
    env = ConnectXGym(opponent=_constant(6), switch_prob=0.0)
    env.reset(seed=0)
    obs, reward, done, truncated, info = env.step(3)
    assert (reward, done, truncated) == (0, False, False)  # -1 per move, +1 for the centre
    assert obs.dtype == np.float32 and obs[0, 5, 3] == 1 and obs[1, 5, 6] == 1
    _, reward, done, _, _ = env.step(0)
    assert (reward, done) == (-1, False)


def test_win_and_loss_rewards() -> None:
    env = ConnectXGym(opponent=_constant(6), switch_prob=0.0)
    env.reset(seed=0)
    assert _play(env, [0, 0, 0, 0])[1:3] == (20, True)

    env = ConnectXGym(opponent=_constant(0), switch_prob=0.0)
    env.reset(seed=0)
    assert _play(env, [6, 5, 4, 6, 5])[1:3] == (-20, True)


def test_invalid_move_is_punished_as_a_loss() -> None:
    """Regression: Kaggle ends the game with reward None on an invalid move, which the wrapper
    turned into 0 (a draw), so playing into a full column was better than losing."""
    env = ConnectXGym(opponent=_constant(0), switch_prob=0.0)
    env.reset(seed=0)
    _, reward, done, _, _ = _play(env, [0, 0, 0, 0])
    assert (reward, done) == (-20, True)


def test_mirrored_episodes_flip_actions_observations_and_masks() -> None:
    env = ConnectXGym(opponent=_constant(6), switch_prob=0.0, apply_symmetry=True)
    seed = next(s for s in range(100) if (env.reset(seed=s), env.is_mirrored)[1])
    env.reset(seed=seed)
    assert env.is_mirrored
    obs, _, _, _, info = env.step(0)  # action 0 is real column 6
    real_board = np.array(env.env.state[0].observation.board).reshape(6, 7)
    assert real_board[5, 6] == 1 and real_board[4, 6] == 2
    assert obs[0, 5, 0] == 1 and obs[1, 4, 0] == 1  # the agent sees it mirrored back
    _, _, _, _, info = _play(env, [0, 0])
    assert info["action_mask"].tolist() == [False] + [True] * 6


def test_player_order_and_mirroring_follow_the_reset_seed() -> None:
    """Regression: the player order used the global ``np.random`` instead of the env RNG."""

    def episode_setups(env: ConnectXGym) -> list[tuple[bool, bool]]:
        setups = []
        for seed in range(20):
            env.reset(seed=seed)
            setups.append((env.pair[0] is None, env.is_mirrored))
        return setups

    first = episode_setups(ConnectXGym(opponent="random", apply_symmetry=True))
    np.random.seed(123)
    second = episode_setups(ConnectXGym(opponent="random", apply_symmetry=True))
    assert first == second
    assert len(set(first)) > 1
