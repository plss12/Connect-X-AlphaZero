import gymnasium as gym
import numpy as np
from kaggle_environments import make
import torch

# Adapting ConnectX to gymnasium for Tianshou
class ConnectXGym(gym.Env):
    def __init__(self, switch_prob=0.5, opponent="negamax", apply_symmetry=False):
        self.env = make("connectx", debug=False)
        self.switch_prob = switch_prob
        self.opponent = opponent
        self.apply_symmetry = apply_symmetry

        self.pair = [None, self.opponent]
        self.trainer = self.env.train(self.pair)

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns
        self.center_col = self.columns // 2

        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,self.rows, self.columns), dtype=np.float32)

        self.is_mirrored = False
    
    def set_opponent(self, opponent):
        self.opponent = opponent
        self.pair = [None, self.opponent]
        self.trainer = self.env.train(self.pair)        
    
    def _process_observation(self, observation):
        board = np.array(observation['board']).reshape(self.rows, self.columns)

        if observation.mark == 2:
            new_board = np.copy(board)
            new_board[board==1] = 2
            new_board[board==2] = 1
            board = new_board
        
        if self.is_mirrored:
            board = np.fliplr(board)

        layer_me = (board == 1)
        layer_opponent = (board == 2)
        layer_empty = (board == 0)

        return np.stack([layer_me, layer_opponent, layer_empty]).astype(np.float32)
    
    def _get_action_mask(self, board):
        mask = np.ones(self.columns, dtype=bool)

        for col in range(self.columns):
            if board[0][col] != 0:
                mask[col] = False
        
        if self.is_mirrored:
            mask = mask[::-1]
        
        return mask
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.is_mirrored = (self.apply_symmetry and self.np_random.random() < 0.5)

        self.pair = [None, self.opponent]

        # Use the env RNG (seeded by reset) so that the player order is reproducible
        if self.np_random.random() < self.switch_prob:
            self.pair = self.pair[::-1]
            self.trainer = self.env.train(self.pair)
        else:
            self.trainer = self.env.train(self.pair)

        observation = self.trainer.reset()
        obs_processed = self._process_observation(observation)
        mask = self._get_action_mask(np.array(observation['board']).reshape(self.rows, self.columns))

        return obs_processed, {"action_mask": mask}

    def step(self, action):
        real_action = action
        if self.is_mirrored:
            real_action = self.columns - 1 - action

        observation, reward, done, info = self.trainer.step(int(real_action))
        processed_obs = self._process_observation(observation)
        mask = self._get_action_mask(np.array(observation['board']).reshape(self.rows, self.columns))
        
        if done:
            # Win reward
            if reward == 1:
                reward = 20
            # Loss reward. Kaggle ends the game with reward None after an invalid action
            # (full column), which must not be scored as a draw
            elif reward == -1 or reward is None:
                reward = -20
            # Draw reward
            else:
                reward = 0
        else:
            # Step negative reward
            reward = -1
            # Center column positive reward
            if int(real_action) == self.center_col:
                reward += 1

        if info is None: info = {}
        info["action_mask"] = mask

        return processed_obs, reward, done, False, info

def check_winning_move(board, col, mark):
    rows, columns = board.shape

    empty_rows = np.where(board[:, col] == 0)[0]
    if len(empty_rows) == 0:
        return False
    row = empty_rows[-1]
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 0
        for step in range(-3, 4):
            r_check = row + step * dr
            c_check = col + step * dc
            
            if 0 <= r_check < rows and 0 <= c_check < columns:
                if (r_check == row and c_check == col) or board[r_check][c_check] == mark:
                    count += 1
                    if count >= 4: return True
                else:
                    count = 0
            else:
                count = 0
    return False

def apply_mask_to_logits(logits, mask, device):

    if mask is None:
        return logits
    
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device)
    
    if logits.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    return torch.where(mask, logits, -100)

def apply_mask_to_distribution(probs, mask, device):
    """Masks a C51 value distribution of shape (batch, actions, atoms).

    Masking the logits, as for PPO, leaves invalid actions with a uniform distribution, whose
    expected value is the middle of the support (0 here). That beats every valid action with a
    negative expected return. Instead, all the mass of an invalid action goes to the lowest atom,
    so its Q is v_min and greedy selection does not pick it.
    """
    if mask is None:
        return probs

    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    lowest_atom = torch.zeros_like(probs)
    lowest_atom[..., 0] = 1.0
    return torch.where(mask.unsqueeze(-1), probs, lowest_atom)

def expected_q_values(probs, support, mask=None):
    """Q(s, a) = sum_z p(z | s, a) * z from the network probabilities; invalid actions get -inf."""
    q_values = (probs * support).sum(dim=-1)
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device)
        q_values = q_values.masked_fill(~mask, float('-inf'))
    return q_values