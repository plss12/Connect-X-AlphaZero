import os
import numpy as np
import torch
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from rainbow_algorithm import ConnectXGym, RainbowCNN

# Loading weights from a checkpoint ignoring extra prefixes and unnecessary information
def load_model_checkpoint(net, model_path, device):

    print(f"Loading weights from: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    expected_keys = set(net.state_dict().keys())
    
    for key, value in state_dict.items():
        new_key = key
        
        if "_optimizers" in key or "model_old" in key:
            continue
            
        if new_key.startswith("policy.model."):
            new_key = new_key.replace("policy.model.", "")
        elif new_key.startswith("model."):
            new_key = new_key.replace("model.", "")
            
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "")
            
        if new_key in expected_keys:
            new_state_dict[new_key] = value
        else:
            pass
        
    net.load_state_dict(new_state_dict, strict=True)
    return net

def evaluate_model(model_path, role, opponent='negamax', n_games=10):

    print(f"\n\nEvaluating agent as {role} against {opponent} with {n_games} games...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    if role=='P1':
        env = DummyVectorEnv([lambda: ConnectXGym(opponent=opponent, apply_symmetry=False, switch_prob=0)])
    else:
        env = DummyVectorEnv([lambda: ConnectXGym(opponent=opponent, apply_symmetry=False, switch_prob=0)])

    net = RainbowCNN(state_shape=(3, 6, 7), action_shape=7, device=device).to(device)

    try:
        net = load_model_checkpoint(net, model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    policy = DiscreteQLearningPolicy(model=net, action_space=env.action_space[0], eps_training=0.0, eps_inference=0.0).to(device)
    policy.eval()

    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=n_games, reset_before_collect=True)

    rewards = result.returns
    
    wins = np.sum(rewards > 50)
    losses = np.sum(rewards < 0)
    draws = len(rewards) - wins - losses

    print(f"\nWins {wins} | Losses {losses} | Draws {draws}")

if __name__ == "__main__":
    model_path = os.path.join("logs_rainbow_self_play", "best_rainbow_agent.pth")

    evaluate_model(model_path, role="P1", opponent="negamax", n_games=10)
    evaluate_model(model_path, role="P2", opponent="negamax", n_games=10)