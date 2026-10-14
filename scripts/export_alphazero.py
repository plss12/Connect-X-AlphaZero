"""Export an AlphaZero training checkpoint to the weights file the Kaggle agent loads.

Training (``alphazero/NNet.py``) saves ``{'state_dict': ...}`` from a ``torch.compile``-d network,
so the keys carry an ``_orig_mod.`` prefix. ``agents/alphazero_agent.load_model`` expects the plain
state dict of its ``Connect4NNet`` at ``models/AlphaZero.pth``. This script unwraps the dict,
strips the prefix and checks that the weights fit the agent's network before writing them.

Usage:
    uv run python scripts/export_alphazero.py alphazero/files/checkpoints/best.pth.tar \
        models/AlphaZero.pth --num-channels 32
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.alphazero_agent import NUM_CHANNELS, Connect4Game, Connect4NNet, dotdict  # noqa: E402

COMPILED_PREFIX = "_orig_mod."


def export_checkpoint(
    source: Path, target: Path, num_channels: int = NUM_CHANNELS
) -> dict[str, torch.Tensor]:
    """Convert a training checkpoint into the agent's plain state dict and save it.

    Args:
        source: Checkpoint written by ``NNetWrapper.save_checkpoint`` (compiled or not). A plain
            state dict is also accepted.
        target: Output path, usually ``models/AlphaZero.pth``. Parent folders are created.
        num_channels: Width of the network; must match the agent's ``NUM_CHANNELS``.

    Returns:
        The exported state dict.

    Raises:
        ValueError: If the weights do not fit ``Connect4NNet(num_channels)``.
    """
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix(COMPILED_PREFIX): value for key, value in state.items()}

    net = Connect4NNet(Connect4Game(), dotdict({"num_channels": num_channels}))
    try:
        net.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(
            f"{source} does not fit the agent network with num_channels={num_channels}: {e}"
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), target)
    return net.state_dict()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="training checkpoint, e.g. best.pth.tar")
    parser.add_argument(
        "target", type=Path, nargs="?", default=Path("models/AlphaZero.pth"),
        help="output weights file (default: models/AlphaZero.pth)",
    )
    parser.add_argument(
        "--num-channels", type=int, default=NUM_CHANNELS,
        help=f"network width used in training (default: {NUM_CHANNELS})",
    )
    args = parser.parse_args(argv)
    state = export_checkpoint(args.source, args.target, args.num_channels)
    print(f"Exported {len(state)} tensors from {args.source} to {args.target}")


if __name__ == "__main__":
    main()
