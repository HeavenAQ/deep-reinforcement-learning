import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def save_model(model: torch.nn.Module, model_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{model_name}_{timestamp}.pth"
    checkpoint_path = PROJECT_ROOT / "models" / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"{checkpoint_name} has been saved to {checkpoint_path}.")


class ReplayBuffer:
    """Fixed-size ring buffer of (obs, action, reward, next_obs, terminated)."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminateds = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, actions, rewards, next_obs, terminateds) -> None:
        idx = (self.pos + np.arange(len(obs))) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.terminateds[idx] = terminateds
        self.pos = (self.pos + len(obs)) % self.capacity
        self.size = min(self.size + len(obs), self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.terminateds[idx],
        )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
