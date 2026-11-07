"""Q-learning and DQN on CartPole with a small MLP Q-network.

- `q_learning`: one online TD(0) update per step on the latest transitions,
  bootstrapping from the Q-network itself.
- `dqn`: adds an experience replay buffer and a periodically synced target network.
"""

import copy
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.basics.envs import ENV_ID, VIDEO_DIR, make_vector_env
from rl.basics.stats import TrainingStats
from rl.commons import utils

logger = logging.getLogger(__name__)


@dataclass
class QLearningConfig:
    name: str = "q_learning"
    num_envs: int = 3
    seed: int | None = None
    capture_video: bool = True
    time_steps: int = 100_000
    lr: float = 1e-4
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5  # fraction of time_steps over which eps decays
    log_interval: int = 1_000
    return_window: int = 20


@dataclass
class DQNConfig(QLearningConfig):
    name: str = "dqn"
    time_steps: int = 50_000
    lr: float = 2.5e-4
    buffer_size: int = 10_000
    batch_size: int = 128
    learning_starts: int = 1_000  # steps of pure collection before training
    train_frequency: int = 1  # one gradient update every n steps
    target_update_interval: int = 500  # steps between target network syncs


class QNetwork(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv, n_hidden: int = 128):
        super().__init__()
        n_obs = int(np.array(env.single_observation_space.shape).prod())
        action_space = env.single_action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        n_actions = int(action_space.n)

        self.network = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    return max(end, start - (start - end) * t / duration)


def select_actions(
    q_network: QNetwork, obs: torch.Tensor, eps: float, action_space: gym.Space
) -> np.ndarray:
    """Epsilon-greedy, decided independently for each env."""
    with torch.no_grad():
        greedy_actions = q_network(obs).argmax(dim=1).cpu().numpy()
    explore = np.random.random(len(greedy_actions)) < eps
    return np.where(explore, action_space.sample(), greedy_actions)


def td_loss(
    q_network: QNetwork,
    target_network: QNetwork,
    batch: tuple[torch.Tensor, ...],
    gamma: float,
) -> torch.Tensor:
    obs, actions, rewards, next_obs, terminateds = batch
    q = q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_network(next_obs).max(dim=1).values
        target = rewards + gamma * (1 - terminateds) * next_q
    return nn.functional.mse_loss(q, target)


def train(cfg: QLearningConfig) -> QNetwork:
    device = torch.accelerator.current_accelerator(
        check_available=True
    ) or torch.device("cpu")
    envs = make_vector_env(
        cfg.num_envs, cfg.seed, cfg.capture_video, f"{VIDEO_DIR}/{cfg.name}"
    )
    q_network = QNetwork(envs).to(device)
    optimizer = optim.AdamW(q_network.parameters(), lr=cfg.lr)

    # DQN-only components stay None for plain Q-learning, which bootstraps
    # from the network being trained
    dqn_cfg = cfg if isinstance(cfg, DQNConfig) else None
    replay_buffer = None
    target_network = q_network
    if dqn_cfg:
        target_network = copy.deepcopy(q_network).requires_grad_(False)
        obs_shape = envs.single_observation_space.shape
        assert obs_shape is not None
        replay_buffer = utils.ReplayBuffer(dqn_cfg.buffer_size, obs_shape)

    def to_tensors(*arrays, dtypes) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.as_tensor(x, dtype=dtype, device=device)
            for x, dtype in zip(arrays, dtypes)
        )

    # dtypes of (obs, actions, rewards, next_obs, terminateds)
    batch_dtypes = (
        torch.float32,
        torch.long,
        torch.float32,
        torch.float32,
        torch.float32,
    )

    logger.info(f"Start {cfg.name} | env={ENV_ID} device={device}")
    logger.info(f"Config | {cfg}")
    logger.debug(f"Q network:\n{q_network}")

    stats = TrainingStats(cfg.num_envs, cfg.return_window)
    obs, _ = envs.reset(seed=cfg.seed)
    # with NEXT_STEP autoreset, the step after an episode ends only resets
    # that env, so its transition is not a real one and is skipped
    autoreset = np.zeros(cfg.num_envs, dtype=bool)

    step = 0
    for step in range(1, cfg.time_steps + 1):
        eps = linear_schedule(
            cfg.eps_start,
            cfg.eps_end,
            cfg.exploration_fraction * cfg.time_steps,
            step - 1,
        )
        (obs_t,) = to_tensors(obs, dtypes=(torch.float32,))
        actions = select_actions(q_network, obs_t, eps, envs.action_space)
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        stats.record_episodes(step, infos)

        valid = ~autoreset
        stats.record_rewards(rewards[valid])

        batch = None
        if dqn_cfg and replay_buffer:
            replay_buffer.add_batch(
                obs[valid],
                actions[valid],
                rewards[valid],
                next_obs[valid],
                terminateds[valid],
            )
            if step > dqn_cfg.learning_starts and step % dqn_cfg.train_frequency == 0:
                batch = to_tensors(
                    *replay_buffer.sample(dqn_cfg.batch_size), dtypes=batch_dtypes
                )
        elif valid.any():
            # reuse obs already on the device
            batch = (obs_t[valid],) + to_tensors(
                actions[valid],
                rewards[valid],
                next_obs[valid],
                terminateds[valid],
                dtypes=batch_dtypes[1:],
            )

        if batch is not None:
            loss = td_loss(q_network, target_network, batch, cfg.gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.record_loss(loss.item())

        if dqn_cfg and step % dqn_cfg.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        autoreset = terminateds | truncateds
        obs = next_obs

        if step % cfg.log_interval == 0:
            stats.log(step, cfg.time_steps, eps)

        if stats.solved():
            logger.info(f"Solved at step {step}: last 10 episodes reached 500")
            break

    envs.close()
    stats.summary(cfg.name, step)
    utils.save_model(q_network, cfg.name)
    return q_network


def q_learning():
    utils.setup_logging()
    train(QLearningConfig())


def dqn():
    utils.setup_logging()
    train(DQNConfig())
