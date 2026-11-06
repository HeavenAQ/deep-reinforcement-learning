import logging
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo

from rl.commons import utils

ENV_ID = "CartPole-v1"
VIDEO_DIR = "videos"
VIDEO_STEP_INTERVAL = 1_000

logger = logging.getLogger(__name__)


def create_single_env(recording: bool):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if recording:
        env = gym.wrappers.RecordVideo(env, VIDEO_DIR)
    obs, info = env.reset()
    return env, obs, info


def make_env(capture_video: bool = False, seed: int | None = None):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            VIDEO_DIR,
            # start a new recording every VIDEO_STEP_INTERVAL env steps
            step_trigger=lambda step: step % VIDEO_STEP_INTERVAL == 0,
        )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def create_multi_envs(n_vecs: int):
    return gym.vector.SyncVectorEnv([make_env for _ in range(n_vecs)])


def single_env():
    recording = True
    env, obs, info = create_single_env(recording)

    print("obs:", obs)
    print("info:", info)

    if recording and isinstance(env, RecordVideo):
        print("Video folder:", env.video_folder)
        print("Recording status:", env.recording)

    G = 0
    # terminated = one episode done
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        G += reward

    env.close()
    print(G)


def multi_env():
    envs = create_multi_envs(3)
    envs.reset()

    terminated, truncated = False, False
    while not (terminated or truncated):
        actions = envs.action_space.sample()
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        # stop when the first environment terminates
        # auto-reset is called for the other environment that terminates earlier than the first one
        truncated = truncateds[0]
        terminated = terminateds[0]
        print(next_obs, rewards, terminateds, truncateds, infos)

    envs.close()


class QNetwork(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv, n_hidden: int = 128):
        super().__init__()
        n_obs = int(np.array(env.single_observation_space.shape).prod())
        action_space = env.single_action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        n_actions = action_space.n

        self.network = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def linear_schedule(start: float, end: float, duration: float, t: int) -> float:
    return max(end, start - (start - end) * t / duration)


def fmt_mean(values, precision: int = 2) -> str:
    return f"{np.mean(values):.{precision}f}" if len(values) else "n/a"


def q_learning():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # parameters
    num_envs = 3
    seed = None
    capture_video = True
    lr = 1e-4
    eps_start = 1.0
    eps_end = 0.05
    exploration_fraction = 0.5  # fraction of time_steps over which eps decays
    time_steps = 100_000
    gamma = 0.99
    log_interval = 1_000
    return_window = 20

    # initialization
    device = torch.accelerator.current_accelerator(
        check_available=True
    ) or torch.device("cpu")
    envs = gym.vector.SyncVectorEnv(
        [
            # bind i at definition time; only record video for the first env
            lambda i=i: make_env(
                capture_video and i == 0, None if seed is None else seed + i
            )
            for i in range(num_envs)
        ]
    )
    q_network = QNetwork(envs).to(device)
    optimizer = optim.AdamW(q_network.parameters(), lr=lr)

    def to_tensor(x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)

    logger.info(
        f"Start Q Learning | env={ENV_ID} num_envs={num_envs} device={device} "
        f"seed={seed} capture_video={capture_video}"
    )
    logger.info(
        f"Hyperparameters | lr={lr:g} eps={eps_start:g}->{eps_end:g} "
        f"(over {exploration_fraction:.0%} of steps) gamma={gamma:g} "
        f"time_steps={time_steps}"
    )
    logger.debug(f"Q network:\n{q_network}")

    # start
    episode_returns, losses, step_rewards = [], [], []
    start_time = time.time()
    obs, _ = envs.reset(seed=seed)
    # with NEXT_STEP autoreset, the step after an episode ends only resets
    # that env, so its transition is not a real one and is skipped
    autoreset = np.zeros(num_envs, dtype=bool)

    step = 0
    for step in range(1, time_steps + 1):
        eps = linear_schedule(
            eps_start, eps_end, exploration_fraction * time_steps, step - 1
        )

        # epsilon-greedy, decided per env
        obs_t = to_tensor(obs)
        with torch.no_grad():
            greedy_actions = q_network(obs_t).argmax(dim=1).cpu().numpy()
        explore = np.random.random(num_envs) < eps
        actions = np.where(explore, envs.action_space.sample(), greedy_actions)
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        if "episode" in infos:
            returns = infos["episode"]["r"][infos["_episode"]].tolist()
            episode_returns.extend(returns)
            logger.debug(f"Step {step} | finished episode returns={returns}")

        # one TD(0) update on the valid transitions
        valid = ~autoreset
        if valid.any():
            step_rewards.append(rewards[valid].mean())

            q = q_network(obs_t[valid])
            q = q.gather(1, to_tensor(actions[valid], torch.long).unsqueeze(1))
            with torch.no_grad():
                next_q = q_network(to_tensor(next_obs[valid])).max(dim=1).values
                not_done = 1 - to_tensor(terminateds[valid])
                target = to_tensor(rewards[valid]) + gamma * not_done * next_q

            loss = nn.functional.mse_loss(q.squeeze(1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        autoreset = terminateds | truncateds
        obs = next_obs

        if step % log_interval == 0:
            sps = int(step * num_envs / (time.time() - start_time))
            logger.info(
                f"Step {step}/{time_steps} | episodes={len(episode_returns)} | "
                f"mean_return(last {return_window})="
                f"{fmt_mean(episode_returns[-return_window:])} | "
                f"mean_reward={fmt_mean(step_rewards, 3)} | "
                f"mean_loss={fmt_mean(losses[-log_interval:], 4)} | "
                f"eps={eps:.3f} | SPS={sps}"
            )
            step_rewards.clear()

        if len(episode_returns) >= 10 and min(episode_returns[-10:]) >= 500:
            logger.info(f"Solved at step {step}: last 10 episodes reached 500")
            break

    envs.close()
    logger.info(
        f"Finished Q Learning | steps={step} episodes={len(episode_returns)} "
        f"best_return={max(episode_returns, default='n/a')} "
        f"mean_return(last {return_window})="
        f"{fmt_mean(episode_returns[-return_window:])} | "
        f"elapsed={time.time() - start_time:.1f}s"
    )

    logger.info("Saving model weight")
    utils.save_model(q_network, "q_network")
