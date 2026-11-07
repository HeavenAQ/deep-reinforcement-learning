import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def fmt_mean(values, precision: int = 2) -> str:
    return f"{np.mean(values):.{precision}f}" if len(values) else "n/a"


class TrainingStats:
    """Collects episode returns, rewards and losses during training and logs them."""

    def __init__(self, num_envs: int, return_window: int = 20):
        self.num_envs = num_envs
        self.return_window = return_window
        self.episode_returns: list[float] = []
        # cleared on every periodic log
        self.interval_rewards: list[float] = []
        self.interval_losses: list[float] = []
        self.start_time = time.time()

    def record_episodes(self, step: int, infos: dict) -> None:
        if "episode" in infos:
            returns = infos["episode"]["r"][infos["_episode"]].tolist()
            self.episode_returns.extend(returns)
            logger.debug(f"Step {step} | finished episode returns={returns}")

    def record_rewards(self, rewards: np.ndarray) -> None:
        if len(rewards):
            self.interval_rewards.append(float(rewards.mean()))

    def record_loss(self, loss: float) -> None:
        self.interval_losses.append(loss)

    def solved(self, threshold: float = 500, n_episodes: int = 10) -> bool:
        recent = self.episode_returns[-n_episodes:]
        return len(recent) == n_episodes and min(recent) >= threshold

    def recent_mean_return(self) -> str:
        return fmt_mean(self.episode_returns[-self.return_window :])

    def log(self, step: int, total_steps: int, eps: float) -> None:
        sps = int(step * self.num_envs / (time.time() - self.start_time))
        logger.info(
            f"Step {step}/{total_steps} | episodes={len(self.episode_returns)} | "
            f"mean_return(last {self.return_window})={self.recent_mean_return()} | "
            f"mean_reward={fmt_mean(self.interval_rewards, 3)} | "
            f"mean_loss={fmt_mean(self.interval_losses, 4)} | "
            f"eps={eps:.3f} | SPS={sps}"
        )
        self.interval_rewards.clear()
        self.interval_losses.clear()

    def summary(self, name: str, step: int) -> None:
        logger.info(
            f"Finished {name} | steps={step} episodes={len(self.episode_returns)} "
            f"best_return={max(self.episode_returns, default='n/a')} "
            f"mean_return(last {self.return_window})={self.recent_mean_return()} | "
            f"elapsed={time.time() - self.start_time:.1f}s"
        )
