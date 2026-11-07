import gymnasium as gym

ENV_ID = "CartPole-v1"
VIDEO_DIR = "videos"
VIDEO_STEP_INTERVAL = 1_000


def make_env(
    capture_video: bool = False,
    seed: int | None = None,
    video_dir: str = VIDEO_DIR,
) -> gym.Env:
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            # start a new recording every VIDEO_STEP_INTERVAL env steps
            step_trigger=lambda step: step % VIDEO_STEP_INTERVAL == 0,
        )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def make_vector_env(
    num_envs: int,
    seed: int | None = None,
    capture_video: bool = False,
    video_dir: str = VIDEO_DIR,
) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [
            # bind i at definition time; only record video for the first env
            lambda i=i: make_env(
                capture_video and i == 0,
                None if seed is None else seed + i,
                video_dir,
            )
            for i in range(num_envs)
        ]
    )
