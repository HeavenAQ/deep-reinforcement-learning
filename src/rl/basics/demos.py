"""Small demos of single and vectorized gymnasium environments with random actions."""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from rl.basics.envs import ENV_ID, VIDEO_DIR, make_vector_env


def create_single_env(recording: bool):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if recording:
        env = gym.wrappers.RecordVideo(env, VIDEO_DIR)
    obs, info = env.reset()
    return env, obs, info


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
    envs = make_vector_env(3)
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
