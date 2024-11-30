from math import trunc
import yaml
from datetime import datetime
from argparse import Namespace
import os
import numpy as np
from src.agent import Agent
from src.f110_sb_env import F110_SB_Env
from src.runtime_visualization import RuntimeVisualizer


def load_map_config(map_name: str) -> Namespace:
    with open("map_configs.yaml") as file:
        maps_config = yaml.load(file, Loader=yaml.FullLoader)

    map_config = maps_config.get(map_name)
    assert map_config is not None, f"Could not load config for map {map_name}"

    return Namespace(**map_config)


def build_env(config: Namespace, enable_recording=False) -> F110_SB_Env:
    return F110_SB_Env(
        map=config.map_path,
        map_ext=config.map_ext,
        reset_pose=(config.starting_x, config.starting_y, config.starting_theta),
        record=enable_recording,
    )


def run_environment(
    env: F110_SB_Env,
    agent: Agent,
    deterministic=True,
    max_timesteps=np.inf,
    verbose=False,
    render_mode="human_slow",
):
    obs, info = env.reset()
    env.enable_beam_rendering()
    env.render()

    with RuntimeVisualizer() as rv:
        t = 0
        while True:
            if t > max_timesteps:
                break
            t += 1

            action = agent.take_action(obs, deterministic=deterministic)
            obs, step_reward, terminated, truncated, info = env.step(action)

            rv.add_data(action, obs)
            env.render(mode=render_mode)

            if verbose:
                print(f"--- t = {t:03} {'-' * 16}")
                print("Action", action)
                print("Velocity X", obs["linear_vel_x"])
                print("Velocity Y", obs["linear_vel_y"])
                print("Angular Velocity Z", obs["angular_vel_z"])
                print("Reward", step_reward)

            if terminated or truncated:
                break


def evaluate(env: F110_SB_Env, agent: Agent, n_eval_episodes=10):
    reward_sums = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        reward_sum = 0
        while True:
            action = agent.take_action(obs, deterministic=False)
            obs, step_reward, terminated, truncated, _ = env.step(action)
            reward_sum += step_reward

            if terminated or truncated:
                break

        reward_sums.append(reward_sum)

    return np.mean(reward_sums)


def save_recording(name: str, actions, rewards, observations, infos):
    directory_path = os.path.join("recordings", name)

    os.makedirs(directory_path, exist_ok=True)

    with open(os.path.join(directory_path, f"actions.csv"), "w") as file:
        for action in actions:
            file.write(",".join(list(map(str, action))) + "\n")

    with open(os.path.join(directory_path, f"rewards.csv"), "w") as file:
        for reward in rewards:
            file.write(str(reward) + "\n")

    with open(os.path.join(directory_path, f"observations.csv"), "w") as file:
        for observation in observations:
            file.write(str(observation) + "\n")

    with open(os.path.join(directory_path, f"info.csv"), "w") as file:
        for info in infos:
            file.write(str(info) + "\n")

    print(f"Recording saved at: {directory_path}")


def get_date_tag() -> str:
    return datetime.now().strftime("%y-%m-%d_%H:%M:%S")
