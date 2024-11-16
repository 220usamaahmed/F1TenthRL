import typing
import yaml
from datetime import datetime
from argparse import Namespace
import os
import numpy as np
from src.agent import Agent
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
from src.playback_agent import PlaybackAgent
from src.f110_sb_env import F110_SB_Env


NUM_AGENTS = 1
LIDAR_NUM_BEAMS = 1080
LIDAR_FOV = 4.7


def load_map_config(map_name: str) -> Namespace:
    with open("map_configs.yaml") as file:
        maps_config = yaml.load(file, Loader=yaml.FullLoader)

    map_config = maps_config.get(map_name)
    assert map_config is not None, f"Could not load config for map {map_name}"

    return Namespace(**map_config)


def build_env(config: Namespace, enable_action_recording=False) -> F110_SB_Env:
    return F110_SB_Env(
        map=config.map_path,
        map_ext=config.map_ext,
        lidar_params={"num_beams": LIDAR_NUM_BEAMS, "fov": LIDAR_FOV},
        reset_pose=(config.starting_x, config.starting_y, config.starting_theta),
        record_actions=enable_action_recording,
    )


def run_environment(
    env: F110_SB_Env,
    agent: Agent,
    deterministic=True,
    max_timesteps=np.inf,
    verbose=False,
):

    obs, info = env.reset()
    env.enable_beam_rendering()
    env.render()

    t = 0

    while True:
        if t > max_timesteps:
            break
        t += 1

        action = agent.take_action(obs, deterministic=deterministic)
        obs, step_reward, done, truncated, info = env.step(action)
        env.render(mode="human")

        if verbose:
            print(f"--- t = {t:03} {'-' * 16}")
            print("Action", action)
            print("Velocity X", obs["linear_vel_x"])
            print("Velocity Y", obs["linear_vel_y"])
            print("Angular Velocity Z", obs["angular_vel_z"])
            print("Reward", step_reward)
            print("Done", done)

        if done:
            break


def save_recording(name: str, recordings: typing.List[typing.List[np.ndarray]]):
    directory_path = os.path.join("action_recordings", name)

    os.makedirs(directory_path, exist_ok=True)

    for i, recording in enumerate(recordings):
        with open(os.path.join(directory_path, f"episode_{i}.csv"), "w") as file:
            for action in recording:
                file.write(",".join(list(map(str, action))) + "\n")

    print(f"{len(recordings)}(s) recordings saved at: {directory_path}")


def main():
    # TODO: Get agent and map from command line arguments
    config = load_map_config("circle")
    env = build_env(config, enable_action_recording=False)

    # Dummy agent
    # dummy_agent = DummyAgent(steer=0, speed=1)
    # run_environment(env, dummy_agent, max_timesteps=300, verbose=True)

    # PPO agent (having issues with inf)
    try:
        ppo_agent = PPOAgent(env)
        ppo_agent.learn()
    except:
        print("Learning failed.")
        if env.record_actions:
            save_recording(
                f"ppo_agent_inf_issue-{datetime.now().strftime('%y-%m-%d_%h:%m:%s')}",
                env.get_recorded_actions(),
            )
        return
    env.enable_recording()
    run_environment(env, ppo_agent, verbose=True)
    save_recording(
        f"ppo_agent_eval-{datetime.now().strftime('%y-%m-%d-%h-%m-%s')}",
        env.get_recorded_actions(),
    )

    # Playback agent (investigating PPO inf issue)
    # playback_agent = PlaybackAgent(
    #     # recording_path="./action_recordings/ppo_agent_inf_issue/episode_1.csv"
    #     recording_path="./action_recordings/ppo_agent_eval-24-11-16-Nov-11-1731715477/episode_1.csv"
    # )
    # run_environment(env, playback_agent, verbose=True, max_timesteps=1000)


if __name__ == "__main__":
    main()
