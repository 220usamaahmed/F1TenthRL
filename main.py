import typing
import yaml
from argparse import Namespace
import os
import numpy as np
from src.agent import Agent
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
from src.playback_agent import PlaybackAgent
from src.f110_sb_env import F110_SB_Env


NUM_AGENTS = 1
LIDAR_NUM_BEAMS = 32
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


def run_environment(env: F110_SB_Env, agent: Agent, verbose=False):

    obs, info = env.reset()
    env.enable_beam_rendering()
    env.render()

    while True:
        action = agent.take_action(obs)
        obs, step_reward, done, truncated, info = env.step(action)
        env.render(mode="human")

        if verbose:
            print(obs)

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
    # dummy_agent = DummyAgent()

    # PPO agent (having issues with inf)
    # try:
    #     ppo_agent = PPOAgent(env)
    #     ppo_agent.learn()
    # except:
    #     print("Learning failed.")
    #     print("Actions recoreded")
    #
    #     save_recording("ppo_agent_inf_issue", env.get_recorded_actions())

    # Playback agent (investigating PPO inf issue)
    playback_agent = PlaybackAgent(
        recording_path="./action_recordings/ppo_agent_inf_issue/episode_1.csv"
    )
    run_environment(env, playback_agent, verbose=True)


if __name__ == "__main__":
    main()
