import typing
import yaml
from datetime import datetime
from argparse import Namespace
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env
from src.agent import Agent, SBAgentLearningException
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
from src.playback_agent import PlaybackAgent
from src.f110_sb_env import F110_SB_Env
from src.runtime_visualization import RuntimeVisualizer


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
        reset_pose=(config.starting_x, config.starting_y, config.starting_theta),
        record_actions=enable_action_recording,
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


def save_recording(name: str, recordings: typing.List[typing.List[np.ndarray]]):
    directory_path = os.path.join("action_recordings", name)

    os.makedirs(directory_path, exist_ok=True)

    for i, recording in enumerate(recordings):
        with open(os.path.join(directory_path, f"episode_{i}.csv"), "w") as file:
            for action in recording:
                file.write(",".join(list(map(str, action))) + "\n")

    print(f"{len(recordings)}(s) recordings saved at: {directory_path}")


def get_date_tag() -> str:
    return datetime.now().strftime("%y-%m-%d_%H:%M:%S")


def main():
    # TODO: Get agent and map from command line arguments
    config = load_map_config("example")
    env = build_env(config, enable_action_recording=True)
    # check_env(env, warn=False)

    # Dummy agent
    # dummy_agent = DummyAgent(steer=0.1, speed=0.1)
    # run_environment(env, dummy_agent, max_timesteps=300, verbose=False)

    # PPO agent
    try:
        ppo_agent = PPOAgent.create(env)
        ppo_agent.learn(total_timesteps=20000)
        ppo_agent.save_model(f"./models/ppo_agent_{get_date_tag()}")

        # ppo_agent = PPOAgent.create_from_saved_model(
        #     # "./models/ppo_agent_24-11-21_16:49:12"  # Trained on example
        #     "./models/ppo_agent_24-11-22_01:06:42"  # Trained on circle
        # )

        # env.enable_recording()
        run_environment(env, ppo_agent, deterministic=True, verbose=False)
        # save_recording(
        #     f"ppo_agent_eval-{get_date_tag()}",
        #     env.get_recorded_actions(),
        # )
    except SBAgentLearningException as e:
        print("Learning failed.")
        print(e)
    finally:
        if env.record_actions:
            save_recording(
                f"ppo_agent_inf_issue-{get_date_tag()}",
                env.get_recorded_actions(),
            )
        return

    # Playback agent
    # playback_agent = PlaybackAgent(
    #     recording_path="./action_recordings/ppo_agent_eval-24-11-17_01:41:00/episode_1.csv"
    # )
    # run_environment(env, playback_agent, deterministic=True, verbose=True)


if __name__ == "__main__":
    main()
