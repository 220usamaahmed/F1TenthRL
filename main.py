import yaml
import numpy as np
from argparse import Namespace
from src.agent import Agent
from src.dummy_agent import DummyAgent
from src.f110_sb_env import F110_SB_Env
from src.feature_extractor import CustomCombinedExtractor
from stable_baselines3 import PPO


NUM_AGENTS = 1
LIDAR_NUM_BEAMS = 32
LIDAR_FOV = 4.7


def load_map_config(map_name: str) -> Namespace:
    with open("map_configs.yaml") as file:
        maps_config = yaml.load(file, Loader=yaml.FullLoader)

    map_config = maps_config.get(map_name)
    assert map_config is not None, f"Could not load config for map {map_name}"

    return Namespace(**map_config)


def build_env(config: Namespace) -> F110_SB_Env:
    return F110_SB_Env(
        map=config.map_path,
        map_ext=config.map_ext,
        lidar_params={"num_beams": LIDAR_NUM_BEAMS, "fov": LIDAR_FOV},
        reset_pose=(config.starting_x, config.starting_y, config.starting_theta),
    )


def run_environment(env: F110_SB_Env, model):

    obs, info = env.reset()
    env.render()

    while True:
        action, _states = model.predict(obs)
        print(action)
        obs, step_reward, done, truncated, info = env.step(action)
        env.render(mode="human")

        if done:
            break


def create_model(env):
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/",
    )

    return model


def main():
    # TODO: Get agent and map from command line arguments
    config = load_map_config("circle")
    env = build_env(config)

    model = create_model(env)
    model.learn(total_timesteps=1000)

    # run_environment(env, model)

    # agent = DummyAgent()
    # run_environment(env, agent)


if __name__ == "__main__":
    main()
