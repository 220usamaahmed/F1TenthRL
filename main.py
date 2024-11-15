import yaml
from argparse import Namespace
from src import dummy_agent
from src.agent import Agent
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
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


def build_env(config: Namespace) -> F110_SB_Env:
    return F110_SB_Env(
        map=config.map_path,
        map_ext=config.map_ext,
        lidar_params={"num_beams": LIDAR_NUM_BEAMS, "fov": LIDAR_FOV},
        reset_pose=(config.starting_x, config.starting_y, config.starting_theta),
    )


def run_environment(env: F110_SB_Env, agent: Agent):

    obs, info = env.reset()
    env.enable_beam_rendering()
    env.render()

    while True:
        action = agent.take_action(obs)
        obs, step_reward, done, truncated, info = env.step(action)
        env.render(mode="human")

        if done:
            break


def main():
    # TODO: Get agent and map from command line arguments
    config = load_map_config("circle")
    env = build_env(config)

    ppo_agent = PPOAgent(env)
    dummy_agent = DummyAgent()

    run_environment(env, ppo_agent)


if __name__ == "__main__":
    main()
