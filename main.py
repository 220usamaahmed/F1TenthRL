from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from pyglet.gl import GL_LINES
from src.agent import Agent
from src.dummy_agent import DummyAgent

NUM_AGENTS = 1

# TODO: Find out how to modify these at the time of initialization
LIDAR_NUM_BEAMS = 32
LIDAR_FOV = np.pi / 2

beam_gl_lines = []


def load_map_config(map_name: str) -> Namespace:
    with open("map_configs.yaml") as file:
        maps_config = yaml.load(file, Loader=yaml.FullLoader)

    map_config = maps_config.get(map_name)
    assert map_config is not None, f"Could not load config for map {map_name}"

    return Namespace(**map_config)


def render_callback(env_renderer):
    global beam_gl_lines

    if not len(beam_gl_lines):
        for _ in range(LIDAR_NUM_BEAMS):
            gl_line = env_renderer.batch.add(
                2,
                GL_LINES,
                None,
                ("v2f/stream", (0, 0, 0, 0)),
                ("c3B/stream", (255, 0, 0, 255, 0, 0)),
            )
            beam_gl_lines.append(gl_line)


def update_beam_gl_lines(obs):
    global beam_gl_lines
    assert len(beam_gl_lines) == LIDAR_NUM_BEAMS

    car_x = obs["poses_x"][0]
    car_y = obs["poses_y"][0]
    car_theta = obs["poses_theta"][0]
    scans = obs["scans"][0]

    assert len(scans) == LIDAR_NUM_BEAMS

    for beam_i in range(LIDAR_NUM_BEAMS):
        gl_line = beam_gl_lines[beam_i]
        theta = car_theta + ((beam_i / LIDAR_NUM_BEAMS) * LIDAR_FOV) - (LIDAR_FOV / 2)

        end_x = car_x + np.cos(theta) * scans[beam_i]
        end_y = car_y + np.sin(theta) * scans[beam_i]

        # TODO: Find out why this is working with 50
        gl_line.vertices = [car_x * 50, car_y * 50, end_x * 50, end_y * 50]


def run_environment(config: Namespace, agent: Agent):
    env = gym.make(
        "f110_gym:f110-v0",
        map=config.map_path,
        map_ext=config.map_ext,
        num_agents=1,
        lidar_params={
            "num_beams": LIDAR_NUM_BEAMS,
            "fov": LIDAR_FOV,
        },
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(
        np.array([[config.starting_x, config.starting_y, config.starting_theta]])
    )

    env.render()

    while not done:
        speed, steer = agent.take_action(obs)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))

        update_beam_gl_lines(obs)
        env.render(mode="human")


def main():
    # TODO: Get agent and map from command line arguments
    config = load_map_config("example")
    agent = DummyAgent()
    run_environment(config, agent)


if __name__ == "__main__":
    main()
