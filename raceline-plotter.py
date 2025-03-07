from src.ppo_agent import PPOAgent
from src.f110_sb_env import F110_SB_Env
from src.sticky_action_wrapper import StickyActionWrapper
from src.multi_map_wrapper import MultiMapWrapper
from src.utils import (
    load_map_config,
    build_env,
    run_environment,
    save_recording,
    get_date_tag,
    load_latest_model,
)
from src.map_generators import roemerlager_map_generator
from src.raceline_plotter import plot_racelines
import os


def plot_intermediates(env: F110_SB_Env, config, folder: str, n=3):
    if not os.path.isdir(folder):
        return None

    files = os.listdir(folder)
    files = [file for file in files if file.endswith(".zip")]

    agents = {}

    for file in sorted(files, key=lambda x: int(x.split("_")[-2]), reverse=True)[:n]:
        file = os.path.join(
            folder, os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]
        )

        print(file)
        agents[file[-15:]] = PPOAgent.create_from_saved_model(file)

    plot_racelines(
        f"{config.map_path}{config.map_ext}",
        env,
        agents,
        per_agent=1
    )


def main():
    config_names = [
        "roemerlager", 
        "roemerlager-wide",
        "roemerlager-narrow", 
        "roemerlager-cones-1", 
        "roemerlager-cones-2", 
        "roemerlager-cones-3",
        "roemerlager-cones-4"
    ]

    for name in config_names:
        config = load_map_config(name)
        env = build_env(
            config,
            other_agents=[],
            enable_recording=True,
        )
        env = StickyActionWrapper(env=env, tick_rate=0.1, fine_rendering=False)
        # env = MultiMapWrapper(env=env, map_generator=roemerlager_map_generator)

        plot_intermediates(env, config, "models/ppo_agent_25-03-06_19-22-38", n=2)


if __name__ == "__main__":
    main()
