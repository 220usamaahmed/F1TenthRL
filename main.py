from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
from src.agent import SBAgentLearningException
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
from src.playback_agent import PlaybackAgent
from src.raceline_follow_agent import RacelineFollowAgent
from src.ppo_agent_optuna import run_ppo_agent_study, display_study_results
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
    load_latest_models,
    run_environment_with_plots
)
from src.map_generators import roemerlager_map_generator
from src.ga_refinement import refine
from src.raceline_plotter import plot_racelines


import argparse

def get_env(config_name="roemerlager", sticky_actions=True, multi_map=False, fine_rendering=False):
    config = load_map_config(config_name)
    env = build_env(
        config,
        other_agents=[],
        enable_recording=False,
    )

    if sticky_actions:
        env = StickyActionWrapper(env=env, tick_rate=0.1, fine_rendering=fine_rendering)

    if multi_map:
        env = MultiMapWrapper(env=env, map_generator=roemerlager_map_generator)

    return env

def train(sticky_actions: bool, multi_map: bool, timesteps: int, save_freq: int):
    print(f"Training with sticky_actions={sticky_actions}, multi_map={multi_map}, timesteps={timesteps}, save_freq={save_freq}")

    env = get_env(sticky_actions=sticky_actions, multi_map=multi_map)

    save_tag = input("Enter save path (defalut: current time): ./models/ppo_agent-[...]") or get_date_tag()
    save_path = f"./models/ppo_agent-{save_tag}/"

    try:
        # Hyper paramters from Optuna study
        ppo_agent = PPOAgent.create(
            env,
            learning_rate=0.00038641971654092917,
            gamma=0.9780603367895372,
            n_steps=2048,
            batch_size=64,
            n_epochs=5,
            net_arch=dict(pi=[64, 32], vf=[256, 128]),
        )
        # ppo_agent = PPOAgent.create_from_saved_model(
        #     "", env=env
        # )
        ppo_agent.learn(model_tag=save_tag, total_timesteps=timesteps, save_freq=save_freq, save_path=save_path)
        # ppo_agent.save_model(f"./models/ppo_agent_{get_date_tag()}")
        run_environment(env, ppo_agent, deterministic=True, verbose=False)
    except SBAgentLearningException as e:
        print("Learning failed.")
        print(e)
    finally:
        if env.record:
            save_recording(
                f"ppo_agent_inf_issue-{get_date_tag()}",
                *env.get_recording(),
                )


def run(config_name: str, model_tag: str, index_from_end: int, runs: int, sticky_actions: bool, multi_map: bool):
    print(f"Running with index_from_end={index_from_end}")

    env = get_env(config_name=config_name, sticky_actions=sticky_actions, multi_map=multi_map, fine_rendering=True)
    model_filepath = load_latest_model(model_tag, index_from_end=index_from_end)

    print("model: ", model_filepath)

    for _ in range(runs):
        ppo_agent = PPOAgent.create_from_saved_model(model_filepath)
        # env.enable_recording()
        run_environment_with_plots(env, ppo_agent, deterministic=False, verbose=False)

def plot_raceline(config_name: str, model_tag: str, index_from_end: int):
    print(f"Plotting racelines with index_from_end={index_from_end}")

    env = get_env(config_name=config_name, sticky_actions=True, multi_map=False, fine_rendering=True)
    config = load_map_config(config_name)

    agents = {}

    for tag in model_tag.split(","):
        filepath = load_latest_model(tag, index_from_end=index_from_end)
        print("model: ", filepath)
        name = filepath.split("__")[-1]
        agents[name] = PPOAgent.create_from_saved_model(filepath)

    plot_racelines(
        f"{config.map_path}{config.map_ext}",
        env,
        agents,
        cmap="speed",
        per_agent=1
    )

def run_dummy_agent(speed: float, steer: float):
    print(f"Running dummy agent with speed={speed}, steer={steer}")

    env = get_env(sticky_actions=True, multi_map=False)
    dummy_agent = DummyAgent(steer=1.0, speed=0.2)
    run_environment(env, dummy_agent, verbose=False)

def run_playback_agent(save_file: str):
    print(f"Running playback agent with save_file={save_file}")

    env = get_env(sticky_actions=True, multi_map=False)
    playback_agent = PlaybackAgent(recording_path=save_file)
    run_environment(env, playback_agent, deterministic=True, verbose=True)

def compare_racelines(config_name, model_tag: str, last_n: int):
    print(f"Comparing racelines with models_path={model_tag}, last_n={last_n}")

    env = get_env(config_name=config_name, sticky_actions=True, multi_map=False, fine_rendering=True)
    config = load_map_config(config_name)
    paths = load_latest_models(model_tag)

    agents = {}

    for path in paths:
        print("model: ", path)
        name = path.split("__")[-1]
        agents[name] = PPOAgent.create_from_saved_model(path)

    plot_racelines(
        f"{config.map_path}{config.map_ext}",
        env,
        agents,
        per_agent=10,
        cmap="success"
        # cmap="speed"
    )

def run_refinement():
    raise NotImplementedError

def run_optuna_study():
    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--sticky_actions", type=bool, required=False, default=True)
    train_parser.add_argument("--multi_map", type=bool, required=False, default=False)
    train_parser.add_argument("--timesteps", type=int, required=False, default=2000000)
    train_parser.add_argument("--save_freq", type=int, required=False, default=100000)
    train_parser.set_defaults(func=lambda args: train(args.sticky_actions, args.multi_map, args.timesteps, args.save_freq))
    
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config_name", type=str, required=False, default="roemerlager")
    run_parser.add_argument("--model_tag", type=str, required=True)
    run_parser.add_argument("--index_from_end", type=int, required=False, default=0)
    run_parser.add_argument("--runs", type=int, required=False, default=1)
    run_parser.add_argument("--sticky_actions", type=bool, required=False, default=True)
    run_parser.add_argument("--multi_map", type=bool, required=False, default=False)
    run_parser.set_defaults(func=lambda args: run(args.config_name, args.model_tag, args.index_from_end, args.runs, args.sticky_actions, args.multi_map))
    
    plot_parser = subparsers.add_parser("plot_raceline")
    plot_parser.add_argument("--config_name", type=str, required=False, default="roemerlager")
    plot_parser.add_argument("--model_tag", type=str, required=True)
    plot_parser.add_argument("--index_from_end", type=int, required=False, default=0)
    plot_parser.set_defaults(func=lambda args: plot_raceline(args.config_name, args.model_tag, args.index_from_end, ))
    
    dummy_parser = subparsers.add_parser("run_dummy_agent")
    dummy_parser.add_argument("--speed", type=float, required=True)
    dummy_parser.add_argument("--steer", type=float, required=True)
    dummy_parser.set_defaults(func=lambda args: run_dummy_agent(args.speed, args.steer))

    playback_parser = subparsers.add_parser("run_playback_agent")
    dummy_parser.add_argument("--save_file", type=str, required=True)
    playback_parser.set_defaults(func=lambda args: run_playback_agent(args.save_file))
    
    compare_parser = subparsers.add_parser("compare_racelines")
    compare_parser.add_argument("--config_name", type=str, required=False, default="roemerlager")
    compare_parser.add_argument("--model_tag", type=str, required=True)
    compare_parser.add_argument("--last_n", type=int, required=False, default=2)
    compare_parser.set_defaults(func=lambda args: compare_racelines(args.config_name, args.model_tag, args.last_n))

    refine_parser = subparsers.add_parser("refine")
    refine_parser.set_defaults(func=lambda _: run_refinement())

    optuna_parser = subparsers.add_parser("optuna")
    optuna_parser.set_defaults(func=lambda _: run_optuna_study())
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()