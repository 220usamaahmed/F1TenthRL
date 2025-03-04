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
)
from src.map_generators import roemerlager_map_generator


def run_dummy_agent(env: F110_SB_Env):
    dummy_agent = DummyAgent(steer=0.0, speed=0.1)
    run_environment(env, dummy_agent, verbose=True)


def train_ppo_agent(env: F110_SB_Env, total_timesteps=10000):
    try:
        # Hyper paramters from Optuna study
        ppo_agent = PPOAgent.create(
            env,
            learning_rate=0.00038641971654092917,
            n_steps=2048,
            batch_size=64,
            n_epochs=5,
            gamma=0.9780603367895372,
            net_arch=dict(pi=[64, 32], vf=[256, 128]),
        )
        # ppo_agent = PPOAgent.create_from_saved_model(
        #     "", env=env
        # )
        ppo_agent.learn(total_timesteps=total_timesteps)
        ppo_agent.save_model(f"./models/ppo_agent_{get_date_tag()}")
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


def run_ppo_agent(env: F110_SB_Env, model_path: str, runs=1):
    for _ in range(runs):
        ppo_agent = PPOAgent.create_from_saved_model(model_path)
        env.enable_recording()
        run_environment(env, ppo_agent, deterministic=True, verbose=False)


def run_playback_agent(env: F110_SB_Env, save_file: str):
    playback_agent = PlaybackAgent(recording_path=save_file)
    run_environment(env, playback_agent, deterministic=True, verbose=True)


def run_raceline_follow_agent(env: F110_SB_Env, map_path: str):
    print(map_path)

    race_line_df = pd.read_csv(
        map_path.replace("_map", "_raceline.csv"),
        skiprows=2,
        sep=";",
    )
    race_line_df.rename(
        columns={
            old_name: old_name.replace("#", "").lstrip()
            for old_name in race_line_df.columns
        },
        inplace=True,
    )

    raceline_follow_agent = RacelineFollowAgent(race_line_df)
    run_environment(env, raceline_follow_agent)


def main():
    train = 0
    runs = 1

    config = load_map_config("roemerlager")
    # config = load_map_config("reference")
    env = build_env(
        config,
        other_agents=[],
        enable_recording=True,
    )
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True)
    # env = StickyActionWrapper(env=env, tick_rate=0.1, fine_rendering=not train)
    # env = MultiMapWrapper(env=env, map_generator=roemerlager_map_generator)

    # check_env(env, warn=False)

    # run_dummy_agent(env)

    if train:
        train_ppo_agent(env, total_timesteps=100000)
    else:
        model_filepath = load_latest_model(index_from_end=0)
        print(f"Loading model: {model_filepath}")
        run_ppo_agent(env, model_filepath, runs=runs)

    # run_ppo_agent_study()
    # display_study_results()

    # run_playback_agent(
    #     env, "./action_recordings/ppo_agent_inf_issue-24-11-29_13:03:01/episode_1.csv"
    # )

    # run_raceline_follow_agent(env, config.map_path)


if __name__ == "__main__":
    main()
