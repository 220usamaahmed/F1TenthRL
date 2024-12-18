from stable_baselines3.common.env_checker import check_env
from src.agent import SBAgentLearningException
from src.dummy_agent import DummyAgent
from src.ppo_agent import PPOAgent
from src.playback_agent import PlaybackAgent
from src.ppo_agent_optuna import run_ppo_agent_study, display_study_results
from src.utils import *


def run_dummy_agent(env: F110_SB_Env):
    dummy_agent = DummyAgent(steer=0.0, speed=0.0)
    run_environment(env, dummy_agent, verbose=False)


def train_ppo_agent(env: F110_SB_Env):
    try:
        # Hyper paramters from Optuna study
        ppo_agent = PPOAgent.create(
            env,
            learning_rate=0.00038641971654092917,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.9780603367895372,
            net_arch=dict(pi=[64, 32], vf=[256, 128]),
        )
        # ppo_agent = PPOAgent.create_from_saved_model(
        #     "", env=env
        # )
        ppo_agent.learn(total_timesteps=100000)
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


def run_ppo_agent(env: F110_SB_Env, model_path: str):
    ppo_agent = PPOAgent.create_from_saved_model(model_path)
    env.enable_recording()
    run_environment(env, ppo_agent, deterministic=True, verbose=True)


def run_playback_agent(env: F110_SB_Env, save_file: str):
    playback_agent = PlaybackAgent(recording_path=save_file)
    run_environment(env, playback_agent, deterministic=True, verbose=True)


def main():
    config = load_map_config("example")
    env = build_env(
        config,
        other_agents=[DummyAgent(0, 0) for _ in range(len(config.starting_poses) - 1)],
        # other_agents=[],
        enable_recording=True,
    )
    # check_env(env, warn=False)

    # run_dummy_agent(env)

    # train_ppo_agent(env)
    # run_ppo_agent(env, "./models/ppo_agent_24-12-01_14:30:03")
    # run_ppo_agent_study()
    display_study_results()

    # run_playback_agent(
    #     env, "./action_recordings/ppo_agent_inf_issue-24-11-29_13:03:01/episode_1.csv"
    # )


if __name__ == "__main__":
    main()
