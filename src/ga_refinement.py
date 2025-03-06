import typing
from src.ppo_agent import PPOAgent
from src.ga_ppo_agent import GA_PPO_Agent
from src.f110_sb_env import F110_SB_Env
from src.utils import load_map_config, build_env, get_date_tag
import numpy as np

N = 20  # Agents per generation
P = 5  # Number of agents to pick from the top


def evaluate(
    env: F110_SB_Env,
    agent: typing.Union[GA_PPO_Agent, PPOAgent],
    n_eval_episodes=3,
    failed_lap_time=100,
):
    lap_times = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        while True:
            action = agent.take_action(obs, info, deterministic=False)
            obs, step_reward, terminated, truncated, info = env.step(action)

            # env.render()

            if terminated or truncated:
                if info.get("checkpoint_done"):
                    lap_times.append(info.get("lap_time"))
                else:
                    lap_times.append(failed_lap_time)
                break

    return np.mean(lap_times)


def run_trails(env, proto_agent, generations=10):
    baseline_score = evaluate(env, proto_agent)
    print("Baseline score", baseline_score)

    generation = [GA_PPO_Agent(proto_agent) for _ in range(N)]

    for g in range(generations):
        scores = [evaluate(env, agent) for agent in generation]
        sorted_score_idx = np.argsort(scores)
        top_score_idx = sorted_score_idx[:P]

        next_generation = []
        for i in top_score_idx:
            pick = generation[i]
            for _ in range(N // P):
                next_generation.append(GA_PPO_Agent(pick.get_mutated_agent()))

        print(f"Generation {g + 1} score: {np.mean(scores)}")
        generation = next_generation

    return generation


def pick_best(env: F110_SB_Env, generation: typing.List[GA_PPO_Agent]) -> GA_PPO_Agent:
    scores = [evaluate(env, agent) for agent in generation]
    sorted_score_idx = np.argsort(scores)
    return generation[sorted_score_idx[0]]


def refine():
    assert N % P == 0

    config = load_map_config("roemerlager-cones-4")
    env = build_env(
        config,
        other_agents=[],
        enable_recording=True,
    )

    ppo_agent = PPOAgent.create_from_saved_model("./models/ppo_agent_25-03-05_14:48:02")
    assert isinstance(ppo_agent, PPOAgent)

    generation = run_trails(env, ppo_agent)
    best = pick_best(env, generation)

    best_score = evaluate(env, best)
    print("Best score", best_score)

    best.get_mutated_agent().save_model(f"./models/ppo_agent_{get_date_tag()}")
