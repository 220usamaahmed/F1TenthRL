import optuna
from optuna.trial import Trial
from src.utils import load_map_config, build_env, evaluate
from src.ppo_agent import PPOAgent


def _objective(trial: Trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)

    # Policy kwargs
    net_arch_pi = trial.suggest_categorical(
        "pi_layer", [[128, 64], [256, 128], [64, 32]]  # type: ignore
    )
    net_arch_vf = trial.suggest_categorical(
        "vf_layer", [[128, 64], [256, 128], [64, 32]]  # type: ignore
    )
    net_arch = dict(pi=net_arch_pi, vf=net_arch_vf)

    config = load_map_config("example")
    env = build_env(config, enable_recording=True)
    agent = PPOAgent.create(
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        net_arch=net_arch,
    )

    agent.learn(total_timesteps=1000)

    return evaluate(env, agent)


def run_ppo_agent_study():
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=50)
    print("Best hyperparamters:", study.best_params)
