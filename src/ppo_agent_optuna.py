import optuna
from optuna.trial import Trial
from src.utils import (
    get_date_tag,
    load_map_config,
    build_env,
    evaluate,
    save_dict_list_to_csv,
)
from src.ppo_agent import PPOAgent
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_contour,
)


STUDY_NAME = "optuna_studies/ppo_agent_study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
TRAILS_SPECS = f"{STUDY_NAME}.csv"

trials = []


def _objective(trial: Trial):
    global trials

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)

    # Policy kwargs
    pi_layer_choices = [[128, 64], [256, 128], [64, 32]]
    vf_layer_choices = [[128, 64], [256, 128], [64, 32]]
    net_arch_pi = trial.suggest_categorical(
        "pi_layer", list(range(len(pi_layer_choices)))  # type: ignore
    )
    net_arch_vf = trial.suggest_categorical(
        "vf_layer", list(range(len(vf_layer_choices)))  # type: ignore
    )
    net_arch = dict(pi=pi_layer_choices[net_arch_pi], vf=vf_layer_choices[net_arch_vf])

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
        verbose=False,
        tensorboard_logs=None,
    )

    agent.learn(total_timesteps=10000)
    total_mean_reward = evaluate(env, agent)

    file_path = f"./models/ppo_agent_{get_date_tag()}"

    agent.save_model(file_path)
    trials.append(
        {
            "file_path": file_path,
            "total_mean_reward": total_mean_reward,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "net_arch": net_arch,
        }
    )

    return total_mean_reward


def run_ppo_agent_study():
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_NAME, direction="maximize"
    )
    study.optimize(_objective, n_trials=50)
    save_dict_list_to_csv(TRAILS_SPECS, trials)
    _display_study_results(study)


def display_study_results():
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_NAME, load_if_exists=True
    )
    _display_study_results(study)


def _display_study_results(study):
    print("Best hyperparamters:", study.best_params)

    plot_optimization_history(study).show()
    plot_parallel_coordinate(study).show()
    plot_param_importances(study).show()
    plot_slice(study).show()
    plot_slice(study).show()
    plot_contour(study).show()
