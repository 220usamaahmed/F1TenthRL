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
TRIALS_SPECS = f"{STUDY_NAME}.csv"

trials = []


def _objective(trial: Trial):
    global trails

    ...
    return 0


def run_reward_func_ppo_study():
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_NAME, direction="maximize"
    )
    study.optimize(_objective, n_trials=50)
    save_dict_list_to_csv(TRIALS_SPECS, trials)
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
