import typing
from pandas import DataFrame
import numpy as np
from src.agent import Agent


class RacelineFollowAgent(Agent):

    def __init__(self, raceline: DataFrame):
        self._raceline = raceline

    def take_action(
        self, obs: typing.Dict, info: typing.Dict, deterministic=True
    ) -> np.ndarray:
        current_x = info["pose_x"]
        current_y = info["pose_y"]
        current_theta = info["pose_theta"]

        # Find row with closest position
        self._raceline["distance"] = np.sqrt(
            (self._raceline["x_m"] - current_x) ** 2
            + (self._raceline["y_m"] - current_y) ** 2
        )

        closest_row = self._raceline.loc[self._raceline["distance"].idxmin()]

        target_velocity = closest_row["vx_mps"]
        target_theta = closest_row["psi_rad"]

        return np.array([0.0, 0.0])
