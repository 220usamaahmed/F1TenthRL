from typing import Callable, Tuple, List
from src.agent import Agent
import random
from os import path


def roemerlager_map_generator() -> Callable[
    [], Tuple[str, str, List[Tuple[float, float, float]], List["Agent"]]
]:
    maps = ["original", "wide", "narrow", "cones-1", "cones-2", "cones-3", "cones-4"]
    # maps = ["cones-4"]
    maps_ext = ".png"
    reset_pose = [[0.0, 0.0, 0.0]]

    map = path.join("maps", "roemerlager", random.choice(maps))

    return map, maps_ext, reset_pose, []
