import typing
from src.f110_sb_env import F110_SB_Env
from src.agent import Agent
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as mcoll
import numpy as np


def get_raceline(env: F110_SB_Env, agent: Agent, deterministic: bool = False):
    raceline = []
    success = False

    obs, info = env.reset()
    while True:
        action = agent.take_action(obs, info, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        x = (info.get("pose_x", 0) + 6.67) / 0.05
        y = (info.get("pose_y", 0) + 1.43) / 0.05

        raceline.append((x, y))
        # env.render()

        if terminated or truncated:
            success = info.get("checkpoint_done")
            break

    return raceline, success


def get_agent_racelines(env: F110_SB_Env, agent: Agent, n=5):
    racelines = []
    successes = []

    for _ in range(n):
        raceline, success = get_raceline(env, agent, deterministic=n == 1)
        racelines.append(raceline)
        successes.append(success)

    return racelines, successes


def calc_speed(xs, ys, dt=0.01):
    vx = np.zeros_like(xs)
    vy = np.zeros_like(ys)

    vx[1:-1] = (xs[2:] - xs[:-2]) / (2 * dt)
    vy[1:-1] = (ys[2:] - ys[:-2]) / (2 * dt)

    vx[0] = (xs[1] - xs[0]) / dt
    vx[-1] = (xs[-1] - xs[-2]) / dt
    vy[0] = (ys[1] - ys[0]) / dt
    vy[-1] = (ys[-1] - ys[-2]) / dt

    speed = np.sqrt(vx**2 + vy**2)

    return speed


def calc_direction(xs, ys):
    vx = np.zeros_like(xs)
    vy = np.zeros_like(ys)

    vx[1:-1] = (xs[2:] - xs[:-2])
    vy[1:-1] = (ys[2:] - ys[:-2])

    vx[0] = (xs[1] - xs[0])
    vx[-1] = (xs[-1] - xs[-2])
    vy[0] = (ys[1] - ys[0])
    vy[-1] = (ys[-1] - ys[-2])

    direction = np.round(np.degrees(np.arctan2(vx, vy)), 2)

    direction[direction < 0] += 360

    return direction

def plot_racelines(
    map_path: str, env: F110_SB_Env, agents: typing.Dict[str, Agent], per_agent=1, cmap="success"
):
    fig, axs = plt.subplots(1, len(agents))
    map = mpimg.imread(map_path)
    map = np.flipud(map)
    if cmap == "speed":
        map = 255 - map

    if len(agents) == 1:
        axs = [axs]

    for i, (name, agent) in enumerate(agents.items()):
        racelines, successes = get_agent_racelines(env, agent, n=per_agent)

        cbar_displayed = False

        for line, success in zip(racelines, successes):
            xs = np.array([point[0] for point in line])
            ys = np.array([point[1] for point in line])
                
            if cmap=="success":
                axs[i].plot(xs, ys, linewidth=1, c="#81C784" if success else "#E57373")
            elif cmap=="speed":
                vs = calc_speed(xs * 0.05, ys * 0.05)
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = mcoll.LineCollection(segments, cmap='RdYlGn_r', norm=plt.Normalize(0.7, 5))
                lc.set_array(vs)

                axs[i].add_collection(lc)

                if not cbar_displayed:
                    cbar = plt.colorbar(lc, ax=axs[i])
                    cbar.set_label("Velocity")
                    cbar_displayed = True
            elif cmap=="direction":
                vs = calc_direction(xs * 0.05, ys * 0.05)

                np.set_printoptions(suppress=True)  # Suppress scientific notation
                np.set_printoptions(precision=2)    # Set decimal precision (optional)

                print(vs)

                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = mcoll.LineCollection(segments, cmap='hsv', norm=plt.Normalize(0, 360))
                lc.set_array(vs)

                axs[i].add_collection(lc)

                if not cbar_displayed:
                    cbar = plt.colorbar(lc, ax=axs[i])
                    cbar.set_label("Direction")
                    cbar_displayed = True


        axs[i].imshow(map, cmap="gray")
        axs[i].set_title(name.split("/")[-1])

    fig.suptitle("Raceline Comparison")
    plt.show()