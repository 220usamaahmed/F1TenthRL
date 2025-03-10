from multiprocessing import Process, Queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
from collections import deque


class RuntimeVisualizer:

    WINDOW_SIZE = 100

    def __init__(self):
        self._data_queue = Queue()
        self._pyqt_process = Process(
            target=RuntimeVisualizer._run_pyqt, args=(self._data_queue,)
        )
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
        if exc_type:
            print(f"Exception occurred: {exc_value}")
            print(traceback)
        return False

    @staticmethod
    def _run_pyqt(queue):
        app = QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(show=True, title="F1Tenth Env Vis")
        win.resize(640, 480)
        win.move(640, 0)

        obs_plot = win.addPlot(title="Observations")
        obs_legend = pg.LegendItem(offset=(70, 10))
        obs_legend.setParentItem(obs_plot.graphicsItem())
        obs_legend.setBrush(pg.mkBrush(100, 100, 100, 150))

        x_lin_vel_curve = obs_plot.plot(pen="r")
        y_lin_vel_curve = obs_plot.plot(pen="g")
        z_ang_vel_curve = obs_plot.plot(pen="y")

        obs_legend.addItem(x_lin_vel_curve, "X Linear Velocity")
        obs_legend.addItem(y_lin_vel_curve, "Y Linear Velocity")
        obs_legend.addItem(z_ang_vel_curve, "Z Angular Velocity")

        win.nextRow()

        action_plot = win.addPlot(title="Actions")
        action_legend = pg.LegendItem(offset=(70, 10))
        action_legend.setParentItem(action_plot.graphicsItem())
        action_legend.setBrush(pg.mkBrush(100, 100, 100, 150))

        steer_curve = action_plot.plot(pen="m")
        speed_curve = action_plot.plot(pen="c")

        action_legend.addItem(steer_curve, "Steer")
        action_legend.addItem(speed_curve, "Speed")

        win.nextRow()

        reward_plot = win.addPlot(title="Reward")
        reward_curve = reward_plot.plot(pen="y")

        x_lin_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        y_lin_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        z_ang_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        steer_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        speed_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        reward_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)

        def update():
            while not queue.empty():
                action, observation, reward = queue.get()

                x_lin_vel_data.append(observation["linear_vel_x"])
                y_lin_vel_data.append(observation["linear_vel_y"])
                z_ang_vel_data.append(observation["angular_vel_z"])

                steer_data.append(action[0])
                speed_data.append(action[1])

                reward_data.append(reward)

            n = len(x_lin_vel_data)
            x = np.linspace(0, n, n)

            x_lin_vel_curve.setData(x, x_lin_vel_data)
            y_lin_vel_curve.setData(x, y_lin_vel_data)
            z_ang_vel_curve.setData(x, z_ang_vel_data)

            steer_curve.setData(x, steer_data)
            speed_curve.setData(x, speed_data)

            reward_curve.setData(x, reward_data)

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)

        app.exec_()

    def start(self):
        self._pyqt_process.start()

    def add_data(self, action, observation, reward):
        if self._pyqt_process.is_alive():
            self._data_queue.put((action, observation, reward))

    def exit(self):
        self._data_queue.close()
        self._pyqt_process.terminate()
        self._pyqt_process.join()
