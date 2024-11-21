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

        self._pyqt_process.start()

    @staticmethod
    def _run_pyqt(queue):
        app = QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(show=True, title="F1Tenth Env Vis")
        win.resize(800, 600)
        obs_plot = win.addPlot(title="Observations")
        x_lin_vel_curve = obs_plot.plot(pen="r")
        y_lin_vel_curve = obs_plot.plot(pen="g")
        z_ang_vel_curve = obs_plot.plot(pen="y")

        x_lin_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        y_lin_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        z_ang_vel_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        steer_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)
        speed_data = deque(maxlen=RuntimeVisualizer.WINDOW_SIZE)

        def update():
            while not queue.empty():
                action, observation = queue.get()

                x_lin_vel_data.append(observation["linear_vel_x"])
                y_lin_vel_data.append(observation["linear_vel_y"])
                z_ang_vel_data.append(observation["angular_vel_z"])

                steer_data.append(action[0])
                speed_data.append(action[1])

            n = len(x_lin_vel_data)
            x = np.linspace(0, n, n)

            x_lin_vel_curve.setData(x, x_lin_vel_data)
            y_lin_vel_curve.setData(x, y_lin_vel_data)
            z_ang_vel_curve.setData(x, z_ang_vel_data)

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)

        app.exec_()

    def add_data(self, action, observation):
        self._data_queue.put((action, observation))

    def exit(self):
        self._pyqt_process.terminate()
        self._pyqt_process.join()
