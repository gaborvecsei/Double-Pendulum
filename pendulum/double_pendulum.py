import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from plotly import graph_objs
from scipy.integrate import odeint


class DoublePendulum:
    def __init__(self, length_1=1.0, mass_1=1.0, length_2=1.0, mass_2=1.0, gravity=9.8):
        self.L1 = length_1
        self.M1 = mass_1
        self.L2 = length_2
        self.M2 = mass_2
        self.G = gravity

        self._t = None
        self._trajectory = None

    @property
    def traj(self):
        if self._trajectory is None:
            raise ValueError("First run the calculation!")
        return self._trajectory

    @traj.setter
    def traj(self, val):
        self._trajectory = val

    @property
    def first_arm(self):
        traj = self.traj
        # (n_steps, (x, y))
        return traj[:, (0, 1)]

    @property
    def second_arm(self):
        traj = self.traj
        # (n_steps, (x, y))
        return traj[:, (2, 3)]

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, val):
        self._t = val

    def copy(self):
        obj = DoublePendulum(length_1=self.L1, mass_1=self.M1, length_2=self.L2, mass_2=self.M2, gravity=self.G)
        return obj

    def _derivs(self, state, t):
        # This fn is from https://matplotlib.org/examples/animation/double_pendulum_animated.html
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        del_ = state[2] - state[0]
        den1 = (self.M1 + self.M2) * self.L1 - self.M2 * self.L1 * np.cos(del_) * np.cos(del_)
        dydx[1] = (self.M2 * self.L1 * state[1] * state[1] * np.sin(del_) * np.cos(del_) +
                   self.M2 * self.G * np.sin(state[2]) * np.cos(del_) +
                   self.M2 * self.L2 * state[3] * state[3] * np.sin(del_) -
                   (self.M1 + self.M2) * self.G * np.sin(state[0])) / den1

        dydx[2] = state[3]

        den2 = (self.L2 / self.L1) * den1
        dydx[3] = (-self.M2 * self.L2 * state[3] * state[3] * np.sin(del_) * np.cos(del_) +
                   (self.M1 + self.M2) * self.G * np.sin(state[0]) * np.cos(del_) -
                   (self.M1 + self.M2) * self.L1 * state[1] * state[1] * np.sin(del_) -
                   (self.M1 + self.M2) * self.G * np.sin(state[2])) / den2

        return dydx

    def calculate_trajectory(self, start_time, end_time, dt, init_angle_1, init_velocity_1, init_angle_2,
                             init_velocity_2):
        t = np.arange(start_time, end_time, dt)
        init_state = np.radians([init_angle_1, init_velocity_1, init_angle_2, init_velocity_2])
        y = odeint(self._derivs, init_state, t)

        x1 = self.L1 * np.sin(y[:, 0])
        y1 = -self.L1 * np.cos(y[:, 0])

        x2 = self.L2 * np.sin(y[:, 2]) + x1
        y2 = -self.L2 * np.cos(y[:, 2]) + y1

        trajectory = np.stack([x1, y1, x2, y2], axis=-1)

        self.t = t
        self.traj = trajectory


class DoublePendulumPlotter:
    def __init__(self, pendulums, dt):
        self.pendulums = pendulums
        self.dt = dt
        self.max_pend_length = np.max([p.L1 + p.L2 for p in self.pendulums])

        self.fig = None
        self.ax = None
        self.time_template = None
        self.time_text = None
        self.lines = None

    def _setup_fig_for_anim(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.grid()
        self.ax.set_xlim((-self.max_pend_length, self.max_pend_length))
        self.ax.set_ylim((-self.max_pend_length, self.max_pend_length))

        # Initialize plot objects
        self.time_template = "time: {0:.1f}"
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.lines = []
        for p in self.pendulums:
            line, = self.ax.plot([], [], 'o-', lw=2)
            self.lines.append(line)

    def _animate(self, i):
        for p, line in zip(self.pendulums, self.lines):
            x = [0, p.first_arm[i][0], p.second_arm[i][0]]
            y = [0, p.first_arm[i][1], p.second_arm[i][1]]
            line.set_data(x, y)
        self.time_text.set_text(self.time_template.format(i * self.dt))
        ret = *self.lines, self.time_text
        return ret

    def anim(self, frame_interval=25, frame_index_array=None):
        self._setup_fig_for_anim()
        if frame_index_array is None:
            frame_index_array = np.arange(1, len(self.pendulums[0].traj))
        anim = animation.FuncAnimation(self.fig,
                                       self._animate,
                                       frame_index_array,
                                       interval=frame_interval,
                                       blit=True)
        plt.close()
        return anim

    def trajectory_plot(self):
        fig = graph_objs.Figure()
        for i, p in enumerate(self.pendulums):
            trace = graph_objs.Scatter(x=p.second_arm[:, 0], y=p.second_arm[:, 1], name="Pendulum {0}".format(i))
            fig.add_trace(trace)
        fig.update_layout(title="Trajectories of the pendulums")
        return fig

    def trajectory_plot_slider(self, custom_dt=None):
        if custom_dt is None:
            custom_dt = self.dt
        nb_steps = int(len(self.pendulums[0].first_arm) * custom_dt)
        fig = graph_objs.Figure()
        for step in np.arange(1, nb_steps + 1):
            for i, p in enumerate(self.pendulums):
                asd = int(step / custom_dt)
                trace = graph_objs.Scatter(x=p.second_arm[:asd, 0],
                                           y=p.second_arm[:asd, 1],
                                           name="Pendulum {0}".format(i),
                                           visible=False)
                fig.add_trace(trace)

        for trace in fig.data[-len(self.pendulums):]:
            trace.visible = True

        steps = []
        for i in range(0, len(fig.data), len(self.pendulums)):
            step = dict(method="restyle", args=["visible", [False] * len(fig.data)])
            for k in range(len(self.pendulums)):
                # We need to make every pendulum visible at the current step
                step["args"][1][i + k] = True
            steps.append(step)

        sliders = [dict(active=len(steps) - 1, currentvalue={"prefix": "Seconds: "}, pad={"t": 50}, steps=steps)]
        fig.update_layout(title="Trajectories of the pendulums",
                          width=640,
                          height=480,
                          sliders=sliders,
                          xaxis=dict(range=[-self.max_pend_length, self.max_pend_length]),
                          yaxis=dict(range=[-self.max_pend_length, self.max_pend_length]))
        return fig
