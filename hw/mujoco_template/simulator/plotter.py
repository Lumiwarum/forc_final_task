import matplotlib.pyplot as plt
import numpy as np
import signal

class Plotter():
    _instance = None

    def __new__(cls, *args, **kwargs):
        # If no instance of class already exits
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, data, model):
        if self._initialized:
            return
        self.nq = model.nq
        self.times = []
        self.joint_pos = {i: [] for i in range(model.nq)}
        self.joint_vel = {i: [] for i in range(model.nv)}
        self.control = {i: [] for i in range(model.nq)}
        self.ee_pos = {i: [] for i in range(6)}
        self.ee_vel = {i: [] for i in range(6)}
        self.er_pos = {i: [] for i in range(6)}
        self.er_vel = {i: [] for i in range(6)}
        self.j_names = [f'Joint {i+1}' for i in range(model.nq)]
        self.ee_pos_names = ["X", "Y", "Z", "Theta", "Psi", "Omega"]
        self._initialized = True
        signal.signal(signal.SIGINT, self._save_plots)
        self.cntr = 0


    def add_data(self, q, dq, t, u, er_pos, er_vel):
        self.cntr +=1
        self.times.append(t)
        for i in range(self.nq):
            self.joint_pos[i].append(q[i])
            self.joint_vel[i].append(dq[i])
            self.control[i].append(u[i])
            self.er_pos[i].append(er_pos[i])
            self.er_vel[i].append(er_vel[i])
        if self.cntr > 100 :
            self._save_plots()
            self.cntr = 0

    def _save_plots(self):
        positions = np.array([self.joint_pos[i] for i in range(self.nq)]).T
        velocities = np.array([self.joint_vel[i] for i in range(self.nq)]).T
        self.plot_results(positions, self.j_names, "Joint_positions", "[rad]")
        self.plot_results(velocities, self.j_names, "Joint_velocities", "[rad/s]")
        positions = np.array([self.er_pos[i] for i in range(6)]).T
        velocities = np.array([self.er_vel[i] for i in range(6)]).T
        self.plot_results(positions, self.ee_pos_names, "End_Effector_positions_error", "[m]/[rad]")
        self.plot_results(velocities, self.ee_pos_names, "End_Effector_velocities_error", "[m/s]/[rad/s]")
            

    def plot_results(self, data, names, title, units):
        """Plot and save simulation results."""
        # Joint positions plot
        plt.figure(figsize=(10, 6))
        for i in range(data.shape[1]):
            plt.plot(self.times, data[:, i], label=names[i])
        plt.xlabel('Time [s]')
        plt.ylabel(title.replace("_", " ") + " " + units)
        plt.title(title.replace("_", " ")+' over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'logs/plots/03_{title}.png')
        plt.close()
