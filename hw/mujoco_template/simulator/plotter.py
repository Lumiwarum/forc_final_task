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
        
    def __init__(self, task_id):
        if self._initialized:
            return
        self.nq = 6
        self.task_id = task_id
        self.times = []
        self.joint_pos = {i: [] for i in range(6)}
        self.joint_vel = {i: [] for i in range(6)}
        self.control = {i: [] for i in range(6)}
        self.des_pos = {i: [] for i in range(6)}
        self.des_vel = {i: [] for i in range(6)}
        self.j_names = [f'Joint {i+1}' for i in range(6)]
        self._initialized = True
        signal.signal(signal.SIGINT, self._save_plots)
        self.cntr = 0


    def add_data(self, q, dq, t, u, des_pos, des_vel):
        self.cntr +=1
        self.times.append(t)
        for i in range(self.nq):
            self.joint_pos[i].append(q[i])
            self.joint_vel[i].append(dq[i])
            self.control[i].append(u[i])
            self.des_pos[i].append(des_pos[i])
            self.des_vel[i].append(des_vel[i])
        if self.cntr > 100 :
            self._save_plots()
            self.cntr = 0

    def _save_plots(self):
        positions = np.array([self.joint_pos[i] for i in range(self.nq)]).T
        velocities = np.array([self.joint_vel[i] for i in range(self.nq)]).T
        self.plot_results(positions, self.j_names, "Joint_positions", "[rad]")
        self.plot_results(velocities, self.j_names, "Joint_velocities", "[rad/s]")
        positions = np.array([np.array(self.des_pos[i])-np.array(self.joint_pos[i]) for i in range(6)]).T
        velocities = np.array([np.array(self.des_vel[i])-np.array(self.joint_vel[i]) for i in range(6)]).T
        self.plot_results(positions, self.j_names, "Joint_positions_error", "[rad]")
        self.plot_results(velocities, self.j_names, "Joint_velocities_error", "[rad/s]")
        control = np.array([self.control[i] for i in range(self.nq)]).T
        self.plot_results(control, self.j_names, "Control_applied_to_joints", "[Nm]")
            

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
        plt.savefig(f'logs/plots/0{self.task_id}_{title}.png')
        plt.close()
