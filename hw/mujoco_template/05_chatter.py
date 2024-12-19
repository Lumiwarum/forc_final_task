"""Example of setting dynamic parameters for the UR5e robot.

This script demonstrates how to modify dynamic parameters of the robot including:
- Joint damping coefficients
- Joint friction
- Link masses and inertias

The example uses a simple PD controller to show the effects of parameter changes.
"""

import numpy as np
from simulator import Simulator, plotter
from pathlib import Path
import os
from typing import Dict
import pinocchio as pin


def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
    pin.computeAllTerms(model, data, q, dq)
    M = data.M
    M_inv = np.linalg.inv(M)
    nle = data.nle
    # Control gains tuned for UR5e
    kp = np.ones(6) *100
    kd = np.ones(6) *20
    
    # Target joint configuration
    q_des = np.array([-1.4, -1.3, 1., 0, 0, 0])
    ddq_des = np.array([0, 0, 0, 0, 0, 0])
    dq_des = np.array([0, 0, 0, 0, 0, 0])
    
    epsi = 5
    q_e = q_des - q
    dq_e = dq_des -dq
    
    L = np.diag([30, 30, 30, 10, 10, 10])*10
    k = 700
    
    s = dq_e + L @ q_e
    
    v_s = k * s / np.linalg.norm(s)
    
    v = ddq_des + L @ dq_e + v_s
    
    tau = M @ v + nle
    
    ploter.add_data(q, dq, t, tau, q_des , dq_des)
    
    return tau


def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,  # Using joint space control
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/02_chatter.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    
    # Set joint damping (example values, adjust as needed)
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Set joint friction (example values, adjust as needed)
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    
    # Get original properties
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print(f"\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Add the end-effector mass and inertia
    sim.modify_body_properties(ee_name, mass=3)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")
    
    # Set controller and run simulation
    sim.set_controller(joint_controller)
    sim.run(time_limit=4.0)

if __name__ == "__main__":
    ploter = plotter.Plotter("2")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    main() 