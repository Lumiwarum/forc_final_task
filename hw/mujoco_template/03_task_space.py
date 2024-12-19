"""Task space (operational space) control example.

This example demonstrates how to implement task space control for a robot arm,
allowing it to track desired end-effector positions and orientations. The example
uses a simple PD control law but can be extended to more sophisticated controllers.

Key Concepts Demonstrated:
    - Task space control implementation
    - End-effector pose tracking
    - Real-time target visualization
    - Coordinate frame transformations

Example:
    To run this example:
    
    $ python 03_task_space.py

Notes:
    - The target pose can be modified interactively using the MuJoCo viewer
    - The controller gains may need tuning for different trajectories
    - The example uses a simplified task space controller for demonstration
"""

import numpy as np
from scipy.linalg import logm
from simulator import Simulator, plotter
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin


def skew_to_vector(skew_matrix):
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

def calc_error(pos, R, pos_des, R_des):
    error_twist = pin.log3(R_des @ R.T)
    error_pos = pos_des - pos
    error = np.concatenate([error_pos, error_twist])
    return error

def jacobians(model, data, q, dq):
    ee_frame_id = model.getFrameId("end_effector")
    J = np.zeros((6, 6))
    dJdq = np.zeros((6))
    J[:3,:] = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)[:3,:]
    J[3:, :] = pin.getFrameJacobian(model, data, ee_frame_id, pin.WORLD)[3:, :]

    ddq = np.array([0., 0., 0., 0., 0., 0.])
    pin.forwardKinematics(model, data, q, dq, ddq)
    dJdq[:3] = pin.getFrameAcceleration(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED).linear
    dJdq[3:] = pin.getFrameAcceleration(model, data, ee_frame_id, pin.WORLD).angular
    return J, dJdq

def get_desired(t):
    pos = np.array([np.sin(np.pi/2*t), np.cos(np.pi/2*t), 1])
    rot = np.eye(3)
    return pos, rot

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Example task space controller."""
    #compute everything
    pin.computeAllTerms(model, data, q, dq)

    kp = np.array([100, 100, 100, 10, 10, 10])
    kd = np.array([20, 20, 20, 2, 2, 2])
    
    # Convert desired pose to SE3
    desired_position = desired['pos']
    desired_quaternion = desired['quat'] # [w, x, y, z] in MuJoCo format
    desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) # Convert to [x,y,z,w] for Pinocchio
    # Convert to pose and SE3
    desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
    desired_se3 = pin.XYZQUATToSE3(desired_pose)
    desired_position = desired_se3.translation
    desired_rotation = desired_se3.rotation
    #desired_position, desired_rotation = get_desired(t)
    
    # Get end-effector frame
    ee_frame_id = model.getFrameId("end_effector")
    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation
    error_full = calc_error(ee_position, ee_rotation, desired_position, desired_rotation) 
    
    pin.updateFramePlacement(model, data, ee_frame_id)
    M = data.M
    nle = data.nle
    J, dJdq = jacobians(model, data, q, dq)
    J_inv = np.linalg.pinv(J)

    dp_e = -J@dq

    inner_loop =  (kp * error_full + kd * dp_e - dJdq)

    u = M @ J_inv @ (inner_loop) + nle

    ploter.add_data(q, dq, t, u, error_full, dp_e)
    
    return u

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/03_task_space.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=6.0)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    ploter = plotter.Plotter("3")
    main() 