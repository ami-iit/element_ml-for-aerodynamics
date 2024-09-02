"""
Author: Antonello Paolino
Date: 2024-08-28
Description:    Python script to generate random samples of joint configurations
                using Latin Hypercube Sampling and optimize them removing the 
                self-collisions. The optimized samples are saved to a csv file.
"""

# Import libraries
import numpy as np
import pathlib
# Import custom classes
from src.robot import Robot
from src.latin_hypercube_sampling import generate_lhs_sampling
from src.optimizer import Optimizer

NUMBER_OF_CONFIGURATIONS = 100
DISPLAY_OPTIMIZED_CONFIGURATIONS = False
SAMPLING_SEED = 42

def main():
    # Get file path
    root_path = pathlib.Path(__file__).parents[0]
    # Initialize robot and flow objects
    robot_name = "iRonCub-Mk3"
    old_robot = Robot(robot_name)
    new_robot = Robot(robot_name)
    
    # Generate joint configurations using Latine Hypercube Sampling
    attitude_ranges = [
        [0, 180], # alpha range
        [-180, 180] # beta range
        ]
    complete_ranges = attitude_ranges + old_robot.wind_tunnel_joint_limits
    samples = generate_lhs_sampling(num_samples=NUMBER_OF_CONFIGURATIONS, ranges=complete_ranges, seed=SAMPLING_SEED)
    joint_configurations = samples[:,2:]

    # Optimize configurations
    opt = Optimizer(new_robot)
    new_joint_configurations = np.zeros(shape=(0, joint_configurations.shape[1]))
    for config_index, joint_configuration in enumerate(joint_configurations):
        old_robot.set_state(pitch_angle=0, yaw_angle=0, joint_positions=joint_configuration*np.pi/180)
        new_robot.set_state(pitch_angle=0, yaw_angle=0, joint_positions=joint_configuration*np.pi/180)
        optimized, new_joint_pos = opt.solve(joint_configuration*np.pi/180)
        new_joint_configurations = np.vstack((new_joint_configurations, new_joint_pos*180/np.pi))
        new_robot.set_state(pitch_angle=0, yaw_angle=0, joint_positions=new_joint_pos)
        if optimized and DISPLAY_OPTIMIZED_CONFIGURATIONS:
            old_robot.visualize_with_collision_spheres(title=f"Config {config_index+1}: non-optimized", non_blocking=False)
            new_robot.visualize_with_collision_spheres(title=f"Config {config_index+1}: optimized", non_blocking=False)
            new_robot.visualize_robot_comparison(old_robot, title=f"Config {config_index+1}: new (green) vs old (red)",non_blocking=False)
        print(f"config {config_index+1}/{NUMBER_OF_CONFIGURATIONS}", end='\r', flush=True)
    optimized_samples = np.hstack((samples[:,:2], new_joint_configurations))
    #Save optimized samples to csv file with precision 5
    file_path = root_path / f"n{NUMBER_OF_CONFIGURATIONS}_optimized_sampling.csv"
    np.savetxt(str(file_path), optimized_samples, delimiter=",", fmt='%.2f')
    print(f"Optimized samples saved to {file_path}")


if __name__ == "__main__":
    main()
