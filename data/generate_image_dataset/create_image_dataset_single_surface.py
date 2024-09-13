"""
Author: Antonello Paolino
Date: 2024-05-29
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates a 2D representation fo the 3D pressure map on
                the robot component surfaces
"""

# Import libraries
import numpy as np
import pathlib
import glob
import pickle
# Import custom classes
from src.robot import Robot
from src.flow import FlowImporter

import matplotlib.pyplot as plt

def main():
    surface_name = "ironcub_head"
    # Initialize robot and flow objects
    robot_name = "iRonCub-Mk3"
    robot = Robot(robot_name)
    # Dataset directory
    data_dir = pathlib.Path(__file__).parents[4] / "solver" / "data" / robot_name[-3:] / "database"
    # Images directory
    dataset_dir = pathlib.Path(__file__).parents[1] / "dataset"
    # Get the file name list
    file_names = [pathlib.Path(file).name for file in glob.glob(str(data_dir / "*.dtbs"))]
    # Group the files by the joint configuration
    joint_configs = sorted(list(set([file_name.split("-")[0] for file_name in file_names])))
    # Load csv file
    joint_config_file_path = pathlib.Path(__file__).parents[4] / "meshing" / "src" / "jointConfigFull-mk3.csv"
    joint_configurations = np.genfromtxt(joint_config_file_path, delimiter=",", dtype=str)
    dataset = {}
    for joint_config_name in joint_configs:        
        # Get the joint positions
        joint_positions = joint_configurations[joint_configurations[:,0] == joint_config_name][0,1:].astype(float)
        # Get the pitch and yaw angles list
        pitch_yaw_angles = find_pitch_yaw_angles(joint_config_name, file_names)
        # Initialize lists
        pitch_angles = []
        yaw_angles = []
        # Cycle on pitch and yaw angles
        counter = 0
        for pitch_angle, yaw_angle in pitch_yaw_angles:
            # Cast angles to integers
            pitch_angle = int(pitch_angle)
            yaw_angle = int(yaw_angle)            
            # Set robot state
            robot.set_state(pitch_angle, yaw_angle, joint_positions*np.pi/180)
            # Initialize flow object
            flow = FlowImporter(robot_name)
            # Compute link to world transformations
            link_H_world_dict = {}
            for surface_index in range(len(robot.surface_list)):
                # Compute the transformation from the link frame to the world frame (using zero rotation angles)
                surface_world_H_link = robot.compute_world_to_link_transform(frame_name=robot.surface_frames[surface_index], rotation_angle=0.0)
                surface_link_H_world = robot.invert_homogeneous_transform(surface_world_H_link) # alternative: np.linalg.inv(world_H_link)
                link_H_world_dict[robot.surface_list[surface_index]] = surface_link_H_world
            # Import fluent data from all surfaces
            flow.import_raw_fluent_data(joint_config_name, pitch_angle, yaw_angle, robot.surface_list)        
            flow.transform_local_fluent_data(link_H_world_dict, flow_velocity=17.0, flow_density=1.225)
            flow.assign_global_fluent_data()
            # Data Interpolation and Image Generation 
            resolution_scaling_factor = 1 # 1 for 1060 [px/m]
            surface_resolution = np.array(robot.image_resolutions)
            image_resolution_scaled = (surface_resolution * resolution_scaling_factor).astype(int) # scale to apply to the image resolution
            flow.interpolate_flow_data(image_resolution_scaled, robot.surface_list, robot.surface_axes)
            image = flow.surface[surface_name].image

            # Store surface data
            if counter == 0:
                database = np.empty(shape=(len(pitch_yaw_angles),image.shape[0],image.shape[1],image.shape[2]),dtype=np.float32)
            database[counter, :, :, :] = image
            pitch_angles.append(pitch_angle)
            yaw_angles.append(yaw_angle)
            # Update counter
            counter += 1
            # print status
            print(f"{joint_config_name} configuration progress: {counter}/{pitch_yaw_angles.shape[0]}", end='\r', flush=True)

        # Assign dataset variables
        dataset[joint_config_name] = {
            "pitch_angles": np.array(pitch_angles),
            "yaw_angles": np.array(yaw_angles),
            "data": database
        }

        # Save dataset
        with open(str(dataset_dir / f"{surface_name}-{joint_config_name}.npy"), "wb") as f:
            pickle.dump(dataset[joint_config_name], f, protocol=4)
        print(f"{joint_config_name} {surface_name} configuration dataset saved. \n")


def find_pitch_yaw_angles(joint_config_name, file_names):
    segmented_files = [file_name.split("-") for file_name in file_names if file_name.split("-")[0] == joint_config_name]
    pitch_yaw_angles = np.empty((0,2))
    for file_name in segmented_files:
        index = 1
        if file_name[index] != "":
            pitch_angle = int(file_name[index])
            index += 1
        else:
            pitch_angle = -int(file_name[index+1])
            index += 2
        if file_name[index] != "":
            yaw_angle = int(file_name[index])
        else:
            yaw_angle = -int(file_name[index+1])
        pitch_yaw_angles = np.vstack((pitch_yaw_angles, [pitch_angle, yaw_angle]))
    # Remove duplicates
    pitch_yaw_angles = np.unique(pitch_yaw_angles, axis=0)
    return pitch_yaw_angles

if __name__ == "__main__":
    main()
