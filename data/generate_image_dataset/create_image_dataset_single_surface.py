"""
Author: Antonello Paolino
Date: 2024-05-29
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates a 2D representation fo the 3D pressure map on
                the robot component surfaces
"""

import numpy as np
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
from src.robot import Robot
from src.flow import FlowImporter

# Define the target surface to be saved
SURFACE_NAME = "ironcub_head"

def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Define reduced surface list
    surface_list = [SURFACE_NAME]
    # Get the path to the raw data
    data_dir = input("Enter the path to the fluent data directory: ")
    data_path = Path(str(data_dir).strip())
    file_names = [file.name for file in data_path.rglob('*.dtbs') if file.is_file()]
    config_names = sorted(list(set([file_name.split("-")[0] for file_name in file_names])))
    joint_config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file_path, delimiter=",", dtype=str)
    dataset = {}
    for config_name in config_names:
        joint_pos = joint_configs[joint_configs[:,0] == config_name][0,1:].astype(float)
        pitch_yaw_angles = find_pitch_yaw_angles(config_name, file_names)
        pitch_angles = []
        yaw_angles = []
        counter = 0
        for pitch_angle, yaw_angle in pitch_yaw_angles:
            pitch_angle = int(pitch_angle)  # Cast to integer
            yaw_angle = int(yaw_angle)  # Cast to integer
            # Set robot state and get link to world transformations
            robot.set_state(pitch_angle, yaw_angle, joint_pos*np.pi/180)
            link_H_world_dict = robot.compute_all_link_H_world()
            # Initialize flow object
            flow = FlowImporter()
            # Import fluent data from all surfaces
            flow.import_raw_fluent_data(data_path, config_name, pitch_angle, yaw_angle, surface_list)        
            flow.transform_local_fluent_data(link_H_world_dict, flow_velocity=17.0, flow_density=1.225)
            flow.assign_global_fluent_data()
            # Data Interpolation and Image Generation
            flow.interpolate_flow_data(robot.image_resolutions, surface_list, robot.surface_axes)
            image = flow.surface[SURFACE_NAME].image
            # Initialize database structure of dimensions (N_images, N_channels, X, Y) and store data
            if counter == 0:
                database = np.empty(shape=(len(pitch_yaw_angles), image.shape[0], image.shape[1], image.shape[2]), dtype=np.float16)
            database[counter, :, :, :] = image
            pitch_angles.append(pitch_angle)
            yaw_angles.append(yaw_angle)
            counter += 1
            print(f"{config_name} configuration progress: {counter}/{pitch_yaw_angles.shape[0]}", end='\r', flush=True)
        # Assign dataset variables
        dataset[config_name] = {
            "pitch_angles": np.array(pitch_angles).astype(np.float16),
            "yaw_angles": np.array(yaw_angles).astype(np.float16),
            "data": np.array(database).astype(np.float16),
        }
        # Save compressed dataset using compressed numpy
        np.savez_compressed(str(dataset_dir / f"{SURFACE_NAME}-{config_name}.npz"),data=dataset[config_name])
        print(f"Dataset for {config_name} configuration saved.")


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
