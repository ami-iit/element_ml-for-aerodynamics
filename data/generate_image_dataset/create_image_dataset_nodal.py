"""
Author: Antonello Paolino
Date: 2024-10-01
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates a 2D representation fo the 3D pressure map on
                the robot component surfaces and saves the images in a dataset.
"""

import numpy as np
import pickle
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
from src.robot import Robot
from src.flow_new import FlowImporter


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    mesh_robot = Robot(robot_name, urdf_path)
    robot = Robot(robot_name, urdf_path)
    # Initialize flow object
    flow = FlowImporter()
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Define map directory
    map_dir = Path(__file__).parents[0] / "maps"
    # Get the path to the raw data
    data_dir = input("Enter the path to the fluent data directory: ")
    data_path = Path(str(data_dir).strip())
    file_names = [file.name for file in data_path.rglob("*.dtbs") if file.is_file()]
    config_names = sorted(
        list(set([file_name.split("-")[0] for file_name in file_names]))
    )
    config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(config_file_path, delimiter=",", dtype=str)

    dataset = {}
    for config_name in config_names:
        joint_pos = joint_configs[joint_configs[:, 0] == config_name][0, 1:].astype(
            float
        )
        joint_pos *= np.pi / 180
        # Import and set mesh mapping
        map_file = list(map_dir.rglob(f"{config_name}-map.npy"))[0]
        map_data = np.load(map_file, allow_pickle=True).item()
        mesh_robot.set_state(0, 0, joint_pos)
        mesh_link_H_world_dict = mesh_robot.compute_all_link_H_world()
        flow.import_mesh_mapping_data(
            map_data, robot.surface_list, mesh_link_H_world_dict
        )
        pitch_yaw_angles = find_pitch_yaw_angles(config_name, file_names)
        pitch_angles = []
        yaw_angles = []
        # Initialize database structure of dimensions (N_images, N_channels, X, Y) and store data
        database = np.empty(
            shape=(
                len(pitch_yaw_angles),
                len(robot.surface_list),
                robot.image_resolution[0],
                robot.image_resolution[1],
            ),
            dtype=np.float16,
        )
        for idx, (pitch_yaw) in enumerate(pitch_yaw_angles):
            pitch = int(pitch_yaw[0])
            yaw = int(pitch_yaw[1])
            # Set robot state and get link to world transformations
            robot.set_state(pitch, yaw, joint_pos)
            link_H_world_dict = robot.compute_all_link_H_world()
            # Import fluent data from all surfaces
            flow.import_node_data(data_path, config_name, pitch, yaw)
            flow.transform_local_data(link_H_world_dict, airspeed=17.0, air_dens=1.225)
            flow.reorder_surface_data()
            flow.assign_global_fluent_data()
            # Data Interpolation and Image Generation
            flow.interp_3d_to_image(robot.image_resolution)
            database[idx, :, :, :] = flow.image
            pitch_angles.append(pitch)
            yaw_angles.append(yaw)
            print(
                f"{config_name} configuration progress: {idx}/{pitch_yaw_angles.shape[0]}",
                end="\r",
                flush=True,
            )
        # Assign dataset variables
        dataset[config_name] = {
            "pitch_angles": np.array(pitch_angles).astype(np.float16),
            "yaw_angles": np.array(yaw_angles).astype(np.float16),
            "database": np.array(database).astype(np.float16),
        }
        # Save compressed dataset using compressed numpy
        with open(str(dataset_dir / f"ironcub-{config_name}-nodal.npz"), "wb") as f:
            pickle.dump(dataset[config_name], f, protocol=4)
        print(f"Nodal image dataset for {config_name} configuration saved.")


def find_pitch_yaw_angles(joint_config_name, file_names):
    segmented_files = [
        file_name.split("-")
        for file_name in file_names
        if file_name.split("-")[0] == joint_config_name
    ]
    pitch_yaw_angles = np.empty((0, 2))
    for file_name in segmented_files:
        index = 1
        if file_name[index] != "":
            pitch_angle = int(file_name[index])
            index += 1
        else:
            pitch_angle = -int(file_name[index + 1])
            index += 2
        if file_name[index] != "":
            yaw_angle = int(file_name[index])
        else:
            yaw_angle = -int(file_name[index + 1])
        pitch_yaw_angles = np.vstack((pitch_yaw_angles, [pitch_angle, yaw_angle]))
    pitch_yaw_angles = np.unique(pitch_yaw_angles, axis=0)
    return pitch_yaw_angles


if __name__ == "__main__":
    main()
