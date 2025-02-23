"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    This code generates a dataset of the aerodynamic flow variables 
                acting on the surface of the iRonCub robot as point cloud data.
                The dataset is structured to be used for training a Multi-Layer
                Perceptron (MLP) learning architecture.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def main():
    # Create a dataset output directory
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Get the path to the raw data
    data_dir = input("Enter the path to the fluent dtbs directory: ")
    data_path = Path(str(data_dir).strip())
    file_names = [file.name for file in data_path.rglob("*.dtbs") if file.is_file()]
    config_names = sorted(
        list(set([file_name.split("-")[0] for file_name in file_names]))
    )
    joint_config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file_path, delimiter=",", dtype=str)
    data = []
    for config_name in config_names:
        joint_pos = joint_configs[joint_configs[:, 0] == config_name][0, 1:].astype(
            float
        )
        pitch_yaw_angles = find_pitch_yaw_angles(config_name, file_names)
        config_data = {"data": [], "pitch_angles": [], "yaw_angles": []}
        sim_num = len(pitch_yaw_angles)
        iter = 0
        for pitch_angle, yaw_angle in pitch_yaw_angles:
            pitch_angle = int(pitch_angle)  # Cast to integer
            yaw_angle = int(yaw_angle)  # Cast to integer
            # Compute relative wind velocity vector
            R_yaw = R.from_euler("z", yaw_angle, degrees=True)
            R_pitch = R.from_euler("y", pitch_angle - 90, degrees=True)
            A_R_b = R_yaw * R_pitch
            b_R_A = A_R_b.inv().as_matrix()
            wind_velocity = b_R_A[:, 0] * 17.0
            # Read simulation file
            sim_file = list(
                data_path.rglob(f"{config_name}-{pitch_angle}-{yaw_angle}-robot.dtbs")
            )[0]
            raw_data = pd.read_csv(sim_file, sep="\s+", skiprows=1, header=None)
            node_pos = raw_data.values[:, 1:4]
            node_pressure = raw_data.values[:, 4]
            node_pressure_coefficient = node_pressure / (0.5 * 1.225 * 17**2)
            node_shear_stress = raw_data.values[:, 5:8]
            node_friction_coefficient = node_shear_stress / (0.5 * 1.225 * 17**2)
            face_vectors = raw_data.values[:, 9:12]
            # Normalize face normals
            face_areas = np.linalg.norm(face_vectors, axis=1, keepdims=True)
            face_normals = face_vectors / face_areas
            # Reshape data
            wind_velocities = np.tile(wind_velocity, (node_pos.shape[0], 1))
            joint_pos_matrix = np.tile(joint_pos, (node_pos.shape[0], 1))
            # Assemble single simulation data
            sim_data = np.hstack(
                (
                    wind_velocities,
                    joint_pos_matrix,
                    node_pos,
                    face_normals,
                    node_pressure_coefficient.reshape(-1, 1),
                    node_friction_coefficient,
                    face_areas.reshape(-1, 1),
                )
            )
            # Append data to current configuration data
            config_data["data"].append(sim_data.astype(np.float32))
            config_data["pitch_angles"].append(pitch_angle)
            config_data["yaw_angles"].append(yaw_angle)
            # Print progress
            iter += 1
            print(
                f"{config_name} configuration progress: {iter}/{sim_num}",
                end="\r",
                flush=True,
            )
        # Save compressed dataset using compressed numpy
        np.savez_compressed(
            dataset_dir / f"ironcub-{config_name}.npz", data=config_data
        )
        print(f"Dataset for {config_name} configuration saved.")
        # Append config_data to the global dataset
        data.extend(config_data)
    # Save compressed dataset using compressed numpy
    np.savez_compressed(dataset_dir / "ironcub-full.npz", data=data)
    print("Full dataset saved.")


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
    # Remove duplicates
    pitch_yaw_angles = np.unique(pitch_yaw_angles, axis=0)
    return pitch_yaw_angles


if __name__ == "__main__":
    main()
