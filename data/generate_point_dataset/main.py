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
from pathlib import Path

import src.functions as fn

WIND_INTENSITY = 17.0  # Fixed simulation wind intensity
DYN_PRESSURE = 0.5 * 1.225 * WIND_INTENSITY**2  # Dynamic pressure at S/L 17 m/s


def main():
    # Create a dataset output directory
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the simulation data
    data_dir = input("Enter the path to the fluent dtbs directory: ")
    data_path = Path(str(data_dir).strip())

    # Get configuration names and joint configurations
    files = [file.name for file in data_path.rglob("*.dtbs") if file.is_file()]
    configs = sorted(list(set([file.split("-")[0] for file in files])))
    joint_config_file = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file, delimiter=",", dtype=str)

    # Start the cycle on configurations
    data = []
    for config in configs:
        joint_pos = joint_configs[joint_configs[:, 0] == config][0, 1:].astype(float)
        pitch_yaw_angles = fn.find_pitch_yaw_angles(config, files)
        config_data = {"data": [], "pitch_angles": [], "yaw_angles": []}

        # Start the cycle on simulations
        sim_num = len(pitch_yaw_angles)
        for idx, pitch_yaw_angle in enumerate(pitch_yaw_angles):
            pitch = int(pitch_yaw_angle[0])
            yaw = int(pitch_yaw_angle[1])

            # Compute relative wind velocity vector
            wind_velocity = fn.compute_wind_velocity(pitch, yaw, WIND_INTENSITY)

            # Read simulation file
            sim_file = list(data_path.rglob(f"{config}-{pitch}-{yaw}-robot.dtbs"))[0]
            raw_data = pd.read_csv(sim_file, sep="\s+", skiprows=1, header=None)

            # Import data
            node_pos = raw_data.values[:, 1:4]
            face_normals = raw_data.values[:, 9:12]
            press_coeff = raw_data.values[:, 4] / DYN_PRESSURE
            fric_coeff = raw_data.values[:, 5:8] / DYN_PRESSURE

            # Transform data
            face_areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = face_normals / face_areas
            wind_velocities = np.tile(wind_velocity, (node_pos.shape[0], 1))
            joint_pos_mat = np.tile(joint_pos, (node_pos.shape[0], 1))

            # Assemble single simulation data
            sim_data = np.hstack(
                (
                    wind_velocities,
                    joint_pos_mat,
                    node_pos,
                    face_normals,
                    press_coeff.reshape(-1, 1),
                    fric_coeff,
                    face_areas.reshape(-1, 1),
                )
            )

            # Append data to current configuration data
            config_data["data"].append(sim_data.astype(np.float32))
            config_data["pitch_angles"].append(pitch)
            config_data["yaw_angles"].append(yaw)

            # Print progress
            print(
                f"{config} configuration progress: {idx+1}/{sim_num}",
                end="\r",
                flush=True,
            )

        # Save compressed dataset using compressed numpy
        np.savez_compressed(dataset_dir / f"ironcub-{config}.npz", data=config_data)
        print(f"Dataset for {config} configuration saved.")

        # Append config_data to the global dataset
        data.extend(config_data)

    # Save compressed dataset using compressed numpy
    np.savez_compressed(dataset_dir / "ironcub-full.npz", data=data)
    print("Full dataset saved.")


if __name__ == "__main__":
    main()
