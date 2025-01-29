"""
Author: Antonello Paolino
Date: 2024-05-15
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates a 2D representation fo the 3D pressure map on
                the robot component surfaces
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
from src.robot import Robot
from src.flow import FlowImporter, FlowVisualizer
import time


SAVE_IMAGE = False


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)

    # Initialize flow object
    flow = FlowImporter()

    # Get the path to the dataset
    data_dir = input("Enter the path to the fluent data directory: ")
    data_path = Path(str(data_dir).strip())
    joint_config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file_path, delimiter=",", dtype=str)
    joint_config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file_path, delimiter=",", dtype=str)

    # Define robot state
    pitch_angle = 30
    yaw_angle = 0
    joint_config_name = "flight30"

    # Set robot state and get link to world transformations
    joint_positions = (
        joint_configs[joint_configs[:, 0] == joint_config_name][0, 1:].astype(float)
        * np.pi
        / 180
    )
    robot.set_state(pitch_angle, yaw_angle, joint_positions)
    link_H_world_dict = robot.compute_all_link_H_world()

    # Import fluent data from all surfaces
    start_time = time.time()
    flow.import_raw_fluent_data(
        data_path, joint_config_name, pitch_angle, yaw_angle, robot.surface_list
    )
    flow.transform_local_fluent_data(
        link_H_world_dict, flow_velocity=17.0, flow_density=1.225
    )
    flow.assign_global_fluent_data()
    end_time = time.time()
    print(f"Time to import and transform data: {end_time - start_time}")

    # Interpolate and generate images
    start_time = time.time()
    flow.interpolate_flow_data_and_assemble_image(
        robot.image_resolutions, robot.surface_list, robot.surface_axes
    )
    end_time = time.time()
    print(f"Time to interpolate and generate images: {end_time - start_time}")

    nan_indices = np.where(np.isnan(flow.image))
    print(f"Number of NaN values: {len(nan_indices[0])}")

    if SAVE_IMAGE:
        print("Saving assembled image ...")
        image_dir = Path(__file__).parents[0] / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        print("Saving images ...")
        np.save(
            str(image_dir / f"{joint_config_name}-{pitch_angle}-{yaw_angle}.npy"),
            flow.image,
        )

    ##############################################################################################
    ################################# Plots and 3D visualization #################################
    ##############################################################################################

    # 3D visualization of the pressure map
    flowViz = FlowVisualizer(flow)
    flowViz.plot_surface_pointcloud(
        flow_variable=flow.cp, robot_meshes=robot.load_mesh()
    )
    # flowViz.plot_surface_contour(flow_variable=flow.cp, robot_meshes=robot.load_mesh())

    # Enable LaTeX text rendering
    plt.rcParams["text.usetex"] = True

    # Plot 2D pressure map for all surfaces
    fig = plt.figure("2D Pressure Map")
    gs = gridspec.GridSpec(
        4, 7, figure=1, width_ratios=[1, 1, 1, 1, 1, 1, 0.1]
    )  # The last column is for the colorbar
    last_plot = None
    for surface_index, surface_name in enumerate(robot.surface_list):
        ax = fig.add_subplot(gs[surface_index // 6, surface_index % 6])
        last_plot = ax.scatter(
            flow.surface[surface_name].theta,
            flow.surface[surface_name].z,
            c=flow.surface[surface_name].pressure_coefficient,
            s=1,
            cmap="jet",
            vmax=1,
            vmin=-2,
        )
        ax.set_title(robot.surface_list[surface_index][8:])
        ax.set_xlabel(r"$\theta r_{mean}$ [m]")
        ax.set_ylabel(r"$z$ [m]")
        ax.set_xlim(
            [
                np.min(flow.surface[surface_name].theta),
                np.max(flow.surface[surface_name].theta),
            ]
        )
        ax.set_ylim(
            [np.min(flow.surface[surface_name].z), np.max(flow.surface[surface_name].z)]
        )
    cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
    cbar = fig.colorbar(last_plot, cax=cbar_ax)
    cbar.set_label(r"C_p")
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.8, hspace=0.6
    )
    # plt.show(block=False)

    # Plot interpolated images for all surfaces
    fig = plt.figure("Interpolated Images")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    gs = gridspec.GridSpec(
        4, 7, figure=1, width_ratios=[1, 1, 1, 1, 1, 1, 0.1]
    )  # The last column is for the colorbar
    last_im = None
    for surface_index, surface_name in enumerate(robot.surface_list):
        ax = fig.add_subplot(gs[surface_index // 6, surface_index % 6])
        image = flow.surface[surface_name].image[0, :, :]
        last_im = ax.imshow(image, origin="lower", cmap="jet", vmax=1, vmin=-2)
        ax.set_title(surface_name[8:])
        ax.set_xlim([-10, image.shape[1] + 10])
        ax.set_ylim([-10, image.shape[0] + 10])
    cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(r"C_p")
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.show()

    input("Press Enter to close all figures ...")


if __name__ == "__main__":
    main()
