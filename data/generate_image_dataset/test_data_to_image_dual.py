"""
Author: Antonello Paolino
Date: 2025-02-20
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
from src.dual_flow import FlowImporter, FlowVisualizer


SAVE_IMAGE = False


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    mesh_robot = Robot(robot_name, urdf_path)
    robot = Robot(robot_name, urdf_path)

    # Initialize flow object
    flow = FlowImporter()

    # Get the path to the dataset
    data_dir = input("Enter the path to the fluent cell data directory: ")
    data_path = Path(str(data_dir).strip())
    config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(config_file_path, delimiter=",", dtype=str)

    # Define map directory
    map_dir = Path(__file__).parents[0] / "maps"

    # Define robot state
    pitch = 30
    yaw = 0
    config_name = "hovering"
    airspeed = 17.0
    air_dens = 1.225

    # Set robot state and get link to world transformations
    joint_pos = (
        joint_configs[joint_configs[:, 0] == config_name][0, 1:].astype(float)
        * np.pi
        / 180
    )
    robot.set_state(pitch, yaw, joint_pos)
    link_H_world_dict = robot.compute_all_link_H_world()
    mesh_robot.set_state(0, 0, joint_pos)
    mesh_link_H_world_dict = mesh_robot.compute_all_link_H_world()

    # Import mesh mapping data
    map_file = list(map_dir.rglob(f"{config_name}-dual-map.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow.import_mesh_mapping_data(map_data, robot.surface_list, mesh_link_H_world_dict)

    # Import fluent data from all surfaces
    flow.import_data(data_path, config_name, pitch, yaw)
    flow.transform_data(link_H_world_dict, airspeed, air_dens)
    flow.reorder_data()
    flow.assign_global_data()

    # Interpolate and generate images
    flow.interp_3d_to_image(robot.image_resolution)

    if SAVE_IMAGE:
        print("Saving assembled image ...")
        image_dir = Path(__file__).parents[0] / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        print("Saving images ...")
        np.save(
            str(image_dir / f"{config_name}-{pitch}-{yaw}.npy"),
            flow.image,
        )

    # for s_data in flow.surface.values():
    #     s_data.press_coeff = s_data.l_nodes[:, 0]

    ##############################################################################################
    ################################# Plots and 3D visualization #################################
    ##############################################################################################

    # 3D visualization of the pressure map
    # flowViz = FlowVisualizer(flow)
    # flowViz.plot_pressure_pointcloud(robot_meshes=robot.load_mesh())

    # Enable LaTeX text rendering
    plt.rcParams["text.usetex"] = True

    # Plot 2D pressure map for all surfaces
    for s_name, s_data in flow.surface.items():
        # plot 2D pressure map
        fig = plt.figure(f"{s_name[8:]}")
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        ax = fig.add_subplot(121)
        ax.scatter(
            s_data.map[:, 1],
            s_data.map[:, 0],
            c=s_data.press_coeff,
            s=5,
            cmap="jet",
            vmax=1,
            vmin=-2,
        )
        ax.set_title(f"{s_name[8:]} original points")
        ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_xlabel(r"$\psi$ [rad]")
        # plot interpolated image
        ax = fig.add_subplot(122)
        image = s_data.image[0, :, :]
        im_plot = ax.imshow(image, origin="lower", cmap="jet", vmax=1, vmin=-2)
        ax.set_title(f"{s_name[8:]} interpolated image")
        ax.set_xlim([-10, image.shape[1] + 10])
        ax.set_ylim([-10, image.shape[0] + 10])
        cbar = fig.colorbar(im_plot)
        cbar.set_label(r"C_p")
    plt.show()

    # Plot 2D pressure map for all surfaces (together)
    fig = plt.figure("2D Pressure Map")
    gs = gridspec.GridSpec(5, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1, 0.1])
    last_plot = None
    for i, (s_name, s_data) in enumerate(flow.surface.items()):
        row, col = divmod(i, 6)
        ax = fig.add_subplot(gs[row, col])
        last_plot = ax.scatter(
            s_data.map[:, 0],
            s_data.map[:, 1],
            c=s_data.press_coeff,
            s=1,
            cmap="jet",
            vmax=1,
            vmin=-2,
        )
        ax.set_title(s_name[8:])
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.set_ylabel(r"$\psi$ [rad]")
        # ax.set_xlim([-np.pi, np.pi])
        # ax.set_ylim([0, np.pi])
    cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
    cbar = fig.colorbar(last_plot, cax=cbar_ax)
    cbar.set_label(r"C_p")
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.8, hspace=0.6
    )
    plt.show()

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
