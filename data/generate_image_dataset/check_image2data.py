"""
Author: Antonello Paolino
Date: 2024-05-29
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates 3D data of flow variables extracted from 2D images
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
from src.robot import Robot
from src.flow import FlowGenerator, FlowVisualizer


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)

    # Initialize flow object
    flow = FlowGenerator()

    # Get the path to the dataset
    data_dir = input("Enter the path to the fluent data directory: ")
    fluent_data_path = Path(str(data_dir).strip())

    # Define robot state parameters
    pitch_angle = 30
    yaw_angle = 0
    joint_positions = (
        np.array(
            [
                0,
                0,
                0,
                -30.7,
                12.9,
                26.5,
                58.3,
                -30.7,
                12.9,
                26.5,
                58.3,
                0,
                10,
                0,
                0,
                0,
                10,
                0,
                0,
            ]
        )
        * np.pi
        / 180
    )

    # Set robot state
    robot.set_state(pitch_angle, yaw_angle, joint_positions)

    ###############################################################################################################
    ####################### HERE THERE SHOULD BE THE ALGORITHM TO GENERATE THE LOCAL IMAGES #######################
    ###############################################################################################################
    # Load image data
    joint_config_name_loading = "flight30"
    pitch_angle_loading = 30
    yaw_angle_loading = 0
    project_directory = Path(__file__).parents[0]
    image_directory = project_directory / "images"
    predicted_image = np.load(
        image_directory
        / f"{joint_config_name_loading}-{pitch_angle_loading}-{yaw_angle_loading}.npy"
    )
    ###############################################################################################################
    ###############################################################################################################

    # Separate the image into the 2D images of the surfaces
    flow.separate_images(predicted_image, robot.surface_list)

    # Mesh_robot for importing the pointcloud mesh
    robot_ref = Robot(robot_name, urdf_path)
    pitch_angle_mesh = 0
    yaw_angle_mesh = 0
    joint_config_mesh = "hovering"
    joint_pos_mesh = (
        np.array([0, 0, 0, 0, 16.6, 40, 15, 0, 16.6, 40, 15, 0, 10, 7, 0, 0, 10, 7, 0])
        * np.pi
        / 180
    )

    robot_ref.set_state(pitch_angle_mesh, yaw_angle_mesh, joint_pos_mesh)
    world_H_link_dict = robot.compute_all_world_H_link()
    link_H_world_ref_dict = robot_ref.compute_all_link_H_world()

    flow.interpolate_flow_data_from_image(
        fluent_data_path,
        robot.surface_list,
        robot.surface_axes,
        link_H_world_ref_dict,
        world_H_link_dict,
        joint_config_mesh,
        pitch_angle_mesh,
        yaw_angle_mesh,
    )

    flow.compute_forces(air_density=1.225, flow_velocity=17.0)

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

    # Display the imported image
    fig1 = plt.figure("Imported Image")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    ax1 = fig1.add_subplot(1, 1, 1)
    image = ax1.imshow(flow.image[0, :, :], origin="upper", cmap="jet", vmax=1, vmin=-2)
    ax1.axis("off")
    fig1.colorbar(image, ax=ax1, orientation="vertical", fraction=0.02, pad=0.45)
    plt.show(block=False)

    # Display the 2D separated images
    fig2 = plt.figure("Separated Images")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    gs = gridspec.GridSpec(
        4, 7, figure=1, width_ratios=[1, 1, 1, 1, 1, 1, 0.1]
    )  # The last column is for the colorbar
    last_im = None
    for surface_index, surface_name in enumerate(robot.surface_list):
        ax2 = fig2.add_subplot(gs[surface_index // 6, surface_index % 6])
        last_im = ax2.imshow(
            flow.surface[surface_name].image[0, :, :],
            origin="lower",
            cmap="jet",
            vmax=1,
            vmin=-2,
        )
        ax2.set_title(surface_name[8:])
        ax2.set_xlim([-10, flow.surface[surface_name].image.shape[2] + 10])
        ax2.set_ylim([-10, flow.surface[surface_name].image.shape[1] + 10])
    cbar_ax = fig2.add_subplot(gs[:, -1])  # Span all rows in the last column
    cbar = fig2.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(r"$C_p$")
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.show(block=False)

    # Plot 2D pressure map for all surfaces
    fig3 = plt.figure("2D Reconstructed Pressure Maps")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    gs = gridspec.GridSpec(
        4, 7, figure=1, width_ratios=[1, 1, 1, 1, 1, 1, 0.1]
    )  # The last column is for the colorbar
    last_plot = None
    for surface_index, surface_name in enumerate(robot.surface_list):
        ax3 = fig3.add_subplot(gs[surface_index // 6, surface_index % 6])
        last_plot = ax3.scatter(
            flow.surface[surface_name].theta,
            flow.surface[surface_name].z,
            c=flow.surface[surface_name].pressure_coefficient,
            s=1,
            cmap="jet",
            vmax=1,
            vmin=-2,
        )
        ax3.set_title(robot.surface_list[surface_index][8:])
        ax3.set_xlabel(r"$\theta r_{mean}$ [m]")
        ax3.set_ylabel(r"$z$ [m]")
        ax3.axis("equal")
        ax3.set_xlim(
            [
                np.min(flow.surface[surface_name].theta),
                np.max(flow.surface[surface_name].theta),
            ]
        )
        ax3.set_ylim(
            [np.min(flow.surface[surface_name].z), np.max(flow.surface[surface_name].z)]
        )
    cbar_ax = fig3.add_subplot(gs[:, -1])  # Span all rows in the last column
    cbar = fig3.colorbar(last_plot, cax=cbar_ax)
    cbar.set_label(r"$C_p$")
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.8, hspace=0.6
    )
    plt.show(block=False)

    input("Press Enter to close all figures...")


if __name__ == "__main__":
    main()
