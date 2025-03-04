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
from src.flow_new import FlowImporter, FlowGenerator, FlowVisualizer
import open3d as o3d
from matplotlib.pyplot import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tabulate import tabulate


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    # urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    urdf_path = r"C:\Users\apaolino\code\ironcub-software-ws\src\component_ironcub\models\iRonCub-Mk3\iRonCub\robots\iRonCub-Mk3\model.urdf"
    mesh_robot = Robot(robot_name, urdf_path)
    robot = Robot(robot_name, urdf_path)
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Define map directory
    map_dir = Path(__file__).parents[0] / "maps"
    # Get the path to the raw data
    # data_dir = input("Enter the path to the fluent data directory: ")
    # data_path = Path(str(data_dir).strip())
    node_data_path = Path(r"C:\Users\apaolino\code\datasets\mk3-cfd-aero\node-data")
    cell_data_path = Path(r"C:\Users\apaolino\code\datasets\mk3-cfd-aero\cell-data")
    file_names = [
        file.name for file in node_data_path.rglob("*.dtbs") if file.is_file()
    ]
    config_names = sorted(
        list(set([file_name.split("-")[0] for file_name in file_names]))
    )
    config_file_path = list(node_data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(config_file_path, delimiter=",", dtype=str)

    for config_name in config_names:
        joint_pos = joint_configs[joint_configs[:, 0] == config_name][0, 1:].astype(
            float
        )
        joint_pos *= np.pi / 180
        # Import and set mesh mapping
        map_file = list(map_dir.rglob(f"{config_name}-nodal-map.npy"))[0]
        map_data = np.load(map_file, allow_pickle=True).item()
        mesh_robot.set_state(0, 0, joint_pos)
        mesh_link_H_world_dict = mesh_robot.compute_all_link_H_world()
        flow_in.import_mesh_mapping_data(
            map_data, robot.surface_list, mesh_link_H_world_dict
        )
        flow_out.import_mesh_mapping_data(
            map_data, robot.surface_list, mesh_link_H_world_dict
        )
        flow_out.load_mesh_cells(cell_data_path, mesh_link_H_world_dict)
        flow_out.reorder_cell_data()
        flow_out.compute_interpolator(robot.image_resolutions)
        # Set pitch and yaw angles
        pitch_yaw_angles = find_pitch_yaw_angles(config_name, file_names)
        aero_force_abs_err = np.zeros((len(pitch_yaw_angles), 3))
        aero_force_rel_err = np.zeros((len(pitch_yaw_angles), 3))
        for idx, pitch_yaw in enumerate(pitch_yaw_angles[:2]):
            pitch = int(pitch_yaw[0])
            yaw = int(pitch_yaw[1])
            # Set robot state and get link to world transformations
            robot.set_state(pitch, yaw, joint_pos)
            link_H_world = robot.compute_all_link_H_world()
            # Import fluent data from all surfaces
            flow_in.import_node_data(node_data_path, config_name, pitch, yaw)
            flow_in.import_cell_data(cell_data_path, config_name, pitch, yaw)
            flow_in.transform_local_data(link_H_world, airspeed=17.0, air_dens=1.225)
            flow_in.reorder_surface_data()
            flow_in.assign_global_fluent_data()
            # Data Interpolation and Image Generation
            flow_in.interp_3d_to_image(robot.image_resolutions)
            flow_in.compute_cell_forces()
            # Cell force computation
            in_tot_aero_force_cell = flow_in.w_aero_force
            in_aero_forces_cell = np.concatenate(
                [data.w_aero_force[np.newaxis, :] for data in flow_in.surface.values()],
                axis=0,
            )

            # Node force computations
            flow_in.compute_cell_values()
            flow_in.compute_cell_forces()
            in_tot_aero_force_node = flow_in.w_aero_force
            in_aero_forces_node = np.concatenate(
                [data.w_aero_force[np.newaxis, :] for data in flow_in.surface.values()],
                axis=0,
            )

            # Data reconstruction
            image = flow_in.image
            world_H_link_dict = robot.compute_all_world_H_link()
            flow_out.transform_mesh_cells(world_H_link_dict)
            flow_out.separate_images(image)
            flow_out.interpolate_flow_data()
            flow_out.compute_cell_values()
            flow_out.compute_cell_forces(airspeed=17.0, air_dens=1.225)
            out_tot_aero_force_node = flow_out.w_aero_force
            out_aero_forces = np.concatenate(
                [
                    data.w_aero_force[np.newaxis, :]
                    for data in flow_out.surface.values()
                ],
                axis=0,
            )

            # Generate total aerodynamic force output
            tot_force_out = [
                ["Input total aero force (cell)"] + in_tot_aero_force_cell.tolist(),
                ["Input total aero force (node)"] + in_tot_aero_force_node.tolist(),
                ["Output total aero force (node)"] + out_tot_aero_force_node.tolist(),
            ]
            headers = ["", "x", "y", "z"]
            print(tabulate(tot_force_out, headers=headers, tablefmt="pretty"))

            # generate local aerodynamic force output on cell level
            abs_err_cell = np.abs(in_aero_forces_cell - out_aero_forces)
            rel_err_cell = abs_err_cell / np.abs(in_aero_forces_cell) * 100

            loc_force_out_1 = [
                [name[8:]] + abs_err_cell[idx].tolist() + rel_err_cell[idx].tolist()
                for idx, name in enumerate(flow_in.surface.keys())
            ]
            header = ["surface", "x", "y", "z", "x_rel", "y_rel", "z_rel"]
            print(f"Local errors (cell)")
            print(tabulate(loc_force_out_1, headers=header, tablefmt="pretty"))

            # generate local aerodynamic force output on node level
            abs_err_node = np.abs(in_aero_forces_node - out_aero_forces)
            rel_err_node = abs_err_node / np.abs(in_aero_forces_node) * 100

            loc_force_out_2 = [
                [name[8:]] + abs_err_node[idx].tolist() + rel_err_node[idx].tolist()
                for idx, name in enumerate(flow_in.surface.keys())
            ]
            header = ["surface", "x", "y", "z", "x_rel", "y_rel", "z_rel"]
            print(f"Local errors (node)")
            print(tabulate(loc_force_out_2, headers=header, tablefmt="pretty"))

            aero_force_abs_err[idx] = np.sum(abs_err_node, axis=0)
            aero_force_rel_err[idx] = (
                aero_force_abs_err[idx]
                / np.abs(np.sum(in_aero_forces_node, axis=0))
                * 100
            )

            # plot_2d_pressure_map_errors(flow_in, flow_out)
            # plot_diff_pressure_pointcloud(flow_in, flow_out)

            print(
                f"{config_name} configuration progress: {idx+1}/{pitch_yaw_angles.shape[0]}",
                end="\r",
                flush=True,
            )

        plot_scatter_force_errors(pitch_yaw_angles, aero_force_abs_err, "absolute")
        plot_scatter_force_errors(pitch_yaw_angles, aero_force_rel_err, "relative")

        print("check")


def plot_scatter_force_errors(pitch_yaw_angles, aero_force_abs_err, err_type):
    cbar_label = r"abs error [N]" if err_type == "absolute" else r"rel error [%]"
    ax_titles = ["x", "y", "z"]

    fig, axes = plt.subplots(1, 3)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    for i in range(len(axes)):
        ax = axes[i]
        scatter_plot = ax.scatter(
            pitch_yaw_angles[:, 0],
            pitch_yaw_angles[:, 1],
            c=aero_force_abs_err[:, i],
            s=20,
            cmap="jet",
        )
        ax.set_title(f"{ax_titles[i]} force errors")
        ax.set_ylabel(r"yaw [deg]") if i == 0 else None
        ax.set_xlabel(r"pitch [deg]")
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.6, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(scatter_plot, cax=cax, orientation="horizontal")
        cbar.set_label(cbar_label)

    plt.show()


def plot_2d_pressure_map_errors(flow_in, flow_out):
    for s_name in flow_in.surface.keys():
        s_data_in = flow_in.surface[s_name]
        s_data_out = flow_out.surface[s_name]
        # plot 2D pressure map error
        fig = plt.figure(f"{s_name[8:]} abs error")
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        ax = fig.add_subplot(121)
        scatter_plot = ax.scatter(
            s_data_in.map[:, 1],
            s_data_in.map[:, 0],
            c=abs(s_data_in.press_coeff - s_data_out.press_coeff),
            s=5,
            cmap="jet",
            # vmax=1,
            # vmin=-2,
        )
        # ax.set_xticks(np.linspace(0, np.pi, 256))
        # ax.set_yticks(np.linspace(-np.pi, np.pi, 256))
        # ax.grid(which="both")
        ax.set_title(f"{s_name[8:]} original points abs error")
        ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_xlabel(r"$\psi$ [rad]")
        cbar = fig.colorbar(scatter_plot)
        cbar.set_label(r"C_p")
        # plot interpolated image
        ax = fig.add_subplot(122)
        image = s_data_in.image[0, :, :]
        im_plot = ax.imshow(image, origin="lower", cmap="jet", vmax=1, vmin=-2)
        ax.set_title(f"{s_name[8:]} interpolated image")
        ax.set_xlim([-10, image.shape[1] + 10])
        ax.set_ylim([-10, image.shape[0] + 10])
        cbar = fig.colorbar(im_plot)
        cbar.set_label(r"C_p")
    plt.show()


def plot_diff_pressure_pointcloud(flow_in, flow_out):
    # Create the point cloud
    var = abs(flow_in.cp - flow_out.cp)
    norm = Normalize(vmin=min(var), vmax=max(var))
    print(f"Min: {min(var)}")
    print(f"Max: {max(var)}")
    norm_var = norm(var)
    cmap = cm.jet
    colors = cmap(norm_var)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flow_in.w_nodes)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Create the global frame
    w_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    # create the relative wind direction vector
    rel_wind = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.02,
        cylinder_height=0.2,
        cone_height=0.04,
    )
    rel_wind.paint_uniform_color([0, 0.5, 1])  # green-ish blue
    rel_wind.rotate(R.from_euler("y", 180, degrees=True).as_matrix(), center=[0, 0, 0])
    rel_wind.translate([0, 0.2, 1.0])
    rel_wind.compute_vertex_normals()
    # Assemble the geometries list
    geom = [
        {"name": "point_cloud", "geometry": pcd},
        {"name": "world_frame", "geometry": w_frame},
        {"name": "wind_vector", "geometry": rel_wind},
    ]
    o3d.visualization.draw(geom, show_skybox=False, non_blocking_and_return_uid=False)


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
