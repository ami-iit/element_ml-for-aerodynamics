# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
from src.robot import Robot
from src.flow import FlowImporter, FlowVisualizer
import pandas as pd


def main():
    # Initialize robot and flow objects
    robot_name = "iRonCub-Mk3"
    # urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    urdf_path = r"C:\Users\apaolino\code\ironcub-software-ws\src\component_ironcub\models\iRonCub-Mk3\iRonCub\robots\iRonCub-Mk3\model.urdf"
    robot = Robot(robot_name, urdf_path)
    flow = FlowImporter()

    # Load forces dataset
    forces_path = r"C:\Users\apaolino\code\element_ml-for-aerodynamics\data\generate_image_dataset\dataset\outputParameters.csv"
    forces_data = np.genfromtxt(forces_path, delimiter=",", dtype=str)
    surface_list = [forces_data[0, i][:-3] for i in range(6, len(forces_data[0]), 3)]
    surface_list = ["ironcub_" + surface.replace("-", "_") for surface in surface_list]
    forces_data = forces_data[1:, :]
    # Joint configurations
    config_path = r"C:\Users\apaolino\code\element_ml-for-aerodynamics\data\generate_image_dataset\dataset\joint-configurations.csv"
    joint_configs = np.genfromtxt(config_path, delimiter=",", dtype=str)

    data_path = r"C:\Users\apaolino\code\element_ml-for-aerodynamics\data\generate_image_dataset\dataset"
    data_path = Path(str(data_path).strip())
    for i in range(len(forces_data)):
        config_name = forces_data[i, 0]
        pitch = int(float(forces_data[i, 1]))
        yaw = int(float(forces_data[i, 2]))
        joint_pos = (
            joint_configs[joint_configs[:, 0] == config_name][0, 1:].astype(float)
            * np.pi
            / 180
        )
        glob_force_cfd = forces_data[i, 3:6].astype(float)
        loc_forces_cfd = forces_data[i, 6:].astype(float).reshape((-1, 3))
        glob_force_cfd = glob_force_cfd * 0.5 * 1.225 * (17**2)
        loc_forces_cfd = loc_forces_cfd * 0.5 * 1.225 * (17**2)

        robot.set_state(pitch, yaw, joint_pos)
        link_H_world_dict = robot.compute_all_link_H_world()
        flow.import_raw_fluent_data(
            data_path, config_name, pitch, yaw, robot.surface_list
        )
        flow.transform_local_fluent_data(
            link_H_world_dict, flow_velocity=17.0, flow_density=1.225
        )
        flow.assign_global_fluent_data()

        flow.compute_forces(air_density=1.225, flow_velocity=17.0)

        loc_forces_im = np.zeros_like(loc_forces_cfd)
        glob_force_im = np.zeros(3)
        for idx, name in enumerate(surface_list):
            f = flow.surface[name].global_force
            aero_frame_force = np.array([-f[2], f[1], -f[0]])
            loc_forces_im[idx, :] = aero_frame_force
            glob_force_im += aero_frame_force

        abs_delta_forces = abs(loc_forces_cfd - loc_forces_im)
        rel_delta_forces = abs_delta_forces / abs(loc_forces_cfd) * 100

        print("Total force Fluent: ", glob_force_cfd)
        print("Total force computed: ", glob_force_im)
        print("Local forces absolute error:", abs_delta_forces)
        print("Local forces relative error:", rel_delta_forces)

        flowViz = FlowVisualizer(flow)
        flowViz.plot_surface_pointcloud(
            flow_variable=flow.cp, robot_meshes=robot.load_mesh()
        )

    print("wow")


if __name__ == "__main__":
    main()
