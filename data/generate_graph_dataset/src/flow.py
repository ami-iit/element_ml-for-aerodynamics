import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
import open3d as o3d
from matplotlib.colors import Normalize
from matplotlib import cm


@dataclass
class SurfaceData:
    w_nodes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    l_nodes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    w_shear_stress: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    press_coeff: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    w_fric_coeff: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    w_area_vector: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    l_area_vector: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))


class FlowImporter:
    def __init__(self):
        self.surface = {}

    def import_mesh_graph(self, data, surface_list, link_H_world_dict):
        for s_name in surface_list:
            self.surface[s_name] = SurfaceData()
        for s_name, s_data in self.surface.items():
            s_data.faces = data[s_name]["faces"]
            s_data.edges = data[s_name]["edges"]
            s_data.w_mesh_nodes = data[s_name]["nodes"] / 1000
            # mesh nodes are stored in local reference frame
            link_H_world = link_H_world_dict[s_name]
            s_data.l_mesh_nodes = self.transform_points(
                link_H_world, s_data.w_mesh_nodes
            )
        return

    def import_data(self, data_path, config_name, pitch, yaw):
        for s_name, s_data in self.surface.items():
            filename = f"{config_name}-{pitch}-{yaw}-{s_name}.dtbs"
            filepath = list(data_path.rglob(filename))[0]
            data = pd.read_csv(filepath, sep="\s+", skiprows=1, header=None)
            s_data.w_nodes = data.values[:, 1:4]
            s_data.pressure = data.values[:, 4]
            s_data.w_friction = data.values[:, 5:8]
            s_data.w_face_areas = data.values[:, 9:12]
        return

    def transform_data(self, link_H_world_dict, base_H_world, airspeed, air_dens):
        for s_name, s_data in self.surface.items():
            link_H_world = link_H_world_dict[s_name]
            # transform the data from world to reference link frame
            s_data.l_nodes = self.transform_points(link_H_world, s_data.w_nodes)
            # transform the data from world to base frame
            s_data.b_nodes = self.transform_points(base_H_world, s_data.w_nodes)
            s_data.b_friction = self.rotate_vectors(base_H_world, s_data.w_friction)
            s_data.b_face_areas = self.rotate_vectors(base_H_world, s_data.w_face_areas)
            # Transform variables to coefficients
            dyn_press = 0.5 * air_dens * airspeed**2
            s_data.press_coeff = s_data.pressure / dyn_press
            s_data.w_fric_coeff = s_data.w_friction / dyn_press
            s_data.b_fric_coeff = s_data.b_friction / dyn_press
            s_data.areas = np.linalg.norm(s_data.w_face_areas, axis=1)
            s_data.w_face_normals = s_data.w_face_areas / s_data.areas[:, None]
            s_data.b_face_normals = s_data.b_face_areas / s_data.areas[:, None]
        return

    def transform_points(self, frame_1_H_frame_2, points):
        ones = np.ones((len(points), 1))
        start_coord = np.hstack((points, ones))
        end_coord = np.dot(frame_1_H_frame_2, start_coord.T).T
        return end_coord[:, :3]

    def rotate_vectors(self, frame_1_H_frame_2, vectors):
        frame_1_R_frame_2 = frame_1_H_frame_2[:3, :3]
        rotated_vectors = np.dot(frame_1_R_frame_2, vectors.T).T
        return rotated_vectors

    def reorder_data(self):
        for s_data in self.surface.values():
            # Reorder data nodes to match mesh_nodes order
            indices = []
            for node in s_data.l_mesh_nodes:
                distances = np.linalg.norm(s_data.l_nodes - node, axis=1)
                indices.append(np.argmin(distances))
            s_data.l_nodes = s_data.l_nodes[indices]
            s_data.w_nodes = s_data.w_nodes[indices]
            s_data.b_nodes = s_data.b_nodes[indices]
            s_data.pressure = s_data.pressure[indices]
            s_data.press_coeff = s_data.press_coeff[indices]
            s_data.w_friction = s_data.w_friction[indices]
            s_data.b_friction = s_data.b_friction[indices]
            s_data.w_fric_coeff = s_data.w_fric_coeff[indices]
            s_data.b_fric_coeff = s_data.b_fric_coeff[indices]
            s_data.areas = s_data.areas[indices]
            s_data.w_face_normals = s_data.w_face_normals[indices]
            s_data.b_face_normals = s_data.b_face_normals[indices]
            s_data.edges = s_data.edges[indices]
        return

    def assign_global_data(self):
        self.w_nodes = np.empty(shape=(0, 3))
        self.b_nodes = np.empty(shape=(0, 3))
        self.press_coeff = np.empty(shape=(0,))
        self.w_fric_coeff = np.empty(shape=(0, 3))
        self.b_fric_coeff = np.empty(shape=(0, 3))
        self.areas = np.empty(shape=(0, 1))
        self.w_face_normals = np.empty(shape=(0, 3))
        self.b_face_normals = np.empty(shape=(0, 3))
        self.edges = np.empty(shape=(0, 2))
        for s_data in self.surface.values():
            edge_bias = len(self.w_nodes)
            self.w_nodes = np.append(self.w_nodes, s_data.w_nodes, axis=0)
            self.b_nodes = np.append(self.b_nodes, s_data.b_nodes, axis=0)
            self.press_coeff = np.append(self.press_coeff, s_data.press_coeff)
            self.w_fric_coeff = np.append(
                self.w_fric_coeff, s_data.w_fric_coeff, axis=0
            )
            self.b_fric_coeff = np.append(
                self.b_fric_coeff, s_data.b_fric_coeff, axis=0
            )
            self.areas = np.append(self.areas, s_data.areas[:, None], axis=0)
            self.w_face_normals = np.append(
                self.w_face_normals, s_data.w_face_normals, axis=0
            )
            self.b_face_normals = np.append(
                self.b_face_normals, s_data.b_face_normals, axis=0
            )
            self.edges = np.append(self.edges, s_data.edges + edge_bias, axis=0)
        return

    def visualize_pointcloud(self, points, values, window_name="Open3D"):
        # Normalize the colormap
        norm = Normalize(vmin=-2, vmax=1)
        normalized_flow_variable = norm(values)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create a point cloud from the dataset
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        # Set visualization parameters
        zoom = 0.75
        x_cen = (points[:, 0].max() + points[:, 0].min()) / 2
        y_cen = (points[:, 1].max() + points[:, 1].min()) / 2
        z_cen = (points[:, 2].max() + points[:, 2].min()) / 2
        center = [x_cen, y_cen, z_cen]
        front = [1.0, 0.0, 1.0]
        up = [0.0, 1.0, 0.0]
        # Display the pointcloud
        o3d.visualization.draw_geometries(
            [frame, pcd],
            zoom=zoom,
            lookat=center,
            front=front,
            up=up,
            window_name=window_name,
        )
