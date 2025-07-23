import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
import open3d as o3d
from matplotlib.colors import Normalize
from matplotlib import cm

import src.ansys as ans
import src.mesh as ms


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
        return

    def import_target_mesh(self, mesh_path, surface_list, link_H_world_dict):
        for s_name in surface_list:
            self.surface[s_name] = SurfaceData()
        self.part_start_ids = []
        self.part_end_ids = []
        start_idx = 0
        for s_name, s_data in self.surface.items():
            l_H_w = link_H_world_dict[s_name]
            # Import data from Fluent format
            filename = f"hovering-{s_name}.dlm"
            datafile = list(mesh_path.rglob(filename))[0]
            celldata = pd.read_csv(datafile, sep="\s+", skiprows=1, header=None)
            nodes = celldata.values[:, 1:4]
            normals = celldata.values[:, 4:7]
            areas = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / areas
            # Transform in dual mesh
            s_data.l_nodes = self.transform_points(l_H_w, nodes)
            s_data.areas = areas
            s_data.l_face_normals = self.rotate_vectors(l_H_w, normals)
            # Append the end index of the current surface data
            end_idx = start_idx + len(s_data.l_nodes)
            self.part_end_ids.append(end_idx)
            self.part_start_ids.append(start_idx)
            start_idx = end_idx
        return

    def import_source_mesh(self, mesh_path, config, link_H_world_dict):
        self.w_nodes_src = np.empty(shape=(0, 3))
        for s_name, s_data in self.surface.items():
            l_H_w = link_H_world_dict[s_name]
            # Import mesh from Fluent format
            filename = f"{config}-{s_name}.msh"
            meshfile = list(mesh_path.rglob(filename))[0]
            nodes, faces = ans.read_fluent_mesh_file(str(meshfile))
            # Import data from Fluent format
            filename = f"{config}-{s_name}.dlm"
            datafile = list(mesh_path.rglob(filename))[0]
            celldata = pd.read_csv(datafile, sep="\s+", skiprows=1, header=None)
            cells = celldata.values[:, 1:4]
            # Transform in dual mesh
            nodes, faces = ms.build_dual_mesh(nodes, faces, cells, s_name)
            nodes /= 1000
            s_data.faces_src = faces
            s_data.w_nodes_src = nodes
            s_data.l_nodes_src = self.transform_points(l_H_w, nodes)
            self.w_nodes_src = np.append(self.w_nodes_src, s_data.w_nodes_src, axis=0)
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

    def import_fluent_simulation_data(
        self, data_path, config, pitch, yaw, link_H_world_dict
    ):
        for s_name, s_data in self.surface.items():
            filename = f"{config}-{pitch}-{yaw}-{s_name}.dtbs"
            filepath = list(data_path.rglob(filename))[0]
            data = pd.read_csv(filepath, sep="\s+", skiprows=1, header=None)
            w_nodes_data = data.values[:, 1:4]
            l_nodes_data = self.transform_points(
                link_H_world_dict[s_name], w_nodes_data
            )
            pressure_data = data.values[:, 4]
            w_friction_data = data.values[:, 5:8]
            # Reorder data to match source mesh nodes
            indices = []
            for node in s_data.l_nodes_src:
                distances = np.linalg.norm(l_nodes_data - node, axis=1)
                indices.append(np.argmin(distances))
            s_data.pressure_src = pressure_data[indices]
            s_data.w_friction_src = w_friction_data[indices]
        return

    def interpolate_fluent_simulation_data(self, world_H_link_dict):
        """Interpolate Fluent simulation data over the target mesh nodes."""
        for s_name, s_data in self.surface.items():
            w_H_l = world_H_link_dict[s_name]
            pressure_t = np.empty(shape=(len(s_data.l_nodes),))
            w_friction_t = np.empty(shape=(len(s_data.l_nodes), 3))
            # Comput face centroids
            centroids = np.empty(shape=(0, 3))
            for face in s_data.faces_src:
                face_nodes = s_data.l_nodes_src[face]
                centroid = np.mean(face_nodes, axis=0)
                centroids = np.append(centroids, [centroid], axis=0)
            for i, node in enumerate(s_data.l_nodes):
                # Find the closest face nodes
                distances = np.linalg.norm(centroids - node, axis=1)
                face_index = np.argmin(distances)
                nodes_src = s_data.l_nodes_src[s_data.faces_src[face_index]]
                pressure_src = s_data.pressure_src[s_data.faces_src[face_index]]
                w_friction_src = s_data.w_friction_src[s_data.faces_src[face_index]]
                d = np.linalg.norm(nodes_src - node, axis=1)
                if d.min() < 1e-6:
                    idx = np.argmin(d)
                    node_src_idx = s_data.faces_src[face_index][idx]
                    pressure_t[i] = s_data.pressure_src[node_src_idx]
                    w_friction_t[i] = s_data.w_friction_src[node_src_idx]
                else:
                    k = 1 / np.sum(1 / d)
                    weights = k / d
                    pressure_t[i] = np.sum(weights * pressure_src)
                    w_friction_t[i] = np.sum(weights[:, None] * w_friction_src, axis=0)
            # Assign interpolated data to surface data in world frame
            s_data.pressure = pressure_t
            s_data.w_friction = w_friction_t
            # Assign geometric data in world frame
            s_data.w_nodes = self.transform_points(w_H_l, s_data.l_nodes)
            s_data.w_face_normals = self.rotate_vectors(w_H_l, s_data.l_face_normals)
        return

    def transform_data_to_base_link(self, b_H_w, dyn_pressure):
        self.w_nodes = np.empty(shape=(0, 3))
        self.b_nodes = np.empty(shape=(0, 3))
        self.w_face_normals = np.empty(shape=(0, 3))
        self.b_face_normals = np.empty(shape=(0, 3))
        self.areas = np.empty(shape=(0,))
        self.press_coeff = np.empty(shape=(0,))
        self.w_fric_coeff = np.empty(shape=(0, 3))
        self.b_fric_coeff = np.empty(shape=(0, 3))
        # Iterate over each surface data
        for s_data in self.surface.values():
            # Transform the data from world to base frame
            s_data.b_nodes = self.transform_points(b_H_w, s_data.w_nodes)
            s_data.b_friction = self.rotate_vectors(b_H_w, s_data.w_friction)
            s_data.b_face_normals = self.rotate_vectors(b_H_w, s_data.w_face_normals)
            # Transform variables to coefficients
            s_data.press_coeff = s_data.pressure / dyn_pressure
            s_data.w_fric_coeff = s_data.w_friction / dyn_pressure
            s_data.b_fric_coeff = s_data.b_friction / dyn_pressure
            # Append data to the global arrays
            self.w_nodes = np.append(self.w_nodes, s_data.w_nodes, axis=0)
            self.b_nodes = np.append(self.b_nodes, s_data.b_nodes, axis=0)
            self.w_face_normals = np.append(
                self.w_face_normals, s_data.w_face_normals, axis=0
            )
            self.b_face_normals = np.append(
                self.b_face_normals, s_data.b_face_normals, axis=0
            )
            self.areas = np.append(self.areas, s_data.areas)
            self.press_coeff = np.append(self.press_coeff, s_data.press_coeff)
            self.w_fric_coeff = np.append(
                self.w_fric_coeff, s_data.w_fric_coeff, axis=0
            )
            self.b_fric_coeff = np.append(
                self.b_fric_coeff, s_data.b_fric_coeff, axis=0
            )
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
