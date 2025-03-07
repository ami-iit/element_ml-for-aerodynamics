import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib.pyplot import Normalize
from matplotlib import cm
from scipy.interpolate import (
    griddata,
    RegularGridInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field


@dataclass
class SurfaceData:
    w_nodes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    l_nodes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    w_shear_stress: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    press_coeff: np.ndarray = field(default_factory=lambda: np.empty(shape=(0,)))
    w_fric_coeff: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    image: np.ndarray = field(default_factory=lambda: np.empty(shape=(4, 0, 0)))
    w_area_vector: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    l_area_vector: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))
    w_force: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 3)))


class FlowImporter:
    def __init__(self):
        self.surface = {}

    def import_mesh_mapping_data(self, data, surface_list, link_H_world_dict):
        for s_name in surface_list:
            self.surface[s_name] = SurfaceData()
        for s_name, s_data in self.surface.items():
            s_data.map = data[s_name]["map"]
            s_data.mesh_faces = data[s_name]["faces"]
            w_mesh_nodes = data[s_name]["nodes"] / 1000
            link_H_world = link_H_world_dict[s_name]
            # mesh nodes are stored in local reference frame
            s_data.mesh_nodes = self.transform_points(link_H_world, w_mesh_nodes)
        return

    def import_data(self, data_path, config_name, pitch, yaw):
        for s_name, s_data in self.surface.items():
            filename = f"{config_name}-{pitch}-{yaw}-{s_name}.dtbs"
            filepath = list(data_path.rglob(filename))[0]
            data = pd.read_csv(filepath, sep="\s+", skiprows=1, header=None)
            s_data.w_nodes = data.values[:, 1:4]
            s_data.pressure = data.values[:, 4]
            s_data.friction = data.values[:, 5:8]
            s_data.w_face_areas = data.values[:, 9:12]
        return

    def transform_data(self, link_H_world_dict, airspeed, air_dens):
        for s_name, s_data in self.surface.items():
            link_H_world = link_H_world_dict[s_name]
            # transform the data from world to reference link frame
            s_data.l_nodes = self.transform_points(link_H_world, s_data.w_nodes)
            # Transform variables to coefficients
            dyn_press = 0.5 * air_dens * airspeed**2
            s_data.press_coeff = s_data.pressure / dyn_press
            s_data.fric_coeff = s_data.friction / dyn_press
        return

    def transform_points(self, frame_1_H_frame_2, points):
        ones = np.ones((len(points), 1))
        start_coord = np.hstack((points, ones))
        end_coord = np.dot(frame_1_H_frame_2, start_coord.T).T
        return end_coord[:, :3]

    def reorder_data(self):
        for s_data in self.surface.values():
            # Reorder data nodes to match mesh_nodes order
            indices = []
            for node in s_data.mesh_nodes:
                distances = np.linalg.norm(s_data.l_nodes - node, axis=1)
                indices.append(np.argmin(distances))
            s_data.l_nodes = s_data.l_nodes[indices]
            s_data.w_nodes = s_data.w_nodes[indices]
            s_data.pressure = s_data.pressure[indices]
            s_data.press_coeff = s_data.press_coeff[indices]
            s_data.friction = s_data.friction[indices]
            s_data.fric_coeff = s_data.fric_coeff[indices]
            s_data.w_face_areas = s_data.w_face_areas[indices]
        return

    def assign_global_data(self):
        self.w_nodes = np.empty(shape=(0, 3))
        self.cp = np.empty(shape=(0,))
        self.cf = np.empty(shape=(0, 3))
        for s_data in self.surface.values():
            self.w_nodes = np.append(self.w_nodes, s_data.w_nodes, axis=0)
            self.cp = np.append(self.cp, s_data.press_coeff)
            self.cf = np.append(self.cf, s_data.fric_coeff, axis=0)
        return

    def interp_3d_to_image(
        self,
        im_res,
    ):
        self.image = np.empty(shape=(0, im_res[0], im_res[1]))
        for s_data in self.surface.values():
            # Create a meshgrid for interpolation over the (theta,psi) domain
            x_image = np.linspace(0, np.pi, im_res[1])  # psi
            y_image = np.linspace(-np.pi, np.pi, im_res[0])  # theta
            X, Y = np.meshgrid(x_image, y_image)
            # Interpolate the pressure coefficient data
            query_points = np.vstack((Y.ravel(), X.ravel())).T
            interp = LinearNDInterpolator(
                s_data.map,
                s_data.press_coeff.ravel(),
            )

            interp_press_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.fric_coeff[:, 0].ravel().reshape(-1, 1)
            interp_x_fric_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.fric_coeff[:, 1].ravel().reshape(-1, 1)
            interp_y_fric_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.fric_coeff[:, 2].ravel().reshape(-1, 1)
            interp_z_fric_coeff = interp(query_points).reshape(X.shape)

            # Extrapolate the data using nearest neighbors
            out_ids = np.isnan(interp_press_coeff)
            if np.any(out_ids):
                extrap = NearestNDInterpolator(
                    s_data.map,
                    s_data.press_coeff.ravel().reshape(-1, 1),
                )
                interp_press_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

                extrap.values = s_data.fric_coeff[:, 0].ravel().reshape(-1, 1)
                interp_x_fric_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

                extrap.values = s_data.fric_coeff[:, 1].ravel().reshape(-1, 1)
                interp_y_fric_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

                extrap.values = s_data.fric_coeff[:, 2].ravel().reshape(-1, 1)
                interp_z_fric_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

            # Assign data
            s_data.image = np.array(
                [
                    interp_press_coeff,
                    interp_x_fric_coeff,
                    interp_y_fric_coeff,
                    interp_z_fric_coeff,
                ]
            )
            self.image = np.concatenate((self.image, s_data.image), axis=0)
        return

    def compute_forces(self):
        w_aero_force = np.empty(shape=(0, 3))
        for s_data in self.surface.values():
            areas = np.linalg.norm(s_data.w_face_areas, axis=1)
            d_force = (
                s_data.pressure[:, np.newaxis] * s_data.w_face_areas
                + s_data.friction * areas[:, np.newaxis]
            )
            s_data.w_aero_force = np.sum(d_force, axis=0)
            w_aero_force = np.vstack((w_aero_force, s_data.w_aero_force))
        self.w_aero_force = np.sum(w_aero_force, axis=0)
        return


class FlowGenerator:
    def __init__(self):
        self.surface = {}

    def import_mesh_mapping_data(self, data, surface_list, link_H_world_dict):
        for s_name in surface_list:
            self.surface[s_name] = SurfaceData()
        for s_name, s_data in self.surface.items():
            s_data.map = data[s_name]["map"]
            s_data.mesh_faces = data[s_name]["faces"]
            w_mesh_nodes = data[s_name]["nodes"] / 1000
            link_H_world = link_H_world_dict[s_name]
            # mesh nodes are stored in local reference frame
            s_data.mesh_nodes = self.transform_points(link_H_world, w_mesh_nodes)
        return

    def load_mesh(
        self,
        fluent_data_path,
        link_H_world_ref,
        ref_config_name="hovering",
        ref_pitch=0,
        ref_yaw=0,
    ):
        for s_name, s_data in self.surface.items():
            database_file_path = list(
                fluent_data_path.rglob(
                    f"{ref_config_name}-{ref_pitch}-{ref_yaw}-{s_name}.dtbs"
                )
            )[0]
            data = np.loadtxt(database_file_path, skiprows=1)
            w_nodes = data[:, 1:4]
            w_areas = data[:, 9:12]
            # transform the data from world to reference link frame
            s_data.l_nodes = self.transform_points(link_H_world_ref[s_name], w_nodes)
            s_data.l_areas = self.rotate_vectors(link_H_world_ref[s_name], w_areas)
        return

    def reorder_mesh(self):
        for s_data in self.surface.values():
            # Reorder data nodes to match mesh_nodes order
            indices = []
            for node in s_data.mesh_nodes:
                distances = np.linalg.norm(s_data.l_nodes - node, axis=1)
                indices.append(np.argmin(distances))
            s_data.l_nodes = s_data.l_nodes[indices]
            s_data.l_areas = s_data.l_areas[indices]
        return

    def transform_mesh(self, world_H_link):
        self.w_nodes = np.empty(shape=(0, 3))
        for s_name, s_data in self.surface.items():
            s_data.w_nodes = self.transform_points(world_H_link[s_name], s_data.l_nodes)
            s_data.w_areas = self.rotate_vectors(world_H_link[s_name], s_data.l_areas)
            self.w_nodes = np.append(self.w_nodes, s_data.w_nodes, axis=0)
        return

    def transform_points(self, frame_1_H_frame_2, points):
        ones = np.ones((len(points), 1))
        start_coord = np.hstack((points, ones))
        end_coord = np.dot(frame_1_H_frame_2, start_coord.T).T
        return end_coord[:, :3]

    def rotate_vectors(self, frame_1_H_frame_2, vectors):
        return np.dot(frame_1_H_frame_2[:3, :3], vectors.T).T

    def separate_images(self, image):
        for idx, s_data in enumerate(self.surface.values()):
            s_data.image = image[4 * idx : 4 * (idx + 1), :, :]
        return

    def compute_interpolator(self, im_res):
        for s_data in self.surface.values():
            # Define the image coordinates
            x_image = np.linspace(0, np.pi, im_res[1])  # psi
            y_image = np.linspace(-np.pi, np.pi, im_res[0])  # theta
            # Create and assign interpolator functions
            interp = RegularGridInterpolator(
                (y_image, x_image),
                np.zeros((len(y_image), len(x_image))),
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            # Assign data
            s_data.interp = interp
        return

    def interpolate_flow_data(self):
        self.cp = np.empty(shape=(0,))
        self.cf = np.empty(shape=(0, 3))
        for s_data in self.surface.values():
            interp = s_data.interp
            query_points = s_data.map
            interp.values = s_data.image[0, :, :]
            s_data.press_coeff = interp(query_points)
            interp.values = s_data.image[1, :, :]
            x_fric_coeff = interp(query_points)
            interp.values = s_data.image[2, :, :]
            y_fric_coeff = interp(query_points)
            interp.values = s_data.image[3, :, :]
            z_fric_coeff = interp(query_points)
            s_data.fric_coeff = np.vstack((x_fric_coeff, y_fric_coeff, z_fric_coeff)).T
            self.cp = np.append(self.cp, s_data.press_coeff)
            self.cf = np.vstack((self.cf, s_data.fric_coeff))
        return

    def compute_forces(self, airspeed, air_dens):
        dyn_pressure = 0.5 * air_dens * airspeed**2
        w_aero_force = np.empty(shape=(0, 3))
        for s_data in self.surface.values():
            pressure = s_data.press_coeff[:, np.newaxis] * dyn_pressure
            friction = s_data.fric_coeff * dyn_pressure
            areas = np.linalg.norm(s_data.w_areas, axis=1)[:, np.newaxis]
            d_force = pressure * s_data.w_areas + friction * areas
            s_data.w_aero_force = np.sum(d_force, axis=0)
            w_aero_force = np.vstack((w_aero_force, s_data.w_aero_force))
        self.w_aero_force = np.sum(w_aero_force, axis=0)
        return


class FlowVisualizer:
    def __init__(self, flow):
        self.__dict__.update(flow.__dict__)

    def plot_local_pointcloud(self, surface_data):
        points = np.vstack(
            (surface_data.x_local, surface_data.y_local, surface_data.z_local)
        ).T  # 3D points
        # Normalize the colormap
        norm = Normalize(vmin=-2, vmax=1)
        normalized_flow_variable = norm(surface_data.pressure_coefficient)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Create the local frame
        local_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        # Add the geometries to the scene
        geometries = [
            {"name": "point_cloud", "geometry": point_cloud},
            {"name": "local_frame", "geometry": local_frame},
        ]
        o3d.visualization.draw(geometries, show_skybox=False)
        return

    def plot_pressure_pointcloud(self, robot_meshes):
        # Normalize the colormap
        norm = Normalize(vmin=-2, vmax=1)
        normalized_flow_variable = norm(self.cp)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.w_nodes)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Create the global frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        # create the relative wind direction vector
        wind_vector = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.2,
            cone_height=0.04,
        )
        wind_vector.paint_uniform_color([0, 0.5, 1])  # green-ish blue
        wind_vector.rotate(
            R.from_euler("y", 180, degrees=True).as_matrix(), center=[0, 0, 0]
        )
        wind_vector.translate([0, 0.2, 1.0])
        wind_vector.compute_vertex_normals()
        # Create transparent mesh material
        mesh_material = o3d.visualization.rendering.MaterialRecord()
        mesh_material.shader = "defaultLitTransparency"
        mesh_material.base_color = [0.5, 0.5, 0.5, 0.5]  # RGBA, A is for alpha
        # Assemble the geometries list
        geom = [
            {"name": "point_cloud", "geometry": point_cloud},
            {"name": "world_frame", "geometry": world_frame},
            {"name": "wind_vector", "geometry": wind_vector},
        ]
        for idx, mesh in enumerate(robot_meshes):
            # Add meshes to the geometries list
            geom.append(
                {
                    "name": f"mesh_{idx}",
                    "geometry": mesh["mesh"],
                    "material": mesh_material,
                }
            )
            print(f"Mesh {mesh['name']} added to the scene.")
        o3d.visualization.draw(
            geom, show_skybox=False, non_blocking_and_return_uid=True
        )
        return

    def plot_surface_contour(self, flow_variable, robot_meshes):
        points = np.vstack(
            (self.flow_properties.x, self.flow_properties.y, self.flow_properties.z)
        ).T  # 3D points
        # Normalize the colormap
        norm = Normalize(vmin=-2, vmax=1)
        normalized_flow_variable = norm(flow_variable)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Create the global frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        # create the relative wind direction vector
        wind_vector = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.2,
            cone_height=0.04,
        )
        wind_vector.paint_uniform_color([0, 0.5, 1])  # green-ish blue
        wind_vector.rotate(
            R.from_euler("y", 180, degrees=True).as_matrix(), center=[0, 0, 0]
        )
        wind_vector.translate([0, 0.2, 1.0])
        wind_vector.compute_vertex_normals()
        # Assemble the geometries list
        geometries = [
            {"name": "world_frame", "geometry": world_frame},
            {"name": "wind_vector", "geometry": wind_vector},
        ]
        for mesh_index, mesh in enumerate(robot_meshes):
            # Color meshes with the flow_variable using griddata
            vertices = np.asarray(mesh["mesh"].vertices)
            # Interpolate the data
            mesh_flow_values = griddata(
                (
                    self.flow_properties.x,
                    self.flow_properties.y,
                    self.flow_properties.z,
                ),
                flow_variable,
                vertices,
                method="linear",
            )
            # Extrapolate the data using nearest neighbors
            outside_indices = np.isnan(mesh_flow_values)
            mesh_flow_values[outside_indices] = griddata(
                (
                    self.flow_properties.x,
                    self.flow_properties.y,
                    self.flow_properties.z,
                ),
                flow_variable,
                vertices[outside_indices],
                method="nearest",
            )
            normalized_mesh_flow_values = norm(mesh_flow_values)
            mesh_colors = colormap(normalized_mesh_flow_values)[:, :3]
            mesh["mesh"].vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
            # Add meshes to the geometries list
            geometries.append({"name": f"mesh_{mesh_index}", "geometry": mesh["mesh"]})
            print(f"Mesh {mesh['name']} added to the scene.")
        o3d.visualization.draw(geometries, show_skybox=False)
        return
