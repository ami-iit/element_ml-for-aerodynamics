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
        self.w_nodes = np.empty(shape=(0, 3))
        self.cp = np.empty(shape=(0,))
        self.cf = np.empty(shape=(0, 3))

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

    def import_raw_fluent_data(
        self,
        data_path,
        config_name,
        pitch,
        yaw,
    ):
        for s_name, s_data in self.surface.items():
            filename = f"{config_name}-{pitch}-{yaw}-{s_name}.dtbs"
            filepath = list(data_path.rglob(filename))[0]
            data = pd.read_csv(filepath, sep="\s+", skiprows=1, header=None)
            s_data.w_nodes = data.values[:, 1:4]
            s_data.pressure = data.values[:, 4]
            s_data.w_shear_stress = data.values[:, 5:8]
        return

    def transform_local_fluent_data(self, link_H_world_dict, airspeed, air_dens):
        for s_name, s_data in self.surface.items():
            link_H_world = link_H_world_dict[s_name]
            # transform the data from world to reference link frame
            s_data.l_nodes = self.transform_points(link_H_world, s_data.w_nodes)
            # Transform variables to coefficients
            dyn_press = 0.5 * air_dens * airspeed**2
            s_data.press_coeff = s_data.pressure / dyn_press
            s_data.w_fric_coeff = s_data.w_shear_stress / dyn_press
        return

    def transform_points(self, frame_1_H_frame_2, points):
        ones = np.ones((len(points), 1))
        start_coord = np.hstack((points, ones))
        end_coord = np.dot(frame_1_H_frame_2, start_coord.T).T
        return end_coord[:, :3]

    def reorder_surface_data(self):
        for s_name, s_data in self.surface.items():
            # Reorder w_nodes to match mesh_nodes order
            node_indices = []
            for node in s_data.mesh_nodes:
                distances = np.linalg.norm(s_data.l_nodes - node, axis=1)
                node_indices.append(np.argmin(distances))
            s_data.l_nodes = s_data.l_nodes[node_indices]
            s_data.w_nodes = s_data.w_nodes[node_indices]
            s_data.pressure = s_data.pressure[node_indices]
            s_data.press_coeff = s_data.press_coeff[node_indices]
            s_data.w_shear_stress = s_data.w_shear_stress[node_indices]
            s_data.w_fric_coeff = s_data.w_fric_coeff[node_indices]

    def assign_global_fluent_data(self):
        for _, s_data in self.surface.items():
            self.w_nodes = np.append(self.w_nodes, s_data.w_nodes, axis=0)
            self.cp = np.append(self.cp, s_data.press_coeff)
            self.cf = np.append(self.cf, s_data.w_fric_coeff, axis=0)
        return

    def interp_3d_to_image(
        self,
        image_resolution_list,
    ):
        im_res = image_resolution_list[0].astype(int)
        self.image = np.empty(shape=(0, im_res[0], im_res[1]))
        for s_name, s_data in self.surface.items():
            # Create a meshgrid for interpolation over the (theta,psi) domain
            x_image = np.linspace(0, np.pi, int(im_res[1]))
            y_image = np.linspace(-np.pi, np.pi, int(im_res[0]))
            X, Y = np.meshgrid(x_image, y_image)
            # Interpolate the pressure coefficient data
            query_points = np.vstack((Y.ravel(), X.ravel())).T
            interp = LinearNDInterpolator(
                s_data.map,
                s_data.press_coeff.ravel(),
            )

            interp_press_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.w_fric_coeff[:, 0].ravel().reshape(-1, 1)
            interp_x_fric_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.w_fric_coeff[:, 1].ravel().reshape(-1, 1)
            interp_y_fric_coeff = interp(query_points).reshape(X.shape)

            interp.values = s_data.w_fric_coeff[:, 2].ravel().reshape(-1, 1)
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

                extrap.values = s_data.w_fric_coeff[:, 0].ravel().reshape(-1, 1)
                interp_x_fric_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

                extrap.values = s_data.w_fric_coeff[:, 1].ravel().reshape(-1, 1)
                interp_y_fric_coeff[out_ids] = extrap(query_points).reshape(X.shape)[
                    out_ids
                ]

                extrap.values = s_data.w_fric_coeff[:, 2].ravel().reshape(-1, 1)
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

    def compute_forces(self, air_density, flow_velocity):
        dyn_pressure = 0.5 * air_density * flow_velocity**2
        for surface in self.surface.values():
            pressure = surface.pressure_coefficient * dyn_pressure
            shear_x = surface.x_friction_coefficient * dyn_pressure
            shear_y = surface.y_friction_coefficient * dyn_pressure
            shear_z = surface.z_friction_coefficient * dyn_pressure
            areas = np.sqrt(
                surface.x_area_global**2
                + surface.y_area_global**2
                + surface.z_area_global**2
            )
            d_force_x = pressure * surface.x_area_global + shear_x * areas
            d_force_y = pressure * surface.y_area_global + shear_y * areas
            d_force_z = pressure * surface.z_area_global + shear_z * areas
            surface.global_force = np.array(
                [
                    np.sum(d_force_x),
                    np.sum(d_force_y),
                    np.sum(d_force_z),
                ]
            )
        return


class FlowGenerator:
    def __init__(self, surface_list, image_resolution_list) -> None:
        self.x = np.empty(shape=(0,))
        self.y = np.empty(shape=(0,))
        self.z = np.empty(shape=(0,))
        self.cp = np.empty(shape=(0,))
        self.fx = np.empty(shape=(0,))
        self.fy = np.empty(shape=(0,))
        self.fz = np.empty(shape=(0,))
        self.surface = {}
        for surf_idx, surf_name in enumerate(surface_list):
            self.surface[surf_name] = SurfaceData()
            img_res = (image_resolution_list[surf_idx]).astype(int)
            self.surface[surf_name].image = np.zeros(shape=(4, img_res[0], img_res[1]))
        return

    def separate_images(self, image, surface_list):
        for surface_index, surface_name in enumerate(surface_list):
            self.surface[surface_name].image = image[
                4 * surface_index : 4 * (surface_index + 1), :, :
            ]
        return

    def load_mesh_points(
        self,
        fluent_data_path,
        surface_list,
        link_H_world_ref_dict,
        reference_joint_config_name="hovering",
        reference_pitch_angle=0,
        reference_yaw_angle=0,
    ):
        for surface_name in surface_list:
            database_file_path = list(
                fluent_data_path.rglob(
                    f"mesh-{reference_joint_config_name}-{reference_pitch_angle}-{reference_yaw_angle}-{surface_name}.dtbs"
                )
            )[0]
            data = np.loadtxt(database_file_path, skiprows=1)
            x_global = data[:, 1]
            y_global = data[:, 2]
            z_global = data[:, 3]
            x_area_global = data[:, 4]
            y_area_global = data[:, 5]
            z_area_global = data[:, 6]
            # transform the data from world to reference link frame
            (
                self.surface[surface_name].x_local,
                self.surface[surface_name].y_local,
                self.surface[surface_name].z_local,
            ) = self.transform_points(
                link_H_world_ref_dict[surface_name], x_global, y_global, z_global
            )
            (
                self.surface[surface_name].x_area_local,
                self.surface[surface_name].y_area_local,
                self.surface[surface_name].z_area_local,
            ) = self.rotate_vectors(
                link_H_world_ref_dict[surface_name],
                x_area_global,
                y_area_global,
                z_area_global,
            )
        return

    def transform_points(self, frame_1_H_frame_2, x, y, z):
        ones = np.ones((len(x),))
        starting_coordinates = np.vstack((x, y, z, ones)).T
        ending_coordinates = np.dot(frame_1_H_frame_2, starting_coordinates.T).T
        return (
            ending_coordinates[:, 0],
            ending_coordinates[:, 1],
            ending_coordinates[:, 2],
        )

    def rotate_vectors(self, frame_1_H_frame_2, x, y, z):
        starting_components = np.vstack((x, y, z)).T
        ending_components = np.dot(frame_1_H_frame_2[:3, :3], starting_components.T).T
        return (
            ending_components[:, 0],
            ending_components[:, 1],
            ending_components[:, 2],
        )

    def transform_mesh_points(
        self,
        surface_list,
        world_H_link_dict,
    ):
        self.x = np.empty(shape=(0,))
        self.y = np.empty(shape=(0,))
        self.z = np.empty(shape=(0,))
        for surface_name in surface_list:
            # compute the transformation from the link frame to the current world frame
            (
                self.surface[surface_name].x_global,
                self.surface[surface_name].y_global,
                self.surface[surface_name].z_global,
            ) = self.transform_points(
                world_H_link_dict[surface_name],
                self.surface[surface_name].x_local,
                self.surface[surface_name].y_local,
                self.surface[surface_name].z_local,
            )
            (
                self.surface[surface_name].x_area_global,
                self.surface[surface_name].y_area_global,
                self.surface[surface_name].z_area_global,
            ) = self.rotate_vectors(
                world_H_link_dict[surface_name],
                self.surface[surface_name].x_area_local,
                self.surface[surface_name].y_area_local,
                self.surface[surface_name].z_area_local,
            )
            # assign global coordinates
            self.x = np.append(self.x, self.surface[surface_name].x_global)
            self.y = np.append(self.y, self.surface[surface_name].y_global)
            self.z = np.append(self.z, self.surface[surface_name].z_global)
        return

    def compute_interpolator(
        self,
        surface_list,
        main_axes,
    ):
        for surface_index, surface_name in enumerate(surface_list):
            if main_axes[surface_index] == 0:
                x = self.surface[surface_name].y_local
                y = self.surface[surface_name].z_local
                z = self.surface[surface_name].x_local
            elif main_axes[surface_index] == 1:
                x = self.surface[surface_name].x_local
                y = self.surface[surface_name].z_local
                z = self.surface[surface_name].y_local
            elif main_axes[surface_index] == 2:
                x = self.surface[surface_name].x_local
                y = self.surface[surface_name].y_local
                z = self.surface[surface_name].z_local
            # Trasform to cylindrical coordinates
            r = np.sqrt(x**2 + y**2)
            r_mean = np.mean(r)
            theta = np.arctan2(y, x)
            theta_r = theta * r_mean
            # Define the image coordinates
            x_image = np.linspace(
                np.min(theta_r),
                np.max(theta_r),
                self.surface[surface_name].image.shape[2],
            )
            y_image = np.linspace(
                np.min(z),
                np.max(z),
                self.surface[surface_name].image.shape[1],
            )
            # Create and assign interpolator functions
            interp = RegularGridInterpolator(
                (y_image, x_image),
                np.zeros((len(y_image), len(x_image))),
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            # Assign data
            self.surface[surface_name].interp = interp
            self.surface[surface_name].z = z
            self.surface[surface_name].theta = theta
            self.surface[surface_name].theta_r = theta_r
        return

    def interpolate_flow_data(self, surface_list):
        self.cp = np.empty(shape=(0,))
        self.fx = np.empty(shape=(0,))
        self.fy = np.empty(shape=(0,))
        self.fz = np.empty(shape=(0,))
        for surface_name in surface_list:
            interp = self.surface[surface_name].interp
            query_points = np.vstack(
                (self.surface[surface_name].z, self.surface[surface_name].theta_r)
            ).T
            interp.values = self.surface[surface_name].image[0, :, :]
            self.surface[surface_name].pressure_coefficient = interp(query_points)
            interp.values = self.surface[surface_name].image[1, :, :]
            self.surface[surface_name].x_friction_coefficient = interp(query_points)
            interp.values = self.surface[surface_name].image[2, :, :]
            self.surface[surface_name].y_friction_coefficient = interp(query_points)
            interp.values = self.surface[surface_name].image[3, :, :]
            self.surface[surface_name].z_friction_coefficient = interp(query_points)
            # Assign data
            self.cp = np.append(
                self.cp, self.surface[surface_name].pressure_coefficient
            )
            self.fx = np.append(
                self.fx, self.surface[surface_name].x_friction_coefficient
            )
            self.fy = np.append(
                self.fy, self.surface[surface_name].y_friction_coefficient
            )
            self.fz = np.append(
                self.fz, self.surface[surface_name].z_friction_coefficient
            )
        return

    def compute_forces(self, air_density, flow_velocity):
        dyn_pressure = 0.5 * air_density * flow_velocity**2
        for surface in self.surface.values():
            pressure = surface.pressure_coefficient * dyn_pressure
            shear_x = surface.x_friction_coefficient * dyn_pressure
            shear_y = surface.y_friction_coefficient * dyn_pressure
            shear_z = surface.z_friction_coefficient * dyn_pressure
            areas = np.sqrt(
                surface.x_area_global**2
                + surface.y_area_global**2
                + surface.z_area_global**2
            )
            d_force_x = pressure * surface.x_area_global + shear_x * areas
            d_force_y = pressure * surface.y_area_global + shear_y * areas
            d_force_z = pressure * surface.z_area_global + shear_z * areas
            surface.global_force = np.array(
                [
                    np.sum(d_force_x),
                    np.sum(d_force_y),
                    np.sum(d_force_z),
                ]
            )
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
        geometries = [
            {"name": "point_cloud", "geometry": point_cloud},
            {"name": "world_frame", "geometry": world_frame},
            {"name": "wind_vector", "geometry": wind_vector},
        ]
        for idx, mesh in enumerate(robot_meshes):
            # Add meshes to the geometries list
            geometries.append(
                {
                    "name": f"mesh_{idx}",
                    "geometry": mesh["mesh"],
                    "material": mesh_material,
                }
            )
            print(f"Mesh {mesh['name']} added to the scene.")
        o3d.visualization.draw(geometries, show_skybox=False)
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
