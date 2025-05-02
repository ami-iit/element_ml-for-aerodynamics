import numpy as np
import open3d as o3d
import torch
from vtk import vtkXMLUnstructuredGridReader
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize
from matplotlib import cm
from scipy.interpolate import (
    RegularGridInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from numpy.linalg import solve
from scipy.spatial.distance import cdist

from . import mesh as ms


class FlowImporter:
    def __init__(self):
        self.surface = {}

    def import_mapping(self, data):
        self.map = data["map"]
        self.map_faces = data["faces"]
        self.map_nodes = data["nodes"]
        return

    def import_solution_data(self, file):
        # Read the source file.
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(file))
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()
        # Get coordinates
        num_points = output.GetPoints().GetNumberOfPoints()
        points_array = [output.GetPoints().GetPoint(i) for i in range(num_points)]
        # Get solution data
        cp_array = output.GetPointData().GetArray("Pressure_Coefficient")
        cp_data = [cp_array.GetValue(i) for i in range(cp_array.GetNumberOfTuples())]
        cf_array = output.GetPointData().GetArray("Skin_Friction_Coefficient")
        cf_data = [cf_array.GetTuple(i) for i in range(cf_array.GetNumberOfTuples())]
        # Assign data
        self.cp = np.array(cp_data)
        self.cf = np.array(cf_data)
        self.points = np.array(points_array)
        return

    def reorder_data(self):
        indices = []
        for node in self.map_nodes:
            distances = np.linalg.norm(self.points - node, axis=1)
            indices.append(np.argmin(distances))
        self.points = self.points[indices]
        self.cp = self.cp[indices]
        self.cf = self.cf[indices]
        return

    def interp_3d_to_image(self, im_res):
        x_image = np.linspace(self.map[:, 1].min(), self.map[:, 1].max(), im_res[1])
        y_image = np.linspace(self.map[:, 0].min(), self.map[:, 0].max(), im_res[0])
        X, Y = np.meshgrid(x_image, y_image)
        # Interpolate the pressure coefficient data
        query_points = np.vstack((Y.ravel(), X.ravel())).T
        interp = LinearNDInterpolator(self.map, self.cp.ravel())

        interp_cp = interp(query_points).reshape(X.shape)

        interp.values = self.points[:, 0].ravel().reshape(-1, 1)
        interp_x = interp(query_points).reshape(X.shape)

        interp.values = self.points[:, 1].ravel().reshape(-1, 1)
        interp_y = interp(query_points).reshape(X.shape)

        interp.values = self.points[:, 2].ravel().reshape(-1, 1)
        interp_z = interp(query_points).reshape(X.shape)

        interp.values = self.cf[:, 0].ravel().reshape(-1, 1)
        interp_x_cf = interp(query_points).reshape(X.shape)

        interp.values = self.cf[:, 1].ravel().reshape(-1, 1)
        interp_y_cf = interp(query_points).reshape(X.shape)

        interp.values = self.cf[:, 2].ravel().reshape(-1, 1)
        interp_z_cf = interp(query_points).reshape(X.shape)

        # Extrapolate the data using nearest neighbors
        out_ids = np.isnan(interp_cp)
        if np.any(out_ids):
            extrap = NearestNDInterpolator(self.map, self.cp.ravel().reshape(-1, 1))
            interp_cp[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.points[:, 0].ravel().reshape(-1, 1)
            interp_x[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.points[:, 1].ravel().reshape(-1, 1)
            interp_y[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.points[:, 2].ravel().reshape(-1, 1)
            interp_z[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.cf[:, 0].ravel().reshape(-1, 1)
            interp_x_cf[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.cf[:, 1].ravel().reshape(-1, 1)
            interp_y_cf[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

            extrap.values = self.cf[:, 2].ravel().reshape(-1, 1)
            interp_z_cf[out_ids] = extrap(query_points).reshape(X.shape)[out_ids]

        # Assign data
        self.image = np.array(
            [
                interp_x,
                interp_y,
                interp_z,
                interp_cp,
                interp_x_cf,
                interp_y_cf,
                interp_z_cf,
            ]
        )
        return

    def import_mesh(self, nodes, faces):
        self.nodes = nodes
        self.faces = faces
        # Compute face normals and areas
        self.face_normals = np.zeros((len(self.faces), 3))
        self.face_areas = np.zeros(len(self.faces))
        self.face_centroids = np.zeros((len(self.faces), 3))
        for idx, face in enumerate(self.faces):
            face_vertices = self.nodes[face]
            face_normal = np.cross(
                face_vertices[1] - face_vertices[0],
                face_vertices[-1] - face_vertices[0],
            )
            self.face_normals[idx] = face_normal / np.linalg.norm(face_normal)
            self.face_areas[idx], self.face_centroids[idx] = (
                ms.compute_cell_area_and_centroid(face_vertices)
            )
        return

    def compute_aerodynamic_coefficients(self, angle_of_attack, ref_area):
        # Compute the aerodynamic force on each face (assuming dyn_press = 1.0)
        face_forces = np.zeros((len(self.faces), 3))
        for idx, face in enumerate(self.faces):
            cp = np.mean(self.cp[face])
            cf = np.mean(self.cf[face], axis=0)
            d_Fp = -self.face_areas[idx] * cp * self.face_normals[idx]
            d_Ff = self.face_areas[idx] * cf
            face_forces[idx] = d_Fp + d_Ff
        # Compute the total aerodynamic force
        total_force = np.sum(face_forces, axis=0)
        # Compute the aerodynamic coefficients
        rot = R.from_euler("y", angle_of_attack, degrees=True)
        aero_force = np.dot(rot.as_matrix(), total_force)
        self.drag_coefficient = aero_force[0] / ref_area
        self.lift_coefficient = aero_force[2] / ref_area
        self.side_force_coefficient = aero_force[1] / ref_area
        return


class FlowGenerator:
    def __init__(self):
        self.surface = {}

    def import_mapping(self, data):
        self.map = data["map"]
        self.faces = data["faces"]
        self.map_nodes = data["nodes"]
        return

    def compute_interpolator(self, im_res):
        x_image = np.linspace(self.map[:, 1].min(), self.map[:, 1].max(), im_res[1])
        y_image = np.linspace(self.map[:, 0].min(), self.map[:, 0].max(), im_res[0])
        # Create and assign interpolator functions
        interp = RegularGridInterpolator(
            (y_image, x_image),
            np.zeros((len(y_image), len(x_image))),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        # Assign interpolator
        self.interp = interp
        return

    # Load the trained models
    def load_models(self, encoder_path, decoder_path):
        self.encoder = torch.jit.load(encoder_path).to("cpu")
        self.decoder = torch.jit.load(decoder_path).to("cpu")
        self.encoder.eval()
        self.decoder.eval()
        return

    # Thin Plate Spline RBF
    def tps_rbf(self, r, epsilon=1e-6):
        r = np.maximum(r, epsilon)  # Avoid log(0)
        return r**2 * np.log(r)

    # Train RBF using TPS
    def train_rbf_tps(self, X_train, Y_train, epsilon=1e-10):
        pairwise_dists = cdist(X_train, X_train, "euclidean")  # (N, N)
        Phi = self.tps_rbf(pairwise_dists, epsilon)  # (N, N)
        # Solve Phi * W = Y
        W = solve(Phi, Y_train)  # (N, 3)
        self.rbf_centers = X_train
        self.rbf_weights = W
        self.rbf_epsilon = epsilon
        return

    # Predict using TPS
    def predict_rbf_tps(self, X_new):
        pairwise_dists = cdist(X_new, self.rbf_centers, "euclidean")  # (M, N)
        Phi_new = self.tps_rbf(pairwise_dists, self.rbf_epsilon)  # (M, N)
        return Phi_new @ self.rbf_weights  # (M, 3)

    def interpolate_flow_data(self, image):
        interp = self.interp
        query_points = self.map
        # x coordinates
        interp.values = image[0, :, :]
        x = interp(query_points)
        # y coordinates
        interp.values = image[1, :, :]
        y = interp(query_points)
        # z coordinates
        interp.values = image[2, :, :]
        z = interp(query_points)
        # pressure coefficient
        interp.values = image[3, :, :]
        cp = interp(query_points)
        # friction coefficients
        interp.values = image[4, :, :]
        cf_x = interp(query_points)
        interp.values = image[5, :, :]
        cf_y = interp(query_points)
        interp.values = image[6, :, :]
        cf_z = interp(query_points)
        # Assign data
        self.points = np.vstack((x, y, z)).T
        self.cp = cp
        self.cf = np.vstack((cf_x, cf_y, cf_z)).T
        return

    def import_mesh(self, nodes, faces):
        self.nodes = self.points
        self.faces = faces
        # Compute face normals and areas
        self.face_normals = np.zeros((len(self.faces), 3))
        self.face_areas = np.zeros(len(self.faces))
        self.face_centroids = np.zeros((len(self.faces), 3))
        for idx, face in enumerate(self.faces):
            face_vertices = self.nodes[face]
            face_normal = np.cross(
                face_vertices[1] - face_vertices[0],
                face_vertices[-1] - face_vertices[0],
            )
            self.face_normals[idx] = face_normal / np.linalg.norm(face_normal)
            self.face_areas[idx], self.face_centroids[idx] = (
                ms.compute_cell_area_and_centroid(face_vertices)
            )
        return

    def compute_aerodynamic_coefficients(self, angle_of_attack, ref_area):
        # Compute the aerodynamic force on each face (assuming dyn_press = 1.0)
        face_forces = np.zeros((len(self.faces), 3))
        for idx, face in enumerate(self.faces):
            cp = np.mean(self.cp[face])
            cf = np.mean(self.cf[face], axis=0)
            d_Fp = -self.face_areas[idx] * cp * self.face_normals[idx]
            d_Ff = self.face_areas[idx] * cf
            face_forces[idx] = d_Fp + d_Ff
        # Compute the total aerodynamic force
        total_force = np.sum(face_forces, axis=0)
        # Compute the aerodynamic coefficients
        rot = R.from_euler("y", angle_of_attack, degrees=True)
        aero_force = np.dot(rot.as_matrix(), total_force)
        self.drag_coefficient = aero_force[0] / ref_area
        self.lift_coefficient = aero_force[2] / ref_area
        self.side_force_coefficient = aero_force[1] / ref_area
        return


class FlowVisualizer:
    def __init__(self, flow):
        self.__dict__.update(flow.__dict__)

    def plot_wing_pressure(self, window_name="Open3D"):
        # Normalize the colormap
        norm = Normalize(vmin=min(self.cp), vmax=max(self.cp))
        normalized_flow_variable = norm(self.cp)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Set visualization parameters
        zoom = 0.5
        x_cen = (self.points[:, 0].max() + self.points[:, 0].min()) / 2
        y_cen = (self.points[:, 1].max() + self.points[:, 1].min()) / 2
        z_cen = (self.points[:, 2].max() + self.points[:, 2].min()) / 2
        center = [x_cen, y_cen, z_cen]
        front = [-1.0, -1.0, 1.0]
        up = [0.0, 0.0, 1.0]
        # Display the pointcloud
        o3d.visualization.draw_geometries(
            [pcd], zoom=zoom, lookat=center, front=front, up=up, window_name=window_name
        )
        return

    def plot_2D_latent_space_projections(self, input, latent_space):
        input_names = [key for key in input.keys()]
        ls_dim = latent_space.shape[1]
        for k, input_name in enumerate(input_names):
            c = input[input_name]
            vmin = np.min(c)
            vmax = np.max(c)
            cmap = "jet"
            fig, axes = plt.subplots(ls_dim, ls_dim)
            fig.suptitle(f"Latent space according to {input_name}")
            for i in range(ls_dim):
                for j in range(ls_dim):
                    if j > i:
                        axes[i, j].set_axis_off()
                    elif i == j:
                        kde = gaussian_kde(latent_space[:, i])
                        dist_space = np.linspace(
                            min(latent_space[:, i]), max(latent_space[:, i]), 100
                        )
                        axes[i, j].plot(dist_space, kde(dist_space))
                    elif j < i:
                        axes[i, j].scatter(
                            latent_space[:, j],
                            latent_space[:, i],
                            c=c,
                            s=20,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                        )
                    axes[i, j].grid()
                    axes[i, j].set_ylabel(f"Latent Var {i+1}") if j == 0 else None
                    (
                        axes[i, j].set_xlabel(f"Latent Var {j+1}")
                        if i == ls_dim - 1
                        else None
                    )
            plt.show()

    def plot_3D_latent_space(self, input, latent_space):
        input_names = [key for key in input.keys()]
        for k, input_name in enumerate(input_names):
            c = input[input_name]
            vmin = np.min(c)
            vmax = np.max(c)
            cmap = "hsv" if input_name == "yaw" else "jet"
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=latent_space[:, 0],
                        y=latent_space[:, 1],
                        z=latent_space[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=c,
                            colorscale=cmap,
                            cmin=vmin,
                            cmax=vmax,
                            colorbar=dict(title=input_name),
                        ),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Latent Var 1",
                    yaxis_title="Latent Var 2",
                    zaxis_title="Latent Var 3",
                )
            )
            fig.show()

    def plot_rbf_result(self, input, X_image, latent_space_pred):
        # Plot the RBF result
        input_names = [key for key in input.keys()]
        ls_dim = latent_space_pred.shape[1]
        fig, axes = plt.subplots(1, ls_dim)
        fig.suptitle(f"Latent space reconstruction via RBF")
        for i in range(ls_dim):
            c = latent_space_pred[:, i]
            axes[i].set_title(f"latent Var #{i+1}")
            axes[i].imshow(
                c.reshape(X_image[0].shape),
                extent=[
                    X_image[1].min(),
                    X_image[1].max(),
                    X_image[0].min(),
                    X_image[0].max(),
                ],
                cmap="jet",
                origin="lower",
                vmin=np.min(c),
                vmax=np.max(c),
            )
            axes[i].set_xlabel(input_names[1])
            axes[i].set_ylabel(input_names[0])
            plt.tight_layout()
        plt.show()
