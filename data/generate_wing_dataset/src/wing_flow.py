import numpy as np
import pandas as pd
import open3d as o3d
from vtk import vtkXMLUnstructuredGridReader
from matplotlib.pyplot import Normalize
from matplotlib import cm
from scipy.interpolate import (
    griddata,
    RegularGridInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt


class FlowImporter:
    def __init__(self):
        self.surface = {}

    def import_mapping(self, data):
        self.map = data["map"]
        self.faces = data["faces"]
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
        x_image = np.linspace(0, np.pi, im_res[1])  # psi
        y_image = np.linspace(-np.pi, np.pi, im_res[0])  # theta
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


class FlowGenerator:
    def __init__(self):
        self.surface = {}

    def import_mapping(self, data):
        self.map = data["map"]
        self.faces = data["faces"]
        self.map_nodes = data["nodes"]
        return

    def compute_interpolator(self, im_res):
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
        # Assign interpolator
        self.interp = interp
        return

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


class FlowVisualizer:
    def __init__(self, flow):
        self.__dict__.update(flow.__dict__)

    def plot_wing_pressure(self):
        # Normalize the colormap
        norm = Normalize(vmin=min(self.cp), vmax=max(self.cp))
        normalized_flow_variable = norm(self.cp)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Display the pointcloud
        o3d.visualization.draw_geometries([pcd])
        return
