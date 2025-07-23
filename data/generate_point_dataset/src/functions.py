import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
from matplotlib.colors import Normalize


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
    # Remove duplicates
    pitch_yaw_angles = np.unique(pitch_yaw_angles, axis=0)
    return pitch_yaw_angles


def compute_wind_velocity(pitch, yaw, wind_intensity):
    a_R_b = R.from_euler("zxy", [-90, -pitch, yaw], degrees=True).as_matrix()
    b_R_A = a_R_b.T
    wind_versor = b_R_A @ np.array([0, 0, -1])
    wind_velocity = wind_versor * wind_intensity
    return wind_velocity


def rotate_geometry(raw_node_pos, raw_face_normals, pitch, yaw):
    R_pitch = R.from_euler("x", pitch, degrees=True)
    R_yaw = R.from_euler("y", -yaw, degrees=True)
    Rot = R_pitch * R_yaw
    node_pos = np.dot(Rot.as_matrix(), raw_node_pos.T).T
    face_normals = np.dot(Rot.as_matrix(), raw_face_normals.T).T
    return node_pos, face_normals


def transform_points(frame_1_H_frame_2, points):
    ones = np.ones((len(points), 1))
    start_coord = np.hstack((points, ones))
    end_coord = np.dot(frame_1_H_frame_2, start_coord.T).T
    return end_coord[:, :3]


def rotate_vectors(frame_1_H_frame_2, vectors):
    frame_1_R_frame_2 = frame_1_H_frame_2[:3, :3]
    rotated_vectors = np.dot(frame_1_R_frame_2, vectors.T).T
    return rotated_vectors


def visualize_pointcloud(points, values, window_name="Open3D"):
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


def check_geometry(node_pos, pressure_coefficient):
    # Normalize the colormap
    norm = Normalize(vmin=-2, vmax=1)
    normalized_flow_variable = norm(pressure_coefficient)
    colormap = cm.jet
    colors = colormap(normalized_flow_variable)[:, :3]
    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(node_pos)
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
    geom = [
        {"name": "point_cloud", "geometry": point_cloud},
        {"name": "world_frame", "geometry": world_frame},
        {"name": "wind_vector", "geometry": wind_vector},
    ]
    o3d.visualization.draw(geom, show_skybox=False, non_blocking_and_return_uid=False)
    return
