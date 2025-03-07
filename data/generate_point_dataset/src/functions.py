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
    R_pitch = R.from_euler("y", -pitch, degrees=True)
    R_yaw = R.from_euler("z", -yaw, degrees=True)
    R_align = R.from_euler("y", -90, degrees=True)
    A_R_b = R_align * R_yaw * R_pitch
    wind_velocity = A_R_b.inv().as_matrix()[:, 0] * wind_intensity
    return wind_velocity


def rotate_geometry(raw_node_pos, raw_face_normals, pitch, yaw):
    R_pitch = R.from_euler("x", pitch, degrees=True)
    R_yaw = R.from_euler("y", -yaw, degrees=True)
    Rot = R_pitch * R_yaw
    node_pos = np.dot(Rot.as_matrix(), raw_node_pos.T).T
    face_normals = np.dot(Rot.as_matrix(), raw_face_normals.T).T
    return node_pos, face_normals


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
