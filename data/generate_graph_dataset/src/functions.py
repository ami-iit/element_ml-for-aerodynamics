import numpy as np
from scipy.spatial.transform import Rotation as R


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
