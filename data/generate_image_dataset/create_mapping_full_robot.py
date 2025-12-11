import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import src.ansys as ans
import src.mesh as ms
import src.sdem as sdem
import src.mapping as mp

MESH = "dual"  # "dual" or "nodal"
SHOW_PLOTS = True
#
#
# Surface names to include in the mapping
SURFACE_NAMES = [
    "ironcub_head",
    "ironcub_root_link",
    "ironcub_torso",
    "ironcub_torso_pitch",
    "ironcub_torso_roll",
    "ironcub_left_leg_pitch",
    "ironcub_right_leg_pitch",
    "ironcub_left_leg_roll",
    "ironcub_right_leg_roll",
    # "ironcub_left_arm",
    # "ironcub_left_arm_pitch",
    # "ironcub_left_arm_roll",
    # "ironcub_left_turbine",
    # "ironcub_right_arm",
    # "ironcub_right_arm_pitch",
    # "ironcub_right_arm_roll",
    # "ironcub_right_turbine",
    # "ironcub_left_foot",
    # "ironcub_left_leg_lower",
    # "ironcub_left_leg_yaw",
    # "ironcub_left_leg_upper",
    # "ironcub_right_foot",
    # "ironcub_right_leg_lower",
    # "ironcub_right_leg_yaw",
    # "ironcub_right_leg_upper",
    # "ironcub_upper_jetpack",
    # "ironcub_lower_jetpack",
    # "ironcub_left_back_turbine",
    # "ironcub_right_back_turbine",
]


def main():
    # Get the path to the raw data
    # mesh_dir = input("Enter the path to the fluent msh directory: ")
    # mesh_path = Path(str(mesh_dir).strip())
    # data_dir = input("Enter the path to the fluent dlm directory: ")
    # data_path = Path(str(data_dir).strip())
    mesh_path = Path(r"C:\Users\apaolino\code\datasets\cfd-ironcub-mk3\mesh\ascii")
    data_path = Path(r"C:\Users\apaolino\code\datasets\cfd-ironcub-mk3\mesh\dlm")
    # Get the list of files and build the data dictionary
    files = [
        f
        for f in mesh_path.rglob("*.msh")
        if f.is_file() and len(f.stem.split("-")) > 1
    ]
    config_names = sorted(list(set([f.stem.split("-")[0] for f in files])))
    surface_names = sorted(list(set([f.stem.split("-")[1] for f in files])))
    # Create repo to store the mappings
    map_dir = Path(__file__).parents[0] / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Assemble mesh in a single one
    for config in config_names:
        all_nodes = []
        all_faces = []

        node_offset = 0
        for surface in SURFACE_NAMES:
            # Import mesh from Fluent format
            filename = f"{config}-{surface}.msh"
            meshfile = list(mesh_path.rglob(filename))[0]
            nodes, faces = ans.read_fluent_mesh_file(str(meshfile))

            # Offset face indices before adding
            faces = [[idx + node_offset for idx in face] for face in faces]

            # Append
            if len(all_nodes) == 0:
                all_nodes = nodes
            else:
                all_nodes = np.vstack((all_nodes, nodes))
            all_faces.extend(faces)

            node_offset = len(all_nodes)

        # --- REMOVE DUPLICATE NODES ---
        # round to mitigate float precision issues
        rounded_nodes = np.round(all_nodes, decimals=12)
        unique_nodes, unique_indices, inverse_indices = np.unique(
            rounded_nodes, axis=0, return_index=True, return_inverse=True
        )

        # --- UPDATE FACES ---
        new_faces = [[int(inverse_indices[idx]) for idx in face] for face in all_faces]

        # Assign final cleaned arrays
        all_nodes = unique_nodes
        all_faces = new_faces

        # Create and visualize open mesh
        open_mesh = ms.create_mesh_from_faces(all_nodes, all_faces)
        ms.visualize_mesh_with_edges(open_mesh) if SHOW_PLOTS else None
        mesh = ms.close_mesh_boundaries(open_mesh)
        mesh.compute_vertex_normals()
        ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None

        # Check if mesh is genus-0
        if len(mesh.vertices) - 3 * len(mesh.triangles) / 2 + len(mesh.triangles) != 2:
            try:
                mesh = ms.close_mesh_boundaries(mesh)
                ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None
            except:
                ans.warn("The mesh is not a genus-0 closed surface.")

        # Spherical conformal mapping and Mobius area correction
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
        # Compute Laplace-Beltrami smoothing on the sphere
        map = mp.laplacian_smoothing_with_internal_pressure(
            v,
            f,
            num_iters=30000,
            step_size=0.001,
        )
        sdem.plot_mesh(map, f) if SHOW_PLOTS else None

        # Compute spherical conformal map
        bigtri_idx = 0  # try with 0
        scm = sdem.spherical_conformal_map(v, f, bigtri_idx)
        m = np.min(sdem.face_area(f, scm))
        # Recompute if there are null areas
        while m == 0:
            bigtri_idx += 1
            scm = sdem.spherical_conformal_map(v, f, bigtri_idx)
            m = np.min(sdem.face_area(f, scm))
        sdem.plot_mesh(scm, f) if SHOW_PLOTS else None
        # Mobius area correction
        map, _ = sdem.mobius_area_correction_spherical(v, f, scm)
        sdem.plot_mesh(map, f) if SHOW_PLOTS else None

        # Equalize points on spherical surface
        # eq_map = mp.spherical_edge_equalization(
        #     map,
        #     f,
        #     num_iters=100000,
        #     step_size=0.0001,
        # )
        # sdem.plot_mesh(eq_map, f) if SHOW_PLOTS else None

        # Generate and equalize planar map
        map2d = mp.cartesian_to_polar(map[: all_nodes.shape[0]])
        if SHOW_PLOTS:
            mp.plot_single_mapping(map2d, all_nodes[:, 0])
            mp.plot_single_mapping(map2d, all_nodes[:, 1])
            mp.plot_single_mapping(map2d, all_nodes[:, 2])
        # Redistribute points
        map2dr = mp.redistribute_points_sparse(
            map2d,
            bandwidth=0.1,
            num_iterations=10000,
            step_size=0.001,
            alpha=0.1,  # weight for density-based force
            beta=2.0,  # weight for distance-preservation force
            k_neighbors=10,  # number of original neighbors for distance-preservation
        )
        # Rescale the redistributed points to the original range
        map2d_min = np.min(map2d, axis=0)
        map2dr_min = np.min(map2dr, axis=0)
        map2d_ptp = np.ptp(map2d, axis=0)
        map2dr_ptp = np.ptp(map2dr, axis=0)
        map2dr = (map2dr - map2dr_min) / map2dr_ptp * map2d_ptp + map2d_min
        # Plot the original and final maps
        # if SHOW_PLOTS:
        mp.plot_mapping(map2d, map2dr, all_nodes[:, 0], config, "main_body")
        mp.plot_mapping(map2d, map2dr, all_nodes[:, 1], config, "main_body")
        mp.plot_mapping(map2d, map2dr, all_nodes[:, 2], config, "main_body")

        print("wtf")


if __name__ == "__main__":
    main()
