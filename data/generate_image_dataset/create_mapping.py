import numpy as np
from pathlib import Path

import src.ansys as ans
import src.mesh as ms
import src.sdem as sdem
import src.mapping as mp

SHOW_PLOTS = False


def main():

    # Get the path to the raw data
    data_dir = input("Enter the path to the fluent msh directory: ")
    data_path = Path(str(data_dir).strip())
    # Get the list of files and build the data dictionary
    files = [file for file in data_path.rglob("*.msh") if file.is_file()]
    config_names = sorted(list(set([file.stem.split("-")[0] for file in files])))
    surface_names = sorted(list(set([file.stem.split("-")[1] for file in files])))
    data = {key: {surface: None for surface in surface_names} for key in config_names}
    # Create repo to store the mappings
    map_dir = Path(__file__).parents[0] / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Generate mappings
    for file in files:
        config_name = file.stem.split("-")[0]
        surface_name = file.stem.split("-")[1]

        # Import and close mesh from Fluent format
        nodes, faces = ans.read_fluent_mesh_file(file)
        # Create and visualize open mesh
        mesh = ms.create_mesh_from_faces(nodes, faces)
        mesh = ms.close_mesh_boundaries(mesh)
        mesh.compute_vertex_normals()
        ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None
        # Check if mesh is genus-0
        if len(mesh.vertices) - 3 * len(mesh.triangles) / 2 + len(mesh.triangles) != 2:
            ans.warn("The mesh is not a genus-0 closed surface.")

        # Spherical conformal mapping and Mobius area correction
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
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

        # Generate and equalize planar map
        map2d = mp.cartesian_to_polar(map[: nodes.shape[0]])
        # Redistribute points
        map2dr = mp.redistribute_points(
            map2d,
            bandwidth=0.1,
            num_iterations=50,
            step_size=0.05,
            alpha=1.0,  # weight for density-based force
            beta=1.0,  # weight for distance-preservation force
            k_neighbors=10,  # number of original neighbors for distance-preservation
        )
        # Rescale the redistributed points to the original range
        map2d_min = np.min(map2d, axis=0)
        map2dr_min = np.min(map2dr, axis=0)
        map2d_ptp = np.ptp(map2d, axis=0)
        map2dr_ptp = np.ptp(map2dr, axis=0)
        map2dr = (map2dr - map2dr_min) / map2dr_ptp * map2d_ptp + map2d_min
        # Plot the original and final maps
        if SHOW_PLOTS:
            mp.plot_mapping(map2d, map2dr, nodes[:, 0], config_name, surface_name)
        # Save the final map
        data[config_name][surface_name] = {
            "map": map2dr,
            "nodes": nodes,
            "faces": faces,
        }
        print(f"{config_name}-{surface_name} mapping generated.")
    # Save map data to file
    for key in data.keys():
        np.save(map_dir / f"{key}-node-map.npy", data[key])


if __name__ == "__main__":
    main()
