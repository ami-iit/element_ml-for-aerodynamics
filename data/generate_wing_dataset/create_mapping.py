import numpy as np
from pathlib import Path

import src.su2 as su2
import src.mesh as ms
import src.sdem as sdem
import src.mapping as mp

SHOW_PLOTS = True


def main():
    # Get the path to the raw data
    mesh_dir = Path(__file__).parents[0] / "simulations"
    data_dir = Path(__file__).parents[0] / "simulations"
    # Get the list of files and build the data dictionary
    files = [file for file in mesh_dir.rglob("*.su2") if file.is_file()]
    wing_names = sorted(list(set([file.stem for file in files])))
    # Create repo to store the mappings
    map_dir = Path(__file__).parents[0] / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Generate mappings
    data = {}
    alt_data = {}
    for wing in wing_names[1:]:
        wing_data = {}
        filename = f"{wing}.su2"
        meshfile = list(mesh_dir.rglob(filename))[0]
        su2_mesh = su2.read(str(meshfile))
        su2_nodes = su2_mesh.points
        su2_faces = []
        for idx, cells in enumerate(su2_mesh.cells_dict.values()):
            cell_list = cells.tolist()
            wing_indices = np.where(su2_mesh.cell_data["su2:tag"][idx] == 1)[0].tolist()
            add_faces = [cell_list[i] for i in wing_indices]
            su2_faces.extend(add_faces)
        # Keep just nodes and faces of wing
        unique_nodes = sorted(set(idx for face in su2_faces for idx in face))
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        faces = [[index_map[idx] for idx in face] for face in su2_faces]
        nodes = su2_nodes[unique_nodes]
        # Create and visualize open mesh
        mesh = ms.create_mesh_from_faces(nodes, faces)
        ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None
        mesh = ms.close_mesh_boundaries(mesh)
        mesh.compute_vertex_normals()
        ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None
        # Check if mesh is genus-0
        if len(mesh.vertices) - 3 * len(mesh.triangles) / 2 + len(mesh.triangles) != 2:
            try:
                mesh = ms.close_mesh_boundaries(mesh)
                ms.visualize_mesh_with_edges(mesh) if SHOW_PLOTS else None
            except:
                su2.warn("The mesh is not a genus-0 closed surface.")

        # Spherical conformal mapping and Mobius area correction
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
        # Compute spherical conformal map
        bigtri_idx = 2  # try with 0
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

        # Generate and redistribute planar map
        map2d = mp.cartesian_to_polar(map[: nodes.shape[0]])
        map2dr = mp.density_cdf_warping(map2d)
        # Plot the original and final maps
        if SHOW_PLOTS:
            mp.plot_mapping(map2d, map2dr, nodes[:, 0], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 1], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 2], wing)

        # Generate and equalize planar map
        map2d = mp.cartesian_to_polar(map[: nodes.shape[0]])
        # Get adjacency list
        adj_list = mp.get_adjacency_list(faces)
        # Redistribute points
        map2dr = mp.redistribute_points(
            map2d,
            bandwidth=0.1,
            num_iterations=250,
            step_size=0.05,
            alpha=0.01,  # weight for density-based force
            beta=1.0,  # weight for distance-preservation force
            adj_list=adj_list,
        )
        # Plot the original and final maps
        if SHOW_PLOTS:
            mp.plot_mapping(map2d, map2dr, nodes[:, 0], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 1], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 2], wing)
        # Save the final map
        wing_data = {
            "map": map2dr,
            "nodes": nodes,
            "faces": faces,
        }
        np.save(map_dir / f"{wing}-map.npy", wing_data)
        data[wing] = wing_data

        # Ragularization
        map2dr_new = mp.density_cdf_warping(map2dr)
        wing_alt_data = {
            "map": map2dr_new,
            "nodes": nodes,
            "faces": faces,
        }
        np.save(map_dir / f"{wing}-map-new.npy", wing_alt_data)
        alt_data[wing] = wing_alt_data
        print(f"{wing} mapping generated.")
    # Save map data to file
    np.save(map_dir / f"wing-maps.npy", data)
    np.save(map_dir / f"wing-maps-new.npy", alt_data)
    print(f"wing mappings saved.")


if __name__ == "__main__":
    main()
