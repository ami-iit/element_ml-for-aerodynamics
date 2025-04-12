import numpy as np
from pathlib import Path

import src.su2 as su2
import src.mesh as ms
import src.mapping as mp

SHOW_PLOTS = True
DENSITY_EXP = 3


def main():
    # Get the path to the raw data
    mesh_dir = Path(__file__).parents[0] / "simulations"
    # Get the list of files and build the data dictionary
    files = [file for file in mesh_dir.rglob("*.su2") if file.is_file()]
    wing_names = sorted(list(set([file.stem for file in files])))
    # Create repo to store the mappings
    map_dir = Path(__file__).parents[0] / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Generate mappings
    for wing in wing_names:
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

        # Compute the planar map
        boundary_edges = np.array(ms.get_boundary_edges(mesh))
        boundary_polygons = ms.generate_boundary_polygons(
            np.array(boundary_edges), nodes
        )
        poly = boundary_polygons[0]
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
        square_map = mp.planar_conformal_map(v, f, poly)
        map2d = square_map[: nodes.shape[0]]
        mp.plot_planar_conformal_map(map2d, faces) if SHOW_PLOTS else None

        # Redistribute points
        adj_list = mp.get_adjacency_list(faces)
        map2dr = mp.redistribute_points_with_equilibrium_constrain(
            map2d,
            adj_list=adj_list,
            boundary_nodes=poly[1],
        )

        mp.plot_planar_conformal_map(map2dr, faces) if SHOW_PLOTS else None

        map2dr_dens = mp.redistribute_points_with_equilibrium_constrain_density(
            map2dr,
            adj_list=adj_list,
            boundary_nodes=poly[1],
            density_exp=DENSITY_EXP,
        )

        if SHOW_PLOTS:
            mp.plot_planar_conformal_map(map2dr, faces)
            mp.plot_mapping(map2dr, map2dr_dens, nodes[:, 0], wing)
            mp.plot_mapping(map2dr, map2dr_dens, nodes[:, 1], wing)
            mp.plot_mapping(map2dr, map2dr_dens, nodes[:, 2], wing)

        # Save the mapping
        wing_data = {
            "map": map2dr_dens,
            "nodes": nodes,
            "faces": faces,
        }
        np.save(map_dir / f"{wing}-map-pcm-{DENSITY_EXP}.npy", wing_data)


if __name__ == "__main__":
    main()
