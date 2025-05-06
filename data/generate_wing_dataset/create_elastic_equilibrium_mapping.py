"""
Author: Antonello Paolino
Date: 2025-03-25
Description: This code generates the Elastic Equilibrium Mapping (EEM) for a set of wings.
             It reads the mesh data from SU2 files, and computes the planar map at elastic
             equilibrium. The mapping is then redistributed by reapplying the elastic
             equilibrium condition with a density-based non-uniform general stiffness matrix.
"""

import numpy as np
from pathlib import Path

import src.su2 as su2
import src.mesh as ms
import src.mapping as mp

BOUNDARY_SHAPE = "square"
SHOW_PLOTS = False
DENSITY_EXP = 3


def main():
    root = Path(__file__).parents[0]
    # Get the path to the raw data
    mesh_dir = root / "simulations"
    # Get the list of files and build the data dictionary
    files = [file for file in mesh_dir.rglob("*.su2") if file.is_file()]
    wing_names = sorted(list(set([file.stem for file in files])))
    # Create repo to store the mappings
    map_dir = root / "maps"
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

        # Compute the planar map at elastic equilibrium
        boundary_edges = np.array(ms.get_boundary_edges(mesh))
        boundary_polygons = ms.generate_boundary_polygons(boundary_edges, nodes)
        boundary_nodes = boundary_polygons[0][1]
        start_idx = np.argmin(nodes[boundary_nodes][:, 0])
        boundary_nodes = boundary_nodes[start_idx:] + boundary_nodes[:start_idx]
        adj_list = mp.get_adjacency_list(faces)
        map2d = mp.compute_elastic_equilibrium_planar_map(
            boundary_nodes, adj_list, boundary_shape=BOUNDARY_SHAPE
        )
        mp.plot_planar_map(map2d, faces) if SHOW_PLOTS else None

        # Redistribute points with density function
        map2dr = mp.redistribute_points_with_equilibrium_constrain_density(
            map2d,
            adj_list=adj_list,
            boundary_nodes=boundary_nodes,
            density_exp=DENSITY_EXP,
        )
        if SHOW_PLOTS:
            mp.plot_planar_map(map2dr, faces)
            mp.plot_mapping(map2d, map2dr, nodes[:, 0], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 1], wing)
            mp.plot_mapping(map2d, map2dr, nodes[:, 2], wing)

        # Save the mapping
        wing_data = {
            "map": map2dr,
            "nodes": nodes,
            "faces": faces,
        }
        np.save(map_dir / f"{wing}-map-eem-{DENSITY_EXP}.npy", wing_data)

        print(f"Mapping for {wing} saved.")


if __name__ == "__main__":
    main()
