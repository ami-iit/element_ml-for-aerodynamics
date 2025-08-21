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
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

import src.mesh as ms
import src.mapping as mp

REDISTRIBUTION_STRATEGY = "density-based"  ## "mesh-based" or "density-based"
DENSITY_EXP = 10

ITER_NUM = 100  # Number of iterations for redistribution
BOUNDARY_SHAPE = "square"
SHOW_PLOTS = False


def main():
    root = Path(__file__).parents[0]
    # Get the path to the raw data
    mesh_dir = root / "simulations" / "new_database"
    # Get the list of files and build the data dictionary
    files = [file for file in mesh_dir.rglob("*AoA0.vtu") if file.is_file()]
    wing_names = sorted(list(set([file.parent.stem for file in files])))
    # Create repo to store the mappings
    map_dir = root / "maps" / "new"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Generate mappings
    for wing, file in zip(wing_names, files):
        # Read the source file
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(file))
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()
        # Get points and faces
        num_points = output.GetPoints().GetNumberOfPoints()
        nodes_list = [output.GetPoints().GetPoint(i) for i in range(num_points)]
        nodes = np.array(nodes_list)
        num_cells = output.GetNumberOfCells()
        faces = []
        for idx in range(num_cells):
            num_cell_nodes = output.GetCell(idx).GetNumberOfPoints()
            cell = [output.GetCell(idx).GetPointId(i) for i in range(num_cell_nodes)]
            faces.append(cell)

        # Create and visualize open mesh
        mesh = ms.create_mesh_from_faces_split(nodes, faces)
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

        # Redistribution algorithm
        # 1. Mesh-based redistribution
        if REDISTRIBUTION_STRATEGY == "density-based":
            map2dr = mp.redistribute_points_with_equilibrium_constrain_density(
                map2d,
                boundary_nodes=boundary_nodes,
                adj_list=adj_list,
                density_exp=DENSITY_EXP,
                strategy=REDISTRIBUTION_STRATEGY,
            )
        # 2. Mesh-based redistribution
        elif REDISTRIBUTION_STRATEGY == "mesh-based":
            mesh_edge_lengths = mp.compute_norm_edge_lengths(nodes, adj_list)
            map_edge_lengths = mp.compute_norm_edge_lengths(map2d, adj_list)
            map2dr_old = map2d.copy()
            iter, epsilon = 0, 1.0
            while iter < ITER_NUM and epsilon > 0.001:
                # Compute the edge differences
                edge_deltas = mesh_edge_lengths
                for i in range(len(mesh_edge_lengths)):
                    for j in range(len(mesh_edge_lengths[i])):
                        edge_deltas[i][j] = (
                            mesh_edge_lengths[i][j] - map_edge_lengths[i][j]
                        )
                # Compute the redistributed points
                map2dr = mp.redistribute_points_with_equilibrium_constrain_density(
                    map2d,
                    boundary_nodes=boundary_nodes,
                    adj_list=adj_list,
                    edge_deltas=edge_deltas,
                    density_exp=DENSITY_EXP,
                    strategy=REDISTRIBUTION_STRATEGY,
                )
                epsilon = np.linalg.norm(map2dr - map2dr_old)
                map2dr_old = map2dr.copy()
                map_edge_lengths = mp.compute_norm_edge_lengths(map2dr, adj_list)
                print(f"iteration: {iter+1}/{ITER_NUM}, epsilon: {epsilon}")
                iter += 1

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
            "adjacency_list": adj_list,
            "boundary_nodes": boundary_nodes,
        }
        np.save(map_dir / f"{wing}-map-eem-{DENSITY_EXP}.npy", wing_data)

        print(f"Mapping for {wing} saved.")


if __name__ == "__main__":
    main()
