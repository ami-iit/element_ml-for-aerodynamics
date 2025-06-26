import numpy as np
from pathlib import Path
import pandas as pd

import src.ansys as ans
import src.mesh as ms


MESH = "dual"  # "dual" or "nodal"
SHOW_PLOTS = True


def main():

    # Get the path to the raw data
    mesh_dir = input("Enter the path to the fluent msh directory: ")
    mesh_path = Path(str(mesh_dir).strip())
    data_dir = input("Enter the path to the fluent dlm directory: ")
    data_path = Path(str(data_dir).strip())

    # Get the list of files and build the data dictionary
    files = [file for file in mesh_path.rglob("*.msh") if file.is_file()]
    config_names = sorted(list(set([file.stem.split("-")[0] for file in files])))
    surface_names = sorted(list(set([file.stem.split("-")[1] for file in files])))

    # Create repo to store the graphs
    graph_dir = Path(__file__).parents[0] / "disconnected-graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Generate graphs
    for config in config_names:
        config_data = {}
        all_nodes = []
        all_faces = []
        all_edges = []
        for surface in surface_names:

            # Import mesh from Fluent format
            filename = f"{config}-{surface}.msh"
            meshfile = list(mesh_path.rglob(filename))[0]
            nodes, faces = ans.read_fluent_mesh_file(str(meshfile))
            if MESH == "dual":

                # Transform in dual mesh
                filename = f"{config}-{surface}.dlm"
                datafile = list(data_path.rglob(filename))[0]
                celldata = pd.read_csv(datafile, sep="\s+", skiprows=1, header=None)
                cells = celldata.values[:, 1:4]
                nodes, faces = ms.build_dual_mesh(nodes, faces, cells)

            # Create and visualize open mesh
            if SHOW_PLOTS:
                mesh = ms.create_mesh_from_faces(nodes, faces)
                ms.visualize_mesh_with_edges(mesh, nodes, faces) if SHOW_PLOTS else None

            # Get all mesh edges
            edges = []
            for face in faces:
                edges.extend(
                    [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
                )

            # Remove duplicates (check also for reversed edges)
            edges = [tuple(sorted(edge)) for edge in edges]
            edges = list(set(edges))

            # extend edges with their reverse
            edges.extend([(edge[1], edge[0]) for edge in edges])

            # Add nodes, faces and edges to the global lists
            current_node_num = len(all_nodes)
            all_nodes.extend(nodes)
            all_faces.extend(
                [[node + current_node_num for node in face] for face in faces]
            )
            all_edges.extend(
                [[node + current_node_num for node in edge] for edge in edges]
            )

            # Save the final graph
            edges = np.array(edges)
            config_data[surface] = {
                "nodes": nodes,
                "faces": faces,
                "edges": edges,
            }

            print(f"Surface {surface} edges added to the graph.")

        # Transform all_nodes and all_edges to numpy arrays
        all_nodes = np.array(all_nodes)
        all_edges = np.array(all_edges).T

        # Create and visualize closed mesh
        if SHOW_PLOTS:
            mesh = ms.create_mesh_from_faces(all_nodes, all_faces)
            ms.visualize_mesh_with_edges(mesh, all_nodes, all_faces)

        # Save graph to file
        np.save(graph_dir / f"{config}-{MESH}-graph.npy", config_data)
        print(f"{config} {MESH} graph saved.")


if __name__ == "__main__":
    main()
