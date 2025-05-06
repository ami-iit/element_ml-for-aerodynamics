import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve


def plot_mapping(original_map, redistributed_map, values, wing_name, im_res=256):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].xaxis.set_major_locator(ticker.MultipleLocator(2 * np.pi / im_res))
    # ax[0].yaxis.set_major_locator(ticker.MultipleLocator(np.pi / im_res))
    ax[0].grid(True, which="major")
    ax[0].scatter(
        original_map[:, 0],
        original_map[:, 1],
        c=values,
        cmap="jet",
        alpha=0.6,
    )
    ax[0].set_title(f"{wing_name} Original Points")
    ax[0].legend()
    ax[0].axis("equal")
    # ax[1].xaxis.set_major_locator(ticker.MultipleLocator(2 * np.pi / im_res))
    # ax[1].yaxis.set_major_locator(ticker.MultipleLocator(np.pi / im_res))
    ax[1].grid(True, which="major")
    ax[1].scatter(
        redistributed_map[:, 0],
        redistributed_map[:, 1],
        c=values,
        cmap="jet",
        alpha=0.6,
    )
    ax[1].set_title(f"{wing_name} Redistributed Points")
    ax[1].legend()
    ax[1].axis("equal")
    plt.tight_layout()
    plt.show()


def get_adjacency_list(faces):
    # Determine number of nodes
    num_nodes = max(max(face) for face in faces) + 1

    # Step 1: Initialize adjacency list
    adjacency_list = [set() for _ in range(num_nodes)]

    # Step 2: Populate adjacency list
    for face in faces:
        for i, node in enumerate(face):
            for j in range(len(face)):
                if i != j:  # Avoid self-connections
                    adjacency_list[node].add(face[j])

    # Step 3: Convert sets to sorted lists (optional for consistency)
    adjacency_list = [sorted(list(neighbors)) for neighbors in adjacency_list]

    return adjacency_list


def compute_unit_square(n):
    if n < 4:
        raise ValueError("Need at least 4 points to include all square corners.")

    # Determine base number of points per edge and remainder
    base = n // 4
    remainder = n % 4

    # Distribute remainder to edges (some edges may get one extra point)
    edge_counts = [base + (1 if i < remainder else 0) for i in range(4)]

    points = []

    # Define edge endpoints
    edges = [
        ((0, 0), (1, 0)),  # bottom
        ((1, 0), (1, 1)),  # right
        ((1, 1), (0, 1)),  # top
        ((0, 1), (0, 0)),  # left
    ]

    for count, (start, end) in zip(edge_counts, edges):
        xs = np.linspace(start[0], end[0], count, endpoint=False)
        ys = np.linspace(start[1], end[1], count, endpoint=False)
        edge_points = np.stack((xs, ys), axis=1)
        points.append(edge_points)

    return np.vstack(points)


def compute_unit_circle(n):
    if n < 3:
        raise ValueError("Need at least 3 points to form a circle.")

    # Generate angles for the points
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Stack the coordinates
    points = np.column_stack((x, y))

    return points


def circular_map_to_square(map):
    """
    Convert a circular map to a square map by applying a transformation.
    """
    # Compute theta and r coordinates as x and y coordinates
    theta = np.arctan2(map[:, 1], map[:, 0])
    r = np.sqrt(map[:, 0] ** 2 + map[:, 1] ** 2)

    # Normalize theta between 0 and 1
    theta = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    return np.column_stack((theta, r))


def plot_planar_map(nodes, faces):
    fig, ax = plt.subplots()
    plt.grid(True)

    # Plot each face as connected black edges
    for face in faces:
        # Get the positions of the nodes in the face
        face_nodes = nodes[face]
        # Close the loop by appending the first point at the end
        face_nodes = np.vstack([face_nodes, face_nodes[0]])
        ax.plot(face_nodes[:, 0], face_nodes[:, 1], "k-")

    # Plot nodes as blue dots
    ax.scatter(nodes[:, 0], nodes[:, 1], s=20, c="blue", marker="o")

    ax.set_aspect("equal")
    plt.show()


def compute_global_stiffness_matrix(adj_list):
    """
    Compute the global stiffness matrix using a sparse matrix representation.

    Parameters
    ----------
    adj_list : list of lists
        Adjacency list representing the mesh connectivity.

    Returns
    -------
    K_global : csr_matrix, shape (n_nodes, n_nodes)
        Sparse global stiffness matrix.
    """
    n_nodes = len(adj_list)
    K_global = lil_matrix(
        (n_nodes, n_nodes)
    )  # Use LIL format for efficient row-wise construction

    for i in range(n_nodes):
        for j in adj_list[i]:
            K_global[i, j] -= 1
            K_global[i, i] += 1

    return (
        K_global.tocsr()
    )  # Convert to CSR format for efficient arithmetic and solving


def compute_elastic_equilibrium_planar_map(
    boundary_nodes,
    adj_list,
    boundary_shape="square",
):
    # Compute planar boundary points
    if boundary_shape == "square":
        boundary_points = compute_unit_square(len(boundary_nodes))
    elif boundary_shape == "circular":
        boundary_points = compute_unit_circle(len(boundary_nodes))

    # Compute the global stiffness matrix
    K = compute_global_stiffness_matrix(adj_list).tolil()

    gx = np.zeros((K.shape[0], 1))
    gy = np.zeros((K.shape[0], 1))

    for point, node in zip(boundary_points, boundary_nodes):
        K.rows[node] = [node]
        K.data[node] = [1.0]
        gx[node] = point[0]
        gy[node] = point[1]

    K = K.tocsr()  # Convert back to CSR for solving
    u = spsolve(K, gx)
    v = spsolve(K, gy)
    new_points = np.column_stack((u, v))
    return new_points


def compute_local_density(points, adj_list):
    """
    Estimate the local density of each point as the inverse of the average distance to its neighbors.
    """
    n_nodes = len(points)
    densities = np.zeros(n_nodes)

    for i in range(n_nodes):
        neighbors = adj_list[i]
        if not neighbors:
            densities[i] = 1e-8  # avoid division by zero
            continue
        distances = [np.linalg.norm(points[i] - points[j]) for j in neighbors]
        avg_dist = np.mean(distances)
        densities[i] = 1.0 / (avg_dist + 1e-8)  # avoid division by zero
    return densities


def compute_weighted_stiffness_matrix(adj_list, densities, density_exp=1.0):
    """
    Compute stiffness matrix with weights inversely proportional to local density.
    """
    from scipy.sparse import lil_matrix

    n_nodes = len(adj_list)
    K_global = lil_matrix((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in adj_list[i]:
            weight = 1.0 / (
                0.5 * (densities[i] + densities[j]) + 1e-8
            )  # spring stiffness
            K_global[i, j] -= weight**density_exp
            K_global[i, i] += weight**density_exp
    return K_global.tocsr()


def redistribute_points_with_equilibrium_constrain_density(
    points, boundary_nodes, adj_list, density_exp=1.0
):
    x = points[:, 0]
    y = points[:, 1]

    # Step 1: Compute local densities
    densities = compute_local_density(points, adj_list)

    # Step 2: Compute stiffness matrix with density-based weights
    K = compute_weighted_stiffness_matrix(adj_list, densities, density_exp).tolil()

    # Step 3: Apply boundary conditions
    gx = np.zeros_like(x)
    gy = np.zeros_like(y)
    for node in boundary_nodes:
        K.rows[node] = [node]
        K.data[node] = [1.0]
        gx[node] = x[node]
        gy[node] = y[node]

    # Step 4: Solve and update
    K = K.tocsr()
    u = spsolve(K, gx)
    v = spsolve(K, gy)
    new_points = np.column_stack((u, v))

    return new_points
