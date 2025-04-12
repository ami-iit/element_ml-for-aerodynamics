import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import lil_matrix, csr_matrix, coo_array, csr_array, find
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


def cotangent_laplacian(v, f):
    """
    Compute the cotangent Laplacian of a mesh.

    Parameters:
    v : ndarray
        (N x 3) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices

    Returns:
    L : scipy.sparse.coo_array
        (N x N) sparse cotangent Laplacian matrix
    """
    nv = len(v)

    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    l1 = np.linalg.norm(v[f2] - v[f3], axis=1)
    l2 = np.linalg.norm(v[f3] - v[f1], axis=1)
    l3 = np.linalg.norm(v[f1] - v[f2], axis=1)

    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

    cot12 = (l1**2 + l2**2 - l3**2) / (2 * area)
    cot23 = (l2**2 + l3**2 - l1**2) / (2 * area)
    cot31 = (l1**2 + l3**2 - l2**2) / (2 * area)

    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    II = np.hstack([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    JJ = np.hstack([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    V = np.hstack([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])

    L = coo_array((V, (II, JJ)), shape=(nv, nv))

    return L


def planar_conformal_map(v, f, boundary_poly):
    """
    A linear method for computing spherical conformal map of a genus-0 closed surface.

    Input:
    v: nv x 3 vertex coordinates of a genus-0 triangle mesh
    f: nf x 3 triangulations of a genus-0 triangle mesh
    bigtri_idx: index of the most regular triangle to use as the "big triangle"

    Output:
    map: nv x 3 vertex coordinates of the spherical conformal parameterization
    """

    # Set the boundary polygon as a unit square
    b_points = boundary_poly[0]
    b_nodes = boundary_poly[1]
    boundary = compute_unit_square(len(b_nodes))
    # boundary = compute_unit_circle(len(b_nodes))

    # Compute conformal map by solving Laplace equation on a big square
    nv = v.shape[0]
    M = cotangent_laplacian(v, f)

    fixed = np.array(b_nodes)

    # Modify the matrix M to enforce boundary conditions
    M = M.tocsr()
    mrow, mcol, mval = find(M[fixed, :])
    M = (
        M
        - csr_array((mval, (fixed[mrow], mcol)), shape=(nv, nv))
        + csr_array((np.ones(len(fixed)), (fixed, fixed)), shape=(nv, nv))
    )

    # Set the boundary condition for the big square
    x1, y1 = 0, 0
    x2, y2 = 1, 0
    x = boundary[:, 0]
    y = boundary[:, 1]

    # Solve the Laplace equation to obtain a harmonic map
    c = np.zeros(nv)
    d = np.zeros(nv)
    for i in range(len(b_nodes)):
        node = b_nodes[i]
        c[node] = x[i]
        d[node] = y[i]
    z = spsolve(M, c + 1j * d)
    # z = z - np.mean(z)

    map = np.stack((z.real, z.imag), axis=-1)

    return map


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

    # Close the loop with the last corner
    points.append(np.array([[0, 0]]))  # ensures the last point is corner (0,0)

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


def redistribute_points_with_constrain(
    points,
    boundary_nodes,
    num_iterations=50,
    step_size=0.05,
    adj_list=None,
):
    """
    Redistribute points by combining a density-based force (via KDE) and
    a distance-preservation force (keeping originally close points nearby).

    Parameters
    ----------
    points : ndarray, shape (n_samples, 2)
        The initial 2D coordinates of the points.
    bandwidth : float
        Bandwidth for the Kernel Density Estimator.
    num_iterations : int
        Number of iterations for the update.
    step_size : float
        Step size for each update.
    alpha : float
        Weight for the density-based force.
    beta : float
        Weight for the distance-preservation force.
    k_neighbors : int
        Number of nearest neighbors (based on the original configuration)
        to consider for preserving distances.

    Returns
    -------
    points : ndarray, shape (n_samples, 2)
        The redistributed points after the iterations.
    """
    points = points.copy()
    original_points = points.copy()

    # Main iterative update
    for it in range(num_iterations):
        new_points = np.zeros_like(points)

        for i, pt in enumerate(points):
            # --- 1. Compute Distance Preservation Force ---
            distance_force = np.zeros(2)
            # For each neighbor, compute how far the current distance deviates.
            for j in adj_list[i]:
                d_vec = pt - points[j]
                # The force is proportional to (current_distance - original_distance)
                # and acts in the direction of d_vec (normalized).
                distance_force += d_vec

            # --- 2. Update the Point ---
            distance_force /= len(adj_list[i])
            new_points[i] = pt - step_size * distance_force

        points = new_points.copy()
        # Keep boundary nodes fixed
        for node in boundary_nodes:
            points[node] = original_points[node]
        print(
            f"Redistribution progress: {it + 1}/{num_iterations}",
            end="\r",
            flush=True,
        )
    return points


def plot_planar_conformal_map(nodes, faces):
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


def compute_global_stiffness_matrix_density(adj_list, density):
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
            K_global[i, j] -= density[i]
            K_global[i, i] += density[i]

    return K_global.tocsr()


def redistribute_points_with_equilibrium_constrain(
    points,
    boundary_nodes,
    adj_list,
):
    x = points[:, 0]
    y = points[:, 1]
    K = compute_global_stiffness_matrix(adj_list).tolil()

    gx = np.zeros_like(x)
    gy = np.zeros_like(y)

    for node in boundary_nodes:
        K.rows[node] = [node]
        K.data[node] = [1.0]
        gx[node] = x[node]
        gy[node] = y[node]

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


def compute_weighted_stiffness_matrix_sparse(adj_list, densities, density_exp=1.0):
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
    K = compute_weighted_stiffness_matrix_sparse(
        adj_list, densities, density_exp
    ).tolil()

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
