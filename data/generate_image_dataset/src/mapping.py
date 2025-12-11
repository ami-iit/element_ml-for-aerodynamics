import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity, NearestNeighbors


def cartesian_to_polar(v):
    psi = np.arccos(v[:, 2])
    theta = np.arctan2(v[:, 1], v[:, 0])
    return np.stack((theta, psi), axis=1)


def plot_mapping(original_map, redistributed_map, values, config_name, surface_name):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(
        original_map[:, 0],
        original_map[:, 1],
        c=values,
        cmap="jet",
        alpha=0.6,
        label="Original",
    )
    ax[0].set_title(f"{config_name}-{surface_name} Original Points")
    ax[0].legend()
    ax[0].axis("equal")
    ax[1].scatter(
        redistributed_map[:, 0],
        redistributed_map[:, 1],
        c=values,
        cmap="jet",
        alpha=0.6,
        label="Redistributed",
    )
    ax[1].set_title(f"{config_name}-{surface_name} Redistributed Points")
    ax[1].legend()
    ax[1].axis("equal")
    plt.tight_layout()
    plt.show()


def plot_single_mapping(map, values):
    plt.rcParams["text.usetex"] = True
    plt.figure(figsize=(12, 8))
    plt.scatter(map[:, 0], map[:, 1], c=values, cmap="jet", alpha=0.6)
    plt.xlabel(r"$\theta$ [rad]", fontsize=28)
    plt.ylabel(r"$\psi$ [rad]", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.show()


def redistribute_points(
    points,
    bandwidth=0.1,
    num_iterations=50,
    step_size=0.05,
    alpha=1.0,
    beta=1.0,
    k_neighbors=5,
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
    n_points = points.shape[0]

    # Compute each point's original k nearest neighbors (excluding itself)
    nn = NearestNeighbors(
        n_neighbors=k_neighbors + 1
    )  # +1 because the first neighbor is itself
    nn.fit(original_points)
    distances, indices = nn.kneighbors(original_points)

    # Store original neighbors and their distances for each point
    original_neighbors = {}
    for i in range(n_points):
        # Skip the first index (i itself)
        original_neighbors[i] = []
        for j in range(1, k_neighbors + 1):
            neighbor_idx = indices[i, j]
            orig_dist = distances[i, j]
            original_neighbors[i].append((neighbor_idx, orig_dist))

    # Main iterative update
    for it in range(num_iterations):
        # Fit KDE to the current points
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(points)

        new_points = np.zeros_like(points)
        epsilon = 1e-3  # for finite differences

        for i, pt in enumerate(points):
            # --- 1. Compute Density Force ---
            # Estimate the density gradient via finite differences.
            density_grad = np.zeros(2)
            for dim in range(2):
                e = np.zeros(2)
                e[dim] = epsilon
                d_plus = np.exp(kde.score_samples((pt + e).reshape(1, -1)))[0]
                d_minus = np.exp(kde.score_samples((pt - e).reshape(1, -1)))[0]
                density_grad[dim] = (d_plus - d_minus) / (2 * epsilon)
            # Normalize (if nonzero) so that the step size remains controlled.
            norm_grad = np.linalg.norm(density_grad)
            if norm_grad > 1e-12:
                density_force = density_grad / norm_grad
            else:
                density_force = np.zeros(2)
            # We subtract the density force so that we move away from high density.

            # --- 2. Compute Distance Preservation Force ---
            distance_force = np.zeros(2)
            # For each original neighbor, compute how far the current distance deviates.
            for j, orig_dist in original_neighbors[i]:
                d_vec = pt - points[j]
                d_current = np.linalg.norm(d_vec)
                if d_current < 1e-8:
                    continue
                # The force is proportional to (current_distance - original_distance)
                # and acts in the direction of d_vec (normalized).
                distance_force += (d_current - orig_dist) * (d_vec / d_current)
            # No normalization here so that the magnitude reflects the deviation.

            # --- 3. Combine the Forces and Update the Point ---
            total_force = alpha * density_force + beta * distance_force
            # We subtract the total force: density_force moves away from density peaks,
            # and distance_force adjusts the spacing to match the original distances.
            new_points[i] = pt - step_size * total_force

        points = new_points.copy()
        print(
            f"Redistribution progress: {it + 1}/{num_iterations}",
            end="\r",
            flush=True,
        )
    return points


def redistribute_points_sparse(
    points,
    bandwidth=0.1,
    num_iterations=50,
    step_size=0.05,
    alpha=1.0,
    beta=1.0,
    k_neighbors=5,
    k_density=20,
):
    """
    Scalable redistribution using sparse KDE gradient (O(n·k_density)) and
    vectorized distance-preserving force (O(n·k_neighbors)).

    Parameters
    ----------
    points : ndarray (n, 2)
        Initial 2D coordinates.
    bandwidth : float
        Bandwidth for Gaussian KDE.
    num_iterations : int
        Number of iterations.
    step_size : float
        Integration step size.
    alpha : float
        Weight for density repulsion.
    beta : float
        Weight for distance preservation.
    k_neighbors : int
        Neighbors for distance preservation.
    k_density : int
        Neighbors used for KDE gradient approximation.
    """
    points = points.copy()
    original_points = points.copy()
    n, d = points.shape
    h2 = bandwidth**2

    # --- Precompute original neighbors for distance preservation ---
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(original_points)
    distances, indices = nn.kneighbors(original_points)
    neighbor_indices = indices[:, 1:]
    neighbor_dists = distances[:, 1:]

    for it in range(num_iterations):
        # --- 1. Local neighbors for KDE (recomputed every iteration) ---
        # using current points to follow evolving density
        nn_dens = NearestNeighbors(n_neighbors=k_density + 1)
        nn_dens.fit(points)
        dists_d, idxs_d = nn_dens.kneighbors(points)
        neighbor_idx_dens = idxs_d[:, 1:]  # exclude self
        neighbor_dist_dens = dists_d[:, 1:]

        # --- 2. Sparse KDE gradient (matrix form) ---
        neighbors = points[neighbor_idx_dens]  # (n, k_d, 2)
        diffs = points[:, None, :] - neighbors  # (n, k_d, 2)
        sqdist = np.sum(diffs**2, axis=2)  # (n, k_d)
        weights = np.exp(-sqdist / (2 * h2))  # (n, k_d)
        grad = np.sum(weights[..., None] * diffs / h2, axis=1)  # (n, 2)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        density_force = -grad / norms  # move away from high density

        # --- 3. Distance-preservation force (matrix form) ---
        neighbors = points[neighbor_indices]  # (n, k, 2)
        diffs = points[:, None, :] - neighbors  # (n, k, 2)
        dists = np.linalg.norm(diffs, axis=2, keepdims=True)
        dists_safe = np.maximum(dists, 1e-8)
        unit = diffs / dists_safe
        distance_force = np.sum(
            (dists[..., 0] - neighbor_dists)[..., None] * unit, axis=1
        )

        # --- 4. Update ---
        total_force = alpha * density_force + beta * distance_force
        points -= step_size * total_force

        print(f"Iteration {it + 1}/{num_iterations}", end="\r", flush=True)

    return points


def laplace_beltrami_smoothing(
    points,
    faces,
    num_iterations=10,
    step_size=0.1,
):
    """
    Smooth the 3D points on a sphere using Laplace-Beltrami smoothing.
    Parameters
    ----------
    points : ndarray (n, 3)
        Initial 3D coordinates of the mesh.
    faces : ndarray (m, 3)
        Triangular faces of the mesh.
    num_iterations : int
        Number of smoothing iterations.
    step_size : float
        Step size for each smoothing iteration.
    Returns
    Returns
    -------
    smoothed_points : ndarray (n, 3)
        Smoothed 3D coordinates of the mesh.
    """
    smoothed_points = points.copy()
    for _ in range(num_iterations):
        laplacian = compute_laplacian(smoothed_points, faces)
        smoothed_points -= step_size * laplacian
    return smoothed_points


from scipy.sparse import coo_matrix


def laplacian_smoothing_with_internal_pressure(V, F, num_iters=100, step_size=0.1):
    """
    Laplacian-based relaxation of a mesh constrained to the unit sphere.

    Parameters
    ----------
    V : (n, 3) ndarray
        Vertex coordinates.
    F : (m, 3) ndarray
        Triangular faces (indices into V).
    num_iters : int
        Number of relaxation iterations.
    step_size : float
        Smoothing step size.

    Returns
    -------
    V : (n, 3) ndarray
        Relaxed vertex positions on the sphere.
    """

    # Center points to lie in [-1, 1]
    V = V - V.mean(axis=0)

    n = V.shape[0]

    # --- 1. Build adjacency matrix from faces ---
    i = np.hstack([F[:, 0], F[:, 1], F[:, 2]])
    j = np.hstack([F[:, 1], F[:, 2], F[:, 0]])
    data = np.ones(len(i))
    A = coo_matrix((data, (i, j)), shape=(n, n))
    A = A + A.T  # make symmetric (undirected edges)
    A.setdiag(0)

    # --- 2. Build Laplacian: L = D - A ---
    D = np.array(A.sum(axis=1)).flatten()
    L = coo_matrix(np.diag(D)) - A

    # Convert to dense for clarity (could stay sparse for large meshes)
    # L = L.toarray()

    # --- 3. Iterative Laplacian smoothing + spherical projection ---
    for i in range(num_iters):
        V = V - step_size * (
            (L @ V)
            - 2.0
            / (
                (V - V.mean(axis=0))
                / np.linalg.norm(V - V.mean(axis=0), axis=1, keepdims=True)
            )
        )  # smooth positions
        # V /= np.linalg.norm(V, axis=1, keepdims=True)  # project to unit sphere
        print(
            f"Laplace-Beltrami smoothing iter: {i + 1}/{num_iters}",
            end="\r",
            flush=True,
        )
    # V /= np.linalg.norm(V, axis=1, keepdims=True)  # project to unit sphere

    return V


def laplacian_smoothing_on_sphere(V, F, num_iters=100, step_size=0.1):
    """
    Laplacian-based relaxation of a mesh constrained to the unit sphere.

    Parameters
    ----------
    V : (n, 3) ndarray
        Vertex coordinates.
    F : (m, 3) ndarray
        Triangular faces (indices into V).
    num_iters : int
        Number of relaxation iterations.
    step_size : float
        Smoothing step size.

    Returns
    -------
    V : (n, 3) ndarray
        Relaxed vertex positions on the sphere.
    """

    # Normalize points to lie on the unit sphere
    V = V - V.mean(axis=0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    n = V.shape[0]

    # --- 1. Build adjacency matrix from faces ---
    i = np.hstack([F[:, 0], F[:, 1], F[:, 2]])
    j = np.hstack([F[:, 1], F[:, 2], F[:, 0]])
    data = np.ones(len(i))
    A = coo_matrix((data, (i, j)), shape=(n, n))
    A = A + A.T  # make symmetric (undirected edges)
    A.setdiag(0)

    # --- 2. Build Laplacian: L = D - A ---
    D = np.array(A.sum(axis=1)).flatten()
    L = coo_matrix(np.diag(D)) - A

    # Convert to dense for clarity (could stay sparse for large meshes)
    # L = L.toarray()

    # --- 3. Iterative Laplacian smoothing + spherical projection ---
    for i in range(num_iters):
        V = V - step_size * (L @ V)  # smooth positions
        V /= np.linalg.norm(V, axis=1, keepdims=True)  # project to unit sphere
        print(
            f"Laplace-Beltrami smoothing iter: {i + 1}/{num_iters}",
            end="\r",
            flush=True,
        )

    return V


def spherical_edge_equalization(V, F, num_iters=100, step_size=0.05):
    """
    Redistribute mesh vertices on a unit sphere to equalize edge lengths
    while maintaining elastic equilibrium.

    Parameters
    ----------
    V : (n, 3) ndarray
        Vertex coordinates.
    F : (m, 3) ndarray
        Triangular faces.
    num_iters : int
        Number of iterations.
    step_size : float
        Gradient descent step size.

    Returns
    -------
    V : (n, 3) ndarray
        Updated vertex positions on the sphere.
    """
    n = V.shape[0]

    # --- Build unique edge list from faces ---
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # --- Iterative relaxation ---
    for i in range(num_iters):
        # Compute edge vectors and lengths
        vecs = V[edges[:, 0]] - V[edges[:, 1]]
        lengths = np.linalg.norm(vecs, axis=1, keepdims=True)
        mean_len = np.mean(lengths)

        # Avoid zero division
        lengths[lengths < 1e-12] = 1e-12

        # Compute edge forces
        # force = (current_length - mean_length) * direction
        force = (lengths - mean_len) ** 2 * (vecs / lengths)

        # Accumulate forces on vertices (sparse scatter)
        Fv = np.zeros_like(V)
        np.add.at(Fv, edges[:, 0], -force)
        np.add.at(Fv, edges[:, 1], force)

        # Update positions
        V = V - step_size * Fv

        # Reproject to unit sphere
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        print(
            f"Spherical edge equalization iter: {i + 1}/{num_iters}",
            end="\r",
            flush=True,
        )

    return V
