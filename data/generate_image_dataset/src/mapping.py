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
