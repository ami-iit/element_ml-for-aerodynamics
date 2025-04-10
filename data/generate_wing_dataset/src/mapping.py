import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.neighbors import KernelDensity
from scipy.stats import rankdata


def cartesian_to_polar(v):
    psi = np.arccos(v[:, 2])
    theta = np.arctan2(v[:, 1], v[:, 0])
    return np.stack((theta, psi), axis=1)


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


def redistribute_points(
    points,
    bandwidth=0.1,
    num_iterations=50,
    step_size=0.05,
    alpha=1.0,
    beta=1.0,
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
    n_points = points.shape[0]

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
            # For each neighbor, compute how far the current distance deviates.
            for j in adj_list[i]:
                d_vec = pt - points[j]
                d_current = np.linalg.norm(d_vec)
                if d_current < 1e-8 or d_current > 2.5:
                    continue
                # The force is proportional to (current_distance - original_distance)
                # and acts in the direction of d_vec (normalized).
                distance_force += d_current * (d_vec / d_current)
            # No normalization here so that the magnitude reflects the deviation.

            # --- 3. Combine the Forces and Update the Point ---
            total_force = alpha * density_force + beta * distance_force
            # We subtract the total force: density_force moves away from density peaks,
            # and distance_force adjusts the spacing to match the original distances.
            new_points[i] = pt - step_size * total_force

        points = new_points.copy()
        map2d_min = np.min(original_points, axis=0)
        map2d_ptp = np.ptp(original_points, axis=0)
        map2dr_min = np.min(points, axis=0)
        map2dr_ptp = np.ptp(points, axis=0)
        points = (points - map2dr_min) / map2dr_ptp * map2d_ptp + map2d_min
        print(
            f"Redistribution progress: {it + 1}/{num_iterations}",
            end="\r",
            flush=True,
        )
    return points


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


def density_cdf_warping(points):
    # Normalize points to [0,1] space
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    normalized_points = (points - min_vals) / (max_vals - min_vals)

    # Compute density along Y-axis (for X-scaling)
    kde_x = KernelDensity(bandwidth="scott", kernel="gaussian")
    kde_x.fit(normalized_points[:, 1].reshape(-1, 1))  # Density along Y-axis
    density_x = np.exp(kde_x.score_samples(normalized_points[:, 1].reshape(-1, 1)))

    # Compute density along X-axis (for Y-scaling)
    kde_y = KernelDensity(bandwidth="scott", kernel="gaussian")
    kde_y.fit(normalized_points[:, 0].reshape(-1, 1))  # Density along X-axis
    density_y = np.exp(kde_y.score_samples(normalized_points[:, 0].reshape(-1, 1)))

    # Convert densities into CDF values (using ranking)
    cdf_x = rankdata(density_x) / len(density_x)
    cdf_y = rankdata(density_y) / len(density_y)

    # Apply CDF warping (replacing coordinates with their mapped CDF positions)
    warped_x = np.interp(
        normalized_points[:, 0], np.sort(normalized_points[:, 0]), np.sort(cdf_x)
    )
    warped_y = np.interp(
        normalized_points[:, 1], np.sort(normalized_points[:, 1]), np.sort(cdf_y)
    )

    warped_points = np.column_stack((warped_x, warped_y))

    # Scale back to original space
    redistributed_points = warped_points * (max_vals - min_vals) + min_vals
    return redistributed_points


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
