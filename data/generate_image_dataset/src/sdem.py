import numpy as np
from scipy.sparse import coo_array, csr_array, find
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
import open3d as o3d
from matplotlib import cm


def beltrami_coefficient(v, f, map):
    """
    Compute the Beltrami coefficient of a mapping.

    Parameters:
    v : ndarray
        Vertex coordinates (n x 2)
    f : ndarray
        Face indices (m x 3)
    mapping : ndarray
        Mapped coordinates (n x 2 or n x 3)

    Returns:
    mu : ndarray
        Beltrami coefficient
    """
    nf = len(f)
    mi = np.repeat(np.arange(nf), 3)
    mj = f.flatten()

    e1 = v[f[:, 2], 0:2] - v[f[:, 1], 0:2]
    e2 = v[f[:, 0], 0:2] - v[f[:, 2], 0:2]
    e3 = v[f[:, 1], 0:2] - v[f[:, 0], 0:2]

    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1]) / 2.0
    area = np.vstack([area, area, area])

    mx = np.vstack([e1[:, 1], e2[:, 1], e3[:, 1]]) / (2 * area)
    mx = mx.T.flatten()
    my = -np.vstack([e1[:, 0], e2[:, 0], e3[:, 0]]) / (2 * area)
    my = my.T.flatten()

    dx = coo_array((mx, (mi, mj)), shape=(nf, len(v))).tocsr()
    dy = coo_array((my, (mi, mj)), shape=(nf, len(v))).tocsr()

    if map.shape[1] == 3:
        dXdu, dXdv = dx @ map[:, 0], dy @ map[:, 0]
        dYdu, dYdv = dx @ map[:, 1], dy @ map[:, 1]
        dZdu, dZdv = dx @ map[:, 2], dy @ map[:, 2]

        E = dXdu**2 + dYdu**2 + dZdu**2
        G = dXdv**2 + dYdv**2 + dZdv**2
        F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv

        mu = (E - G + 2j * F) / (E + G + 2 * np.sqrt(E * G - F**2))
    else:
        z = map[:, 0] + 1j * map[:, 1]
        dz = (dx - 1j * dy) / 2
        dc = (dx + 1j * dy) / 2
        mu = (dc @ z) / (dz @ z)
        mu[~np.isfinite(mu)] = 1

    return mu


def centroidal_relaxation(V, F, iterations):
    """
    Perform centroidal Voronoi relaxation on a spherical triangular mesh.

    Parameters:
    V : ndarray
        (N x 3) matrix of vertex coordinates (assumed to lie on a sphere)
    F : ndarray
        (M x 3) matrix of triangular face indices
    iterations : int
        Number of relaxation steps

    Returns:
    V_new : ndarray
        (N x 3) matrix of new vertex positions (on sphere)
    """
    V_new = V.copy()

    for iter in range(iterations):
        # Step 1: Compute triangle centroids
        centroids = (V_new[F[:, 0], :] + V_new[F[:, 1], :] + V_new[F[:, 2], :]) / 3

        # Step 2: Compute new vertex positions as average of connected centroids
        V_avg = np.zeros_like(V_new)
        count = np.zeros(V_new.shape[0])

        for i in range(F.shape[0]):
            for j in range(3):
                v_idx = F[i, j]
                V_avg[v_idx, :] += centroids[i, :]
                count[v_idx] += 1

        # Step 3: Normalize by count to get centroidal positions
        V_new = V_avg / count[:, None]

        # Step 4: Project back onto the sphere
        V_new /= np.linalg.norm(V_new, axis=1)[:, None]

        if iter % 1000 == 0:
            print(iter)

    return V_new


def compute_gradient_3D(v, f, g):
    """
    Compute the 3D gradient.

    Parameters:
    v : ndarray
        (N x 3) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices
    g : ndarray
        (N,) array of function values at vertices

    Returns:
    grad : ndarray
        (M x 3) matrix of gradient vectors per face
    """
    if v.shape[1] != 3:
        v = np.hstack([v, np.zeros((v.shape[0], 1))])

    # Compute edges
    e1 = v[f[:, 2], :] - v[f[:, 1], :]
    e2 = v[f[:, 0], :] - v[f[:, 2], :]
    e3 = v[f[:, 1], :] - v[f[:, 0], :]

    # Compute area
    cross12 = np.cross(e1, e2)
    area = np.linalg.norm(cross12, axis=1) / 2
    N = cross12 / (2 * area)[:, None]

    # Compute gradient
    temp = g[f[:, 0], None] * e1 + g[f[:, 1], None] * e2 + g[f[:, 2], None] * e3

    grad = np.cross(N, temp)
    grad /= (2 * area)[:, None]

    return grad


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


def face_area(f, v):
    """Compute the area of each triangular face."""
    v12 = v[f[:, 1]] - v[f[:, 0]]
    v23 = v[f[:, 2]] - v[f[:, 1]]
    v31 = v[f[:, 0]] - v[f[:, 2]]

    a = np.linalg.norm(v12, axis=1)
    b = np.linalg.norm(v23, axis=1)
    c = np.linalg.norm(v31, axis=1)

    s = (a + b + c) / 2.0
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def f2v_area(v, f):
    """
    Face to vertex interpolation with area weighting.

    Parameters:
    v : ndarray
        (N x 3) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices

    Returns:
    M : scipy.sparse.coo_array
        (N x M) sparse matrix for face-to-vertex area weighting
    """
    nv = len(v)
    nf = len(f)

    if v.shape[1] == 2:
        v = np.hstack([v, np.zeros((nv, 1))])

    # Compute face areas
    area = face_area(f, v)

    # Create sparse matrix
    row = np.hstack([f[:, 2], f[:, 0], f[:, 1]])
    col = np.hstack([np.arange(nf), np.arange(nf), np.arange(nf)])
    val = np.hstack([area, area, area])

    M = coo_array((val, (row, col)), shape=(nv, nf))

    # Normalize
    vertex_area_sum = np.array(M.sum(axis=1)).flatten()
    row, col, val = M.row, M.col, M.data
    M = coo_array((val / vertex_area_sum[row], (row, col)), shape=(nv, nf))

    return M


def laplace_beltrami(v, f):
    """
    Compute the cotangent Laplacian.

    Parameters:
    v : ndarray
        (N x 3) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices

    Returns:
    L : scipy.sparse.coo_array
        (N x N) sparse Laplace-Beltrami operator matrix
    """
    nv = len(v)

    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    # Compute edge lengths
    l1 = np.linalg.norm(v[f2] - v[f3], axis=1)
    l2 = np.linalg.norm(v[f3] - v[f1], axis=1)
    l3 = np.linalg.norm(v[f1] - v[f2], axis=1)

    # Heron's formula for triangle areas
    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

    # Compute cotangent weights
    cot12 = (l1**2 + l2**2 - l3**2) / (4 * area)
    cot23 = (l2**2 + l3**2 - l1**2) / (4 * area)
    cot31 = (l1**2 + l3**2 - l2**2) / (4 * area)

    # Construct sparse matrix
    I = np.hstack([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    J = np.hstack([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    V = np.hstack(
        [
            -cot12,
            -cot12,
            -cot23,
            -cot23,
            -cot31,
            -cot31,
            (cot12 + cot31) / 2,
            (cot12 + cot23) / 2,
            (cot31 + cot23) / 2,
        ]
    )

    L = coo_array((V, (I, J)), shape=(nv, nv))
    return L


def cotangent_weight(V, vi, vj, vk):
    """Compute cotangent weight for edge (vi, vj) opposite to vertex vk."""
    v1, v2, v3 = V[vi], V[vj], V[vk]

    # Compute edge vectors
    e1 = v1 - v3
    e2 = v2 - v3

    # Compute cotangent of angle at vk
    cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    sin_angle = np.sqrt(1 - cos_angle**2)
    return cos_angle / (sin_angle + np.finfo(float).eps)  # Avoid division by zero


def laplacian_smoothing(V, F, iterations, lambda_):
    """
    Perform improved Laplacian smoothing with cotangent weights to avoid self-intersections.

    Parameters:
    V : ndarray
        (N x 3) matrix of vertex coordinates (on sphere)
    F : ndarray
        (M x 3) matrix of triangular face indices
    iterations : int
        Number of smoothing steps
    lambda_ : float
        Smoothing factor (small values: 0.05 - 0.2 recommended)

    Returns:
    V_new : ndarray
        (N x 3) matrix of smoothed vertex positions (on sphere)
    """
    V_new = V.copy()
    N = V.shape[0]

    # Compute the adjacency matrix with cotangent weights
    row, col, val = [], [], []
    for i in range(F.shape[0]):
        for j in range(3):
            v1, v2, v3 = F[i, j], F[i, (j + 1) % 3], F[i, (j + 2) % 3]

            # Compute cotangent weights
            cot1 = cotangent_weight(V, v1, v2, v3)
            cot2 = cotangent_weight(V, v2, v1, v3)
            cot3 = cotangent_weight(V, v3, v1, v2)

            # Store adjacency information
            row.extend([v1, v2, v2, v3, v3, v1])
            col.extend([v2, v1, v3, v2, v1, v3])
            val.extend([cot1, cot1, cot2, cot2, cot3, cot3])

    # Construct sparse weight matrix
    W = coo_array((val, (row, col)), shape=(N, N)).tocsr()
    W = W.multiply(1 / W.sum(axis=1))  # Normalize row-wise

    # Smoothing iterations
    for iter in range(iterations):
        V_avg = W @ V_new  # Apply weighted smoothing

        # Move each vertex slightly in the Laplacian direction
        V_new = V_new + lambda_ * (V_avg - V_new)

        # Project back onto sphere
        V_new /= np.linalg.norm(V_new, axis=1, keepdims=True)

        # Print progress
        if iter % 1000 == 0:
            print(f"Iteration {iter} / {iterations} completed.")
    print("Laplacian smoothing completed!")

    return V_new


def linear_beltrami_solver(v, f, mu, landmark, target):
    """
    Linear Beltrami solver.

    Parameters:
    v : ndarray
        (N x 2) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices
    mu : ndarray
        (M,) array of Beltrami coefficients
    landmark : ndarray
        List of landmark indices
    target : ndarray
        (len(landmark) x 2) matrix of target positions

    Returns:
    map : ndarray
        (N x 2) matrix of mapped vertex coordinates
    """
    af = (1 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)
    bf = -2 * np.imag(mu) / (1.0 - np.abs(mu) ** 2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)

    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]

    uxv0 = v[f1, 1] - v[f2, 1]
    uyv0 = v[f2, 0] - v[f1, 0]
    uxv1 = v[f2, 1] - v[f0, 1]
    uyv1 = v[f0, 0] - v[f2, 0]
    uxv2 = v[f0, 1] - v[f1, 1]
    uyv2 = v[f1, 0] - v[f0, 0]

    l = np.sqrt(
        np.column_stack((uxv0**2 + uyv0**2, uxv1**2 + uyv1**2, uxv2**2 + uyv2**2))
    )
    s = np.sum(l, axis=1) * 0.5

    area = np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))

    v00 = (af * uxv0**2 + 2 * bf * uxv0 * uyv0 + gf * uyv0**2) / area
    v11 = (af * uxv1**2 + 2 * bf * uxv1 * uyv1 + gf * uyv1**2) / area
    v22 = (af * uxv2**2 + 2 * bf * uxv2 * uyv2 + gf * uyv2**2) / area
    v01 = (
        af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0
    ) / area
    v12 = (
        af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1
    ) / area
    v20 = (
        af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2
    ) / area

    I = np.hstack([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    J = np.hstack([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    V = np.hstack([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / -2

    A = coo_array((V, (I, J)), shape=(v.shape[0], v.shape[0])).tocsr()

    targetc = target[:, 0] + 1j * target[:, 1]
    b = -A[:, landmark] @ targetc
    b[landmark] = targetc

    A = A.tolil()
    A[landmark, :] = 0
    A[:, landmark] = 0
    A[landmark, landmark] = 1
    A = A.tocsr()

    map_ = spsolve(A, b)
    return np.column_stack((np.real(map_), np.imag(map_)))


def lumped_mass_matrix(v, f):
    """
    Compute the lumped mass matrix for FEM Laplacian.

    Parameters:
    v : ndarray
        (N x 2) matrix of vertex coordinates
    f : ndarray
        (M x 3) matrix of triangular face indices

    Returns:
    A : sparse matrix
        (N x N) lumped mass matrix
    """
    nv = v.shape[0]
    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    l1 = np.linalg.norm(v[f2] - v[f3], axis=1)
    l2 = np.linalg.norm(v[f3] - v[f1], axis=1)
    l3 = np.linalg.norm(v[f1] - v[f2], axis=1)

    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

    II = np.hstack([f1, f2, f3])
    JJ = np.hstack([f1, f2, f3])
    V = np.hstack([area, area, area]) / 3

    A = coo_array((V, (II, JJ)), shape=(nv, nv))
    return A


def mobius_area_correction_spherical(v, f, map):
    """
    Find an optimal Mobius transformation for reducing the area distortion of a spherical conformal parameterization.

    Parameters:
    v: (nv, 3) numpy array of vertex coordinates of a genus-0 closed triangle mesh.
    f: (nf, 3) numpy array of triangulations of the genus-0 closed triangle mesh.
    map: (nv, 3) numpy array of the spherical conformal parameterization.

    Returns:
    map_mobius: (nv, 3) numpy array of the updated spherical conformal parameterization.
    x: The optimal parameters for the Mobius transformation.
    """

    # Compute the area with normalization
    area_v = face_area(f, v)
    area_v = area_v / np.sum(area_v)

    # Project the sphere onto the plane
    p = stereographic(map)
    z = p[:, 0] + 1j * p[:, 1]

    # Function for calculating the area after the Mobius transformation
    def area_map(x):
        transformed_z = ((x[0] + x[1] * 1j) * z + (x[2] + x[3] * 1j)) / (
            (x[4] + x[5] * 1j) * z + (x[6] + x[7] * 1j)
        )
        projected = stereographic(
            np.column_stack([np.real(transformed_z), np.imag(transformed_z)])
        )
        return face_area(f, projected) / np.sum(face_area(f, projected))

    # Objective function: mean(abs(log(area_map./area_v)))
    def d_area(x):
        return finitemean(np.abs(np.log(area_map(x) / area_v)))

    # Optimization setup
    x0 = np.array([1, 0, 0, 0, 0, 0, 1, 0])  # initial guess
    bounds = [(-100, 100)] * 8  # bounds for the parameters

    # Optimization (using scipy's minimize function)
    result = minimize(d_area, x0, bounds=bounds, options={"disp": False})

    # Extract the optimal parameters
    x = result.x

    # Obtain the conformal parameterization with area distortion corrected
    fz = ((x[0] + x[1] * 1j) * z + (x[2] + x[3] * 1j)) / (
        (x[4] + x[5] * 1j) * z + (x[6] + x[7] * 1j)
    )
    map_mobius = stereographic(np.column_stack([np.real(fz), np.imag(fz)]))

    return map_mobius, x


def finitemean(A):
    """Compute the mean of finite values in the array to avoid Inf values."""
    return np.mean(A[np.isfinite(A)])


def stereographic(u):
    """
    Stereographic projection.

    Parameters:
    u: (N, 2) or (N, 3) numpy array. If (N, 2), projects points in the plane to the sphere.
    If (N, 3), projects points on the sphere to the plane.

    Returns:
    v: (N, 2) numpy array of the projected points.
    """
    if u.shape[1] == 1:
        u = np.column_stack([np.real(u), np.imag(u)])

    x = u[:, 0]
    y = u[:, 1]

    if u.shape[1] < 3:
        z = 1 + x**2 + y**2
        v = np.column_stack([2 * x / z, 2 * y / z, (-1 + x**2 + y**2) / z])
    else:
        z = u[:, 2]
        v = np.column_stack([x / (1 - z), y / (1 - z)])

    return v


def finitemean(A):
    """
    Avoid Inf values caused by division by a very small area.

    Parameters:
    A: Array of values.

    Returns:
    m: Mean of finite values in A.
    """
    return np.mean(A[np.isfinite(A)])


def optimal_rotation(v, landmark, target):
    """
    Optimal rotation to reduce landmark mismatch.

    Parameters:
    v: (nv, 3) numpy array of vertex coordinates.
    landmark: List or numpy array of landmark indices.
    target: (nl, 3) numpy array of target landmark positions.

    Returns:
    v: (nv, 3) numpy array of rotated vertex coordinates.
    """

    S = np.copy(v)
    S_landmark = S[landmark, :]

    # Rotation matrices
    def R_x(t):
        return np.array(
            [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
        )

    def R_y(g):
        return np.array(
            [[np.cos(g), 0, np.sin(g)], [0, 1, 0], [-np.sin(g), 0, np.cos(g)]]
        )

    def R_z(h):
        return np.array(
            [[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]]
        )

    # Derivatives of rotation matrices
    def dR_x(t):
        return np.array(
            [[0, 0, 0], [0, -np.sin(t), -np.cos(t)], [0, np.cos(t), -np.sin(t)]]
        )

    def dR_y(g):
        return np.array(
            [[-np.sin(g), 0, np.cos(g)], [0, 0, 0], [-np.cos(g), 0, -np.sin(g)]]
        )

    def dR_z(h):
        return np.array(
            [[-np.sin(h), -np.cos(h), 0], [np.cos(h), -np.sin(h), 0], [0, 0, 0]]
        )

    # Landmark mismatch error function
    def L(w):
        return (
            np.sum((R_x(w[0]) @ R_y(w[1]) @ R_z(w[2]) @ S_landmark.T).T - target) ** 2
        )

    # Initialization
    E = 1
    para_t, para_g, para_h = 0, 0, 0
    dt = 0.001
    step = 0
    L_old = L([para_t, para_g, para_h])

    while E > 1e-6 and step < 1000:
        # Update parameters
        rotation = R_x(para_t) @ R_y(para_g) @ R_z(para_h)
        rotated_landmark = (rotation @ S_landmark.T).T

        grad_t = np.sum(
            2
            * (rotated_landmark - target)
            * ((dR_x(para_t) @ R_y(para_g) @ R_z(para_h) @ S_landmark.T).T)
        )
        grad_g = np.sum(
            2
            * (rotated_landmark - target)
            * ((R_x(para_t) @ dR_y(para_g) @ R_z(para_h) @ S_landmark.T).T)
        )
        grad_h = np.sum(
            2
            * (rotated_landmark - target)
            * ((R_x(para_t) @ R_y(para_g) @ dR_z(para_h) @ S_landmark.T).T)
        )

        para_t -= dt * grad_t
        para_g -= dt * grad_g
        para_h -= dt * grad_h

        # Update landmark mismatch error
        L_temp = L([para_t, para_g, para_h])
        E = abs(L_temp - L_old)
        L_old = L_temp

        step += 1

    # Apply final rotation
    v = (R_x(para_t) @ R_y(para_g) @ R_z(para_h) @ S.T).T

    return v


def plot_mesh(v, f, arg3=None):
    """
    Plot a mesh using Open3D.

    Parameters:
    v: (nv, 3) numpy array of vertex coordinates.
    f: (nf, 3) numpy array of face indices.
    arg3 (optional): (nv,) numpy array of scalar values defined on vertices.
    """

    # Create lines for edges
    lines = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(v)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Create point cloud for vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(v)
    point_cloud.paint_uniform_color([0, 0, 1])  # Blue points

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()

    if arg3 is not None:
        # Normalize the scalar values for color mapping
        arg3_normalized = (arg3 - arg3.min()) / (arg3.max() - arg3.min())

        # Define colormap
        colormap = cm.jet
        colors = colormap(arg3_normalized)[:, :3]  # Extract RGB values

        # Assign vertex colors
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Show the mesh
    # o3d.visualization.draw_geometries([mesh])
    geometries = [
        {"geometry": mesh, "name": "mesh"},
        {"geometry": line_set, "name": "lines"},
        {"geometry": point_cloud, "name": "points"},
    ]
    o3d.visualization.draw(geometries, show_skybox=False)


def regular_triangle(f, v):
    """
    Find the triangle with the most regular 1-ring neighborhood.

    Parameters:
    f: (nf, 3) numpy array of face indices.
    v: (nv, 3) numpy array of vertex coordinates.

    Returns:
    regular_triangle: Index of the most regular triangle.
    """

    nv = v.shape[0]
    nf = f.shape[0]

    # Compute edge lengths
    temp = v[f.flatten(), :].reshape(nf, 3, 3)
    e1 = np.linalg.norm(temp[:, 1, :] - temp[:, 2, :], axis=1)
    e2 = np.linalg.norm(temp[:, 0, :] - temp[:, 2, :], axis=1)
    e3 = np.linalg.norm(temp[:, 0, :] - temp[:, 1, :], axis=1)

    # Compute face regularity measure
    perimeter = e1 + e2 + e3
    R_f = (
        np.abs(e1 / perimeter - 1 / 3)
        + np.abs(e2 / perimeter - 1 / 3)
        + np.abs(e3 / perimeter - 1 / 3)
    )

    # Create face-to-vertex sparse matrix
    row = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    col = np.repeat(np.arange(nf), 3)
    val = np.full(3 * nf, 1 / 3)
    H = csr_array((val, (row, col)), shape=(nv, nf)).tocsr()

    # Compute vertex regularity
    R_v = H @ R_f

    # Compute average vertex regularity for each face
    R_average = (R_v[f[:, 0]] + R_v[f[:, 1]] + R_v[f[:, 2]]) / 3

    # Find the most regular triangle
    regular_triangle = np.argmin(R_average)

    return regular_triangle


def rotate_sphere(f, v, north_f):
    """
    Rotate the sphere such that the triangle `north_f` becomes the north pole.

    Parameters:
    f: (nf, 3) numpy array of face indices.
    v: (nv, 3) numpy array of vertex coordinates.
    north_f: Index of the face to be rotated to the north pole.

    Returns:
    v_temp: Rotated vertex coordinates.
    M: Rotation matrix.
    Minv: Inverse rotation matrix.
    """

    # Compute the centroid of the selected triangle
    center_v = np.mean(v[f[north_f, :]], axis=0)
    center_v /= np.linalg.norm(center_v)  # Normalize to unit sphere

    # Compute rotation about Z-axis
    sin_z = -center_v[1] / np.sqrt(center_v[0] ** 2 + center_v[1] ** 2)
    cos_z = center_v[0] / np.sqrt(center_v[0] ** 2 + center_v[1] ** 2)
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    B = rot_z @ center_v
    b1, b2, b3 = B

    # Compute rotation about Y-axis
    sin_y = -b1 / np.sqrt(b1**2 + b3**2)
    cos_y = b3 / np.sqrt(b1**2 + b3**2)
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    # Final rotation matrix
    M = rot_y @ rot_z

    # Compute inverse rotation matrix
    Minv = np.array([[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]]) @ np.array(
        [[cos_y, 0, -sin_y], [0, 1, 0], [sin_y, 0, cos_y]]
    )

    # Apply transformation
    v_temp = (M @ v.T).T

    return v_temp, M, Minv


def south_pole(f, v, bigtri):
    """
    Find the south pole triangle with respect to a given north pole triangle.

    Parameters:
    f: (nf, 3) numpy array of face indices.
    v: (nv, 3) numpy array of vertex coordinates.
    bigtri: Index of the north pole triangle.

    Returns:
    south_f: Index of the triangle farthest from the given north pole triangle.
    """

    # Compute face centers
    f_center = np.mean(v[f], axis=1)

    # Project onto the sphere
    radius = np.linalg.norm(f_center, axis=1, keepdims=True)
    f_center /= radius  # Normalize to unit sphere

    # Find the most distant face center
    distances = np.sum((f_center - f_center[bigtri]) ** 2, axis=1)
    south_f = np.argmax(distances)

    return south_f


def spherical_conformal_map(v, f, bigtri_idx):
    """
    A linear method for computing spherical conformal map of a genus-0 closed surface.

    Input:
    v: nv x 3 vertex coordinates of a genus-0 triangle mesh
    f: nf x 3 triangulations of a genus-0 triangle mesh
    bigtri_idx: index of the most regular triangle to use as the "big triangle"

    Output:
    map: nv x 3 vertex coordinates of the spherical conformal parameterization
    """

    # Check whether the input mesh is genus-0
    if len(v) - 3 * len(f) / 2 + len(f) != 2:
        raise ValueError("The mesh is not a genus-0 closed surface.")

    # Find the most regular triangle as the "big triangle"
    e1 = np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1)
    e2 = np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1)
    e3 = np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)
    regularity = (
        np.abs(e1 / (e1 + e2 + e3) - 1 / 3)
        + np.abs(e2 / (e1 + e2 + e3) - 1 / 3)
        + np.abs(e3 / (e1 + e2 + e3) - 1 / 3)
    )
    reg_ids = np.argsort(regularity)
    bigtri = reg_ids[bigtri_idx]

    # North pole step: Compute spherical map by solving Laplace equation on a big triangle
    nv = v.shape[0]
    M = cotangent_laplacian(v, f)

    p1, p2, p3 = f[bigtri, 0], f[bigtri, 1], f[bigtri, 2]
    fixed = np.array([p1, p2, p3])

    # Modify the matrix M to enforce boundary conditions
    M = M.tocsr()
    mrow, mcol, mval = find(M[fixed, :])
    M = (
        M
        - csr_array((mval, (fixed[mrow], mcol)), shape=(nv, nv))
        + csr_array(([1, 1, 1], (fixed, fixed)), shape=(nv, nv))
    )

    # Set the boundary condition for the big triangle
    x1, y1 = 0, 0
    x2, y2 = 1, 0
    a = v[p2, :3] - v[p1, :3]
    b = v[p3, :3] - v[p1, :3]
    sin1 = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    ori_h = np.linalg.norm(b) * sin1
    ratio = np.linalg.norm([x1 - x2, y1 - y2]) / np.linalg.norm(a)
    y3 = ori_h * ratio
    x3 = np.sqrt(np.linalg.norm(b) ** 2 * ratio**2 - y3**2)

    # Solve the Laplace equation to obtain a harmonic map
    c = np.zeros(nv)
    d = np.zeros(nv)
    c[p1], c[p2], c[p3] = x1, x2, x3
    d[p1], d[p2], d[p3] = y1, y2, y3
    z = spsolve(M, c + 1j * d)
    z = z - np.mean(z)

    # Inverse stereographic projection
    S = np.column_stack(
        [
            2 * np.real(z) / (1 + np.abs(z) ** 2),
            2 * np.imag(z) / (1 + np.abs(z) ** 2),
            (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
        ]
    )

    # Find optimal big triangle size
    w = (S[:, 0] + 1j * S[:, 1]) / (1 + S[:, 2])

    # Find the index of the southernmost triangle
    index = np.argsort(np.abs(z[f[:, 0]]) + np.abs(z[f[:, 1]]) + np.abs(z[f[:, 2]]))
    inner = index[0]
    if inner == bigtri:
        inner = index[1]

    # Compute the size of the northernmost and southernmost triangles
    NorthTriSide = (
        np.abs(z[f[bigtri, 0]] - z[f[bigtri, 1]])
        + np.abs(z[f[bigtri, 1]] - z[f[bigtri, 2]])
        + np.abs(z[f[bigtri, 2]] - z[f[bigtri, 0]])
    ) / 3

    SouthTriSide = (
        np.abs(w[f[inner, 0]] - w[f[inner, 1]])
        + np.abs(w[f[inner, 1]] - w[f[inner, 2]])
        + np.abs(w[f[inner, 2]] - w[f[inner, 0]])
    ) / 3

    # Rescale to get the best distribution
    z = z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # Inverse stereographic projection
    S = np.column_stack(
        [
            2 * np.real(z) / (1 + np.abs(z) ** 2),
            2 * np.imag(z) / (1 + np.abs(z) ** 2),
            (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
        ]
    )

    if np.isnan(S).sum() != 0:
        # If harmonic map fails due to very bad triangulations, use Tutte map
        S = spherical_tutte_map(f, bigtri)
        print("Harmonic map failed, using Tutte map!")

    # South pole step
    I = np.argsort(S[:, 2])

    # Number of points near the south pole to be fixed
    fixnum = max(round(len(v) / 10), 3)
    fixed = I[: min(len(v), fixnum)]

    # South pole stereographic projection
    P = np.column_stack([S[:, 0] / (1 + S[:, 2]), S[:, 1] / (1 + S[:, 2])])

    # Compute the Beltrami coefficient
    mu = beltrami_coefficient(P, f, v)

    # Compose the map with another quasi-conformal map to cancel the distortion
    map = linear_beltrami_solver(P, f, mu, fixed, P[fixed, :])

    if np.isnan(map).sum() != 0:
        # If the result has NaN entries, increase the number of boundary constraints and run again
        fixnum = fixnum * 5
        fixed = I[: min(len(v), fixnum)]
        map = linear_beltrami_solver(P, f, mu, fixed, P[fixed, :])

        if np.isnan(map).sum() != 0:
            map = P  # Use the old result

    z = map[:, 0] + 1j * map[:, 1]

    # Inverse south pole stereographic projection
    map = np.column_stack(
        [
            2 * np.real(z) / (1 + np.abs(z) ** 2),
            2 * np.imag(z) / (1 + np.abs(z) ** 2),
            -(np.abs(z) ** 2 - 1) / (1 + np.abs(z) ** 2),
        ]
    )

    return map


def spherical_tutte_map(f, bigtri=1):
    """
    Compute the spherical Tutte map using the Tutte Laplacian.

    Parameters:
    f : ndarray
        (M x 3) matrix of triangular face indices
    bigtri : int, optional
        Index of the big triangle (default is 1)

    Returns:
    map : ndarray
        (N x 3) matrix of mapped vertex coordinates
    """
    nv = np.max(f) + 1
    nf = f.shape[0]

    I = f.ravel()
    J = f[:, [1, 2, 0]].ravel()
    V = np.ones(nf * 3) / 2

    W = coo_array((V, (I, J)), shape=(nv, nv)).tocsr()
    M = W + W.T - np.diag(W.sum(axis=1).A1)

    boundary = f[bigtri, :3]
    M = M.tolil()
    M[boundary, :] = 0
    M[boundary, boundary] = 1

    b = np.zeros(nv, dtype=complex)
    b[boundary] = np.exp(1j * 2 * np.pi * np.arange(3) / 3)

    z = spsolve(M.tocsr(), b)
    z -= np.mean(z)

    S = np.column_stack(
        (
            2 * np.real(z) / (1 + np.abs(z) ** 2),
            2 * np.imag(z) / (1 + np.abs(z) ** 2),
            (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
        )
    )

    w = S[:, :2] / (1 + S[:, 2][:, None])
    sorted_indices = np.argsort(np.sum(np.abs(z[f]), axis=1))
    inner = sorted_indices[0] if sorted_indices[0] != bigtri else sorted_indices[1]

    NorthTriSide = np.mean(np.abs(np.diff(z[f[bigtri]], axis=0)))
    SouthTriSide = np.mean(np.abs(np.diff(w[f[inner]], axis=0)))

    z *= np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    return np.column_stack(
        (
            2 * np.real(z) / (1 + np.abs(z) ** 2),
            2 * np.imag(z) / (1 + np.abs(z) ** 2),
            (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
        )
    )


def stereographic_projection(v):
    """
    Perform stereographic projection and inverse stereographic projection.

    Parameters:
    v : ndarray
        (N x 3) or (N x 2) matrix of vertex coordinates

    Returns:
    p : ndarray
        (N x 2) or (N x 3) matrix of projected coordinates
    """
    if v.shape[1] == 3:
        p = np.column_stack((v[:, 0] / (1 - v[:, 2]), v[:, 1] / (1 - v[:, 2])))
        p[np.isnan(p)] = np.inf
        return p
    else:
        if v.shape[1] == 1:
            v = np.column_stack((np.real(v), np.imag(v)))
        z = 1 + v[:, 0] ** 2 + v[:, 1] ** 2
        p = np.column_stack(
            (2 * v[:, 0] / z, 2 * v[:, 1] / z, (-1 + v[:, 0] ** 2 + v[:, 1] ** 2) / z)
        )
        mask = np.isnan(z) | (~np.isfinite(z))
        p[mask, 0] = 0
        p[mask, 1] = 0
        p[mask, 2] = 1
        return p


def update_and_correct_overlap(f, S, r, bigtri, dr, dt):
    """
    Overlap correction scheme.

    Parameters:
    f : ndarray
        (M x 3) matrix of triangular face indices
    S : ndarray
        (N x 3) matrix of initial sphere coordinates
    r : ndarray
        (N x 3) matrix of current sphere coordinates
    bigtri : int
        Index of the big triangle
    dr : ndarray
        (N x 3) update direction for r
    dt : float
        Step size

    Returns:
    r_new : ndarray
        (N x 3) updated and corrected sphere coordinates
    """
    delta = 0.1  # Truncation parameter
    r_ori = r.copy()
    f_ori = f.copy()
    flag = True

    while flag:
        r = r_ori + dt * dr
        r /= np.linalg.norm(r, axis=1, keepdims=True)  # Normalize to sphere

        # North pole step
        S_rotN, _, _ = rotate_sphere(f, S, bigtri)
        r_rotN, _, RN_inv = rotate_sphere(f, r, bigtri)

        p_S_rotN = stereographic_projection(S_rotN)
        p_r_rotN = stereographic_projection(r_rotN)

        f = np.delete(f, bigtri, axis=0)  # Puncture the triangle

        sorted_indices = np.argsort(S_rotN[:, 2])[::-1]
        ig_N = sorted_indices[: max(round(len(S) / 10), 3)]
        ignore_index_N = np.where(
            np.isin(f[:, 0], ig_N) | np.isin(f[:, 1], ig_N) | np.isin(f[:, 2], ig_N)
        )[0]

        mu_N = beltrami_coefficient(p_S_rotN, f, p_r_rotN)
        overlap_N = np.setdiff1d(np.where(np.abs(mu_N) >= 1)[0], ignore_index_N)

        if overlap_N.size == 0:
            r_newN = r_rotN
            north_success = True
        else:
            mu_N[overlap_N] *= (1 - delta) / np.abs(mu_N[overlap_N])
            p_lbsN = linear_beltrami_solver(p_S_rotN, f, mu_N, ig_N, p_r_rotN[ig_N])
            mu_N = beltrami_coefficient(p_S_rotN, f, p_lbsN)
            overlap_N = np.setdiff1d(np.where(np.abs(mu_N) >= 1)[0], ignore_index_N)

            if overlap_N.size == 0:
                north_success = True
            else:
                dt /= 2
                north_success = False

            r_newN = stereographic_projection(p_lbsN)

        r_newN = (RN_inv @ r_newN.T).T  # Rotate back
        f = f_ori.copy()

        if north_success:
            south_f = south_pole(f, r, bigtri)
            S_rotS, _, _ = rotate_sphere(f, S, south_f)
            r_rotS, _, RS_inv = rotate_sphere(f, r_newN, south_f)

            p_S_rotS = stereographic_projection(S_rotS)
            p_r_rotS = stereographic_projection(r_rotS)

            f = np.delete(f, south_f, axis=0)
            sorted_indices = np.argsort(S_rotS[:, 2])[::-1]
            ig_S = sorted_indices[: max(round(len(S) / 10), 3)]
            ignore_index_S = np.where(
                np.isin(f[:, 0], ig_S) | np.isin(f[:, 1], ig_S) | np.isin(f[:, 2], ig_S)
            )[0]

            mu_S = beltrami_coefficient(p_S_rotS, f, p_r_rotS)
            overlap_S = np.setdiff1d(np.where(np.abs(mu_S) >= 1)[0], ignore_index_S)

            if overlap_S.size == 0:
                r_newS = r_rotS
                south_success = True
            else:
                mu_S[overlap_S] *= (1 - delta) / np.abs(mu_S[overlap_S])
                p_lbsS = linear_beltrami_solver(p_S_rotS, f, mu_S, ig_S, p_r_rotS[ig_S])
                mu_S = beltrami_coefficient(p_S_rotS, f, p_lbsS)
                overlap_S = np.setdiff1d(np.where(np.abs(mu_S) >= 1)[0], ignore_index_S)

                if overlap_S.size == 0:
                    south_success = True
                else:
                    dt /= 2
                    south_success = False

                r_newS = stereographic_projection(p_lbsS)

        if north_success and south_success:
            flag = False

        if dt < 1e-10:
            flag = False

    if dt < 1e-10:
        return r_ori
    else:
        return (RS_inv @ r_newS.T).T


def SDEM(v, f, population, S=None, dt=0.1, epsilon=1e-3, max_iter=200):
    """
    Compute the Spherical Density-Equalizing Map (SDEM) for genus-0 closed surfaces.

    Parameters:
    v : ndarray
        (nv, 3) vertex coordinates of the spherical surface mesh.
    f : ndarray
        (nf, 3) triangulations of the spherical surface mesh.
    population : ndarray
        (nf, 1) positive quantity per face.
    S : ndarray, optional
        (nv, 3) vertex coordinates of the initial spherical conformal parameterization.
        If None, it is computed automatically.
    dt : float, optional
        Step size, default is 0.1.
    epsilon : float, optional
        Stopping parameter, default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations, default is 200.

    Returns:
    map : ndarray
        (nv, 3) vertex coordinates of the spherical density-equalizing map.
    """
    if S is None:
        S1 = spherical_conformal_map(v, f)
        S = mobius_area_correction_spherical(v, f, S1)

    # Normalize input spherical parameterization
    r = S / np.linalg.norm(S, axis=1, keepdims=True)

    bigtri = regular_triangle(f, r)

    # Compute density
    rho_f = population / face_area(f, r)
    rho_v = f2v_area(r, f) @ rho_f

    step = 0
    rho_v_error = np.std(rho_v) / np.mean(rho_v)
    print("Step     std(rho)/mean(rho)")
    print(f"{step}        {rho_v_error:.6e}")

    counter = 0

    while rho_v_error >= epsilon and step < max_iter and counter < 10:
        old_rho_v_error = rho_v_error

        # Update rho
        L = laplace_beltrami(r, f)
        A = lumped_mass_matrix(r, f)
        rho_v_temp = spsolve(A + dt * L, A @ rho_v)

        # Update density gradient
        grad_rho_temp_f = compute_gradient_3D(r, f, rho_v_temp)
        grad_rho_temp_v = f2v_area(r, f) @ grad_rho_temp_f

        # Update displacement
        dr = -grad_rho_temp_v / rho_v_temp[:, None]
        dr_proj = dr - np.sum(dr * r, axis=1, keepdims=True) * r

        r = update_and_correct_overlap(f, S, r, bigtri, dr_proj, dt)

        step += 1
        rho_v_error = np.std(rho_v_temp) / np.mean(rho_v_temp)
        print(f"{step}        {rho_v_error:.6e}")

        # Re-coupling scheme
        rho_f = population / face_area(f, r)
        rho_v = f2v_area(r, f) @ rho_f

        if abs((rho_v_error - old_rho_v_error) / rho_v_error) < 1e-6:
            counter += 1

    return r
