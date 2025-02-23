import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from matplotlib.path import Path as Path2D
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay


def create_mesh_from_faces(points, faces):
    """Creates a mesh using given points and face connectivity, supporting up to 8 vertices per face."""
    # Convert all faces into triangles
    triangular_faces = []
    for face in faces:
        if len(face) >= 3:
            triangles, ctrl_node = triangulate_face(face, points)
            triangular_faces.extend(triangles)
            points = np.concatenate([points, [ctrl_node]], axis=0)
            # triangular_faces.extend(triangulate_face(face))
    is_used = np.zeros(len(points), dtype=bool)
    for face in triangular_faces:
        for node in face:
            is_used[node] = True
    if not np.all(is_used):
        print("Unused nodes:", np.where(~is_used)[0])
    # Create Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangular_faces)
    # Compute normals for better visualization
    mesh.compute_vertex_normals()
    return mesh


def triangulate_face(face, points):
    """Triangulates a face creating a support node at the centroid."""
    if len(face) < 3:
        raise ValueError("Only faces with 3 or more vertices are supported.")
    ctrl_node = np.mean(points[face], axis=0)
    ctrl_node_idx = len(points)
    triangles = []
    for i in range(len(face)):
        triangles.append([ctrl_node_idx, face[i], face[(i + 1) % len(face)]])
    return triangles, ctrl_node


def get_boundary_edges(mesh):
    """Get the boundary edges (edges that belong to only one triangle)."""
    triangles = np.asarray(mesh.triangles)
    edge_count = {}
    for triangle in triangles:
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges


def visualize_mesh_with_edges(mesh):
    """Visualizes the mesh along with points and edges."""
    # Convert mesh to line set (to visualize edges)
    lines = np.asarray(mesh.triangles)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    line_set = o3d.geometry.LineSet()
    line_set.points = mesh.vertices
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Get boundary edges and convert them to line set
    boundary_edges = get_boundary_edges(mesh)
    boundary_lines = np.array(boundary_edges)
    boundary_line_set = o3d.geometry.LineSet()
    boundary_line_set.points = mesh.vertices
    boundary_line_set.lines = o3d.utility.Vector2iVector(boundary_lines)
    boundary_line_set.paint_uniform_color([1, 0, 0])  # Red color for boundary edges
    # Create point cloud for vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices
    point_cloud.paint_uniform_color([0, 0, 1])  # Blue points
    # Show everything
    o3d.visualization.draw_geometries([mesh, line_set, point_cloud, boundary_line_set])


def generate_boundary_polygons(boundary, nodes):
    """Generates multiple boundary polygons from the boundary edges, handling separate loops."""
    boundary = set(map(tuple, boundary))
    polygons = []
    while boundary:
        # Start a new polygon
        start_edge = boundary.pop()
        polygon = [nodes[start_edge[0]], nodes[start_edge[1]]]
        node_indices = [start_edge[0], start_edge[1]]
        visited = {start_edge[0], start_edge[1]}
        while True:
            last_point = node_indices[-1]
            found_next = False
            for edge in list(boundary):
                if edge[0] == last_point and edge[1] not in visited:
                    next_point = edge[1]
                elif edge[1] == last_point and edge[0] not in visited:
                    next_point = edge[0]
                else:
                    continue
                polygon.append(nodes[next_point])
                node_indices.append(next_point)
                visited.add(next_point)
                boundary.remove(edge)
                found_next = True
                break
            if not found_next:
                for edge in list(boundary):
                    if edge[0] == node_indices[0] or edge[1] == node_indices[0]:
                        boundary.remove(edge)
                break
        polygons.append((np.array(polygon), node_indices))
    return polygons


def close_mesh_boundaries(mesh):
    boundary_edges = np.array(get_boundary_edges(mesh))
    nodes = np.asarray(mesh.vertices)
    boundary_polygons = generate_boundary_polygons(np.array(boundary_edges), nodes)
    new_faces = np.empty((0, 3), dtype=int)
    for poly in boundary_polygons:
        boundary_poly, node_indices = poly
        triangles, control_nodes = mesh_with_delaunay(boundary_poly)
        triangle_list = []
        for triangle in triangles:
            node_list = []
            for node in triangle:
                if node < len(boundary_poly):
                    node_list.append(node_indices[node])
                else:
                    node_list.append(node + len(nodes) - len(boundary_poly))
            triangle_list.append(node_list)
        triangle_faces = np.array(triangle_list)
        new_faces = np.concatenate([new_faces, triangle_faces], axis=0)
        nodes = np.vstack((nodes, control_nodes))
    # Assemble closed mesh
    current_faces = np.asarray(mesh.triangles)
    all_faces = np.concatenate([current_faces, new_faces], axis=0)
    # check for duplicate triangles (also if in different order)
    # all_faces = np.unique(np.sort(all_faces, axis=1), axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(nodes)
    mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    mesh.compute_vertex_normals()
    return mesh


def mesh_with_delaunay(poly):
    poly_2d, centroid, basis_x, basis_y = project_onto_plane_and_convert_to_2d(poly)
    edge_lengths = np.linalg.norm(np.diff(np.vstack((poly, poly[0])), axis=0), axis=1)
    density = np.mean(edge_lengths)
    ctrl_nodes_2d = generate_2d_unif_distribution(poly_2d, density)
    ctrl_nodes = np.array(
        [centroid + u * basis_x + v * basis_y for u, v in ctrl_nodes_2d]
    )
    if len(ctrl_nodes) == 0:
        ctrl_nodes = np.empty((0, 3))
    points2d = np.vstack((poly_2d, ctrl_nodes_2d))
    points3d = np.vstack((poly, ctrl_nodes))
    tri = Delaunay(points2d)
    path = Path2D(poly_2d)
    # Filter out triangles with centroid outside of the polygon and (quasi-)zero area
    valid_tri = []
    for triangle in tri.simplices:
        centroid = np.mean(points2d[triangle], axis=0)
        v1 = points2d[triangle[1]] - points2d[triangle[0]]
        v2 = points2d[triangle[2]] - points2d[triangle[0]]
        tri_area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        if path.contains_point(centroid) and tri_area > 1e-6:
            valid_tri.append(triangle)
    triangles = np.array(valid_tri)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # visualize_mesh_with_edges(mesh)
    return triangles, ctrl_nodes


def visualize_3d_polygon_with_ctrl_nodes(poly, ctrl_nodes):
    points = o3d.utility.Vector3dVector(poly)
    ctrl_points = o3d.utility.Vector3dVector(ctrl_nodes)
    boundary_nodes = o3d.geometry.PointCloud()
    boundary_nodes.points = points
    boundary_nodes.paint_uniform_color([0, 0, 1])  # Blue color for nodes
    control_nodes = o3d.geometry.PointCloud()
    control_nodes.points = ctrl_points
    control_nodes.paint_uniform_color([1, 0, 0])  # Red color for nodes
    o3d.visualization.draw_geometries([boundary_nodes, control_nodes])


def project_onto_plane_and_convert_to_2d(points):
    # Step 1: Compute the centroid
    centroid = np.mean(points, axis=0)
    # Step 2: Compute the normal using PCA
    pca = PCA(n_components=3)
    pca.fit(points - centroid)
    normal = pca.components_[-1]  # Last component is the normal (least variance)
    # Step 3: Create a local 2D coordinate system (basis vectors in the plane)
    basis_x = pca.components_[0]  # First principal component (highest variance)
    basis_y = np.cross(
        normal, basis_x
    )  # Second basis vector (perpendicular to normal & basis_x)
    # Step 4: Project points onto the plane
    projected_points = []
    for P in points:
        d = np.dot((P - centroid), normal)  # Signed distance to plane
        P_proj = P - d * normal  # Move point along normal to the plane
        projected_points.append(P_proj)
    projected_points = np.array(projected_points)
    # Step 5: Convert 3D projected points to 2D coordinates
    points_2d = np.array(
        [
            [np.dot(P - centroid, basis_x), np.dot(P - centroid, basis_y)]
            for P in projected_points
        ]
    )
    return points_2d, centroid, basis_x, basis_y


def generate_2d_unif_distribution(poly, density):
    min_x, min_y = np.min(poly, axis=0)
    max_x, max_y = np.max(poly, axis=0)
    len_x = max_x - min_x
    len_y = max_y - min_y
    num_x = int(np.ceil(len_x / density))
    num_y = int(np.ceil(len_y / density))
    # generate regular grid points
    x = np.linspace(min_x, max_x, num_x)
    y = np.linspace(min_y, max_y, num_y)
    xx, yy = np.meshgrid(x, y)
    points = np.array(list(zip(xx.flatten(), yy.flatten())))
    path = Path2D(poly)
    for point in points:
        distances = np.linalg.norm(poly - point, axis=1)
        if not path.contains_point(point) or np.any(distances < 0.6 * density):
            points = np.delete(
                points, np.where(np.all(points == point, axis=1))[0], axis=0
            )
    return points
