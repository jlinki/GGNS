import numpy as np
import pymesh
import open3d
from scipy.spatial import Delaunay
import math
from shapely.ops import unary_union, polygonize
import shapely.geometry as geometry

def generate_mesh_from_pcd_hull(data, timestep, mode="subsample", detailed_print=True):
    """

    Args:
        data:
        mode:

    Returns:

    """
    if timestep == "t":
        data_points = data.grid_positions_old.numpy()
    elif timestep == "t+1":
        data_points = data.grid_positions.numpy()
    else:
        raise ValueError
    # subsample the point cloud and create mesh from subsampled point set

    if mode == "subsample":
        points = np.zeros((data_points.shape[0], 3))
        points[:, 0:2] = data_points
        pcd = open3d.t.geometry.PointCloud(points)
        pcd = pcd.to_legacy()
        pcd_subsampled = pcd.voxel_down_sample(0.24)
        hull_vertices = np.asarray(pcd_subsampled.points)[:,0:2]
        tri_mesh = pymesh.triangle()
        tri_mesh.points = hull_vertices
        #tri_mesh.max_area = 0.041 # max area does not work in second iteration
        tri_mesh.max_num_steiner_points = 20
        tri_mesh.verbosity = 0
        tri_mesh.run()
        hull_mesh = tri_mesh.mesh
        if detailed_print:
            print("Number of vertices in Mesh: ", hull_mesh.vertices.shape[0])
        #hull_mesh, info = pymesh.collapse_short_edges(hull_mesh, 0.105)
        #hull_mesh, info = pymesh.split_long_edges(hull_mesh, 0.45)
        del tri_mesh

    # creates convex hull of point cloud and then a mesh from it.
    # This includes many long edges, which is why they are split

    elif mode == "hull_edge_split":
        tri = np.array([10, 20, 30]).reshape(1, 3)
        pointcloud_mesh = pymesh.meshio.form_mesh(data_points, tri)
        hull = pymesh.convex_hull(pointcloud_mesh)
        hull_vertices = hull.nodes
        tri_mesh = pymesh.triangle()
        tri_mesh.points = hull_vertices
        tri_mesh.verbosity = 0
        tri_mesh.run()
        hull_mesh = tri_mesh.mesh
        hull_mesh, info = pymesh.split_long_edges(hull_mesh, 0.4)
        if detailed_print:
            print("Number of vertices in Mesh: ", hull_mesh.vertices.shape[0])

    # Mesh generation directly from point cloud

    elif mode == "hull_max_area":
        hull_vertices = data_points
        tri_mesh = pymesh.triangle()
        tri_mesh.points = hull_vertices
        tri_mesh.keep_convex_hull = False
        tri_mesh.conforming_delaunay = True
        tri_mesh.max_num_steiner_points = 0
        tri_mesh.split_boundary = True
        tri_mesh.verbosity = 0
        tri_mesh.run()
        hull_mesh = tri_mesh.mesh
        #hull_mesh, info = pymesh.collapse_short_edges(hull_mesh, 0.12)
        del hull_vertices
        del tri_mesh

    # generate mesh from point cloud using ball pivoting algo todo not working

    elif mode == "open3d":
        points = np.ones((data_points.shape[0], 3))
        points[:, 0:2] = data_points
        pcd = open3d.t.geometry.PointCloud(points)
        pcd = pcd.to_legacy()
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        hull_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                   pcd,
                   open3d.utility.DoubleVector([radius, radius * 2]))
        #open3d.visualization.draw_geometries([hull_mesh])

        hull_mesh = pymesh.form_mesh(np.asarray(hull_mesh.vertices), np.asarray(hull_mesh.triangles))

    # creating a mesh from the point cloud using delauney triangulation

    elif mode == "delaunay":
        hull_mesh = Delaunay(data_points)
        hull_mesh = pymesh.form_mesh(np.asarray(hull_mesh.points), np.asarray(hull_mesh.vertices))

    # creates a mesh from point cloud using alpha shapes

    elif mode == "alpha_shapes":
        points = data_points
        points_2d = [point for point in points]
        concave_hull, edge_points, edges, triangles_reduced = alpha_shape(points_2d, alpha=10.0)
        # # PLOTTING
        # from matplotlib.collections import LineCollection
        # import pylab as pl
        # #print concave_hull
        # lines = LineCollection(edge_points)
        # pl.figure(figsize=(10,10))
        # pl.title('Alpha={0} Delaunay triangulation'.format(0.4))
        # pl.gca().add_collection(lines)
        # delaunay_points = np.array([point for point in points_2d])
        # pl.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', color='#f16824')
        # pl.show()

        #_ = pl.plot(x,y,'o', color='#f16824')
        #alpha_shape = alphashape.alphashape(points_2d, 3.0)
        #x, y = concave_hull.exterior.coords.xy
        # fig, ax = plt.subplots()
        # ax.scatter(*zip(*points_2d))
        # ax.scatter(x, y, c='red')
        # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
        # plt.show()
        #print(alpha_shape)
        #hull_mesh = Delaunay(np.array((x,y)).transpose())
        hull_mesh = pymesh.form_mesh(points, np.asarray(triangles_reduced))
        hull_mesh, _ = pymesh.remove_duplicated_faces(hull_mesh, fins_only=False)
        hull_mesh = pymesh.resolve_self_intersection(hull_mesh, engine='auto')
        hull_mesh, _ = pymesh.remove_isolated_vertices(hull_mesh)
        hull_mesh, _ = pymesh.remove_degenerated_triangles(hull_mesh, num_iterations=5)
        #hull_mesh, _ = pymesh.split_long_edges(hull_mesh, 0.35)

    # creates a mesh from a subsampled point cloud using alpha shapes

    elif mode == "alpha_shapes_sub":
        points = np.zeros((data_points.shape[0], 3))
        points[:, 0:2] = data_points
        pcd = open3d.t.geometry.PointCloud(points)
        pcd = pcd.to_legacy()
        pcd_subsampled = pcd.voxel_down_sample(0.24)
        hull_vertices = np.asarray(pcd_subsampled.points)[:, 0:2]
        points = hull_vertices
        points_2d = [point for point in points]
        concave_hull, edge_points, edges, triangles_reduced = alpha_shape(points_2d, alpha=5.0)
        hull_mesh = pymesh.form_mesh(points, np.asarray(triangles_reduced))
        hull_mesh, _ = pymesh.split_long_edges(hull_mesh, 0.35)
        hull_mesh, _ = pymesh.remove_duplicated_faces(hull_mesh, fins_only=False)
        hull_mesh = pymesh.resolve_self_intersection(hull_mesh, engine='auto')
        hull_mesh, _ = pymesh.remove_isolated_vertices(hull_mesh)
        hull_mesh, _ = pymesh.remove_degenerated_triangles(hull_mesh, num_iterations=5)
    else:
        raise ValueError

    return hull_mesh


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    triangles_reduced = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        heron = s*(s-a)*(s-b)*(s-c)
        if heron > 0:
            area = math.sqrt(heron)
            circum_r = a*b*c/(4.0*area)
        else:
            circum_r = 0

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            triangles_reduced.append([ia, ib, ic])

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points, edges, triangles_reduced


def alpha_shape_hull_mesh(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point for point in points])
    tri_simple = np.array([10, 20, 30]).reshape(1, 3)
    pointcloud_mesh = pymesh.meshio.form_mesh(coords, tri_simple)
    hull = pymesh.convex_hull(pointcloud_mesh)
    hull_vertices = hull.nodes
    tri_mesh = pymesh.triangle()
    tri_mesh.points = hull_vertices
    tri_mesh.verbosity = 0
    tri_mesh.run()
    hull_mesh = tri_mesh.mesh
    hull_mesh, info = pymesh.split_long_edges(hull_mesh, 0.4)
    coords = hull_mesh.vertices
    #tri = Delaunay(coords)
    edges = set()
    edge_points = []
    triangles_reduced = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in hull_mesh.faces:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        heron = s*(s-a)*(s-b)*(s-c)
        if heron > 0:
            area = math.sqrt(heron)
            circum_r = a*b*c/(4.0*area)
        else:
            circum_r = 0

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            triangles_reduced.append([ia, ib, ic])

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points, edges, triangles_reduced, coords


