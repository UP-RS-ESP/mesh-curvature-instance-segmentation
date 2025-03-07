from copy import deepcopy
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree as kdtree
from matplotlib import pyplot as pl
from tqdm import trange, tqdm


def color_mapping(var, cmap, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(var)
    if vmax is None:
        vmax = np.nanmax(var)
    norm = pl.Normalize(vmin, vmax)
    return cmap(norm(var))[:, :3]


def graph_coloring(mesh, con, seg):
    vrt = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    rim = np.zeros((np.asarray(mesh.vertices).shape[0], 2), dtype="int")

    # get rim vertices
    rim[tri[con].ravel(), 0] = 1
    rim[tri[~con].ravel(), 1] = 1
    rim = rim.prod(axis=1).astype("bool")
    vrt = vrt[rim]

    # corresponding segment label
    seg = seg[rim]

    # construct neighborhood graph
    tr = kdtree(vrt[:, :2])
    graph = {}
    for la in trange(seg.max() + 1):
        _, ii = tr.query(vrt[seg == la, :2], k=100, workers=-1)
        seg_ii = seg[ii]
        sl = np.argmax(seg_ii != la, axis=1)
        seg_ii = seg_ii[np.arange(sl.shape[0]), sl]
        graph[la] = np.unique(seg_ii).tolist()

    # graph coloring
    color = {}
    for la in graph:
        color_neighbors = {color[i] for i in graph[la] if i in color}
        c = 0
        while c in color_neighbors:
            c += 1
        color[la] = c

    return np.array(list(color.values()))


def erosion_filter(mesh, condition):
    tri = np.asarray(mesh.triangles)
    rim = np.zeros((np.asarray(mesh.vertices).shape[0], 2), dtype="int")
    con = condition.copy()

    # get rim vertices
    rim[tri[con].ravel(), 0] = 1
    rim[tri[~con].ravel(), 1] = 1
    rim = rim.prod(axis=1)

    # rim vertices to rim triangles
    rim = rim[tri]
    rim = rim.sum(axis=1)

    # remove triangles that are part of the rim
    con[rim > 0] = True
    return con


def dilation_filter(mesh, condition):
    tri = np.asarray(mesh.triangles)
    num = np.asarray(mesh.vertices).shape[0]
    con = condition.copy()

    # get segment labels seg at vertex level
    msh_pos = deepcopy(mesh)
    msh_pos.remove_triangles_by_mask(con)
    tri_pos = np.asarray(msh_pos.triangles)
    tric, ntri, area = msh_pos.cluster_connected_triangles()
    tric = np.asarray(tric)
    seg = np.zeros(num, dtype="int") - 1
    seg[tri_pos] = np.c_[tric, tric, tric]
    seg += 1

    # add triangles touching the rim of not part of another label
    tsg = seg[tri]
    sel = np.where(~((tsg[:, 0] == tsg[:, 1]) * (tsg[:, 1] == tsg[:, 2])))[0]
    for i in tqdm(sel):
        js = tri[i]
        si = seg[js]
        si = si[si > 0]
        if (si == si[0]).all():
            con[i] = False
            sj = si[0]
            for j in js:
                seg[j] = sj

    return con


def fill_topological_holes_filter(mesh, condition):
    mesh_pos = deepcopy(mesh)
    mesh_neg = deepcopy(mesh)
    con = condition.copy()
    mesh_pos.remove_triangles_by_mask(con)
    mesh_neg.remove_triangles_by_mask(~con)
    tri_pos = np.asarray(mesh_pos.triangles)
    tri_neg = np.asarray(mesh_neg.triangles)
    seg = np.zeros((np.asarray(mesh.vertices).shape[0], 2), dtype="int") - 1

    # get positive comps
    tric, ntri, area = mesh_pos.cluster_connected_triangles()
    tric = np.asarray(tric)
    seg[tri_pos, 0] = np.c_[tric, tric, tric]

    # get negative comps
    tric, ntri, area = mesh_neg.cluster_connected_triangles()
    tric = np.asarray(tric)
    seg[tri_neg, 1] = np.c_[tric, tric, tric]
    contact = (seg[:, 0] != -1) * (seg[:, 1] != -1)
    idx = np.where(contact)[0]
    adj = np.zeros((seg[idx, 0].max() + 1, seg[idx, 1].max() + 1), dtype="int")

    # build unique connections
    for i in idx:
        adj[seg[i, 0], seg[i, 1]] = 1

    # fill topological holes
    (holes,) = np.where(np.sum(adj, axis=0) == 1)
    sub = con[con]
    for h in tqdm(holes):
        hh = np.where(tric == h)[0]
        sub[hh] = False

    con[con] = sub
    return con


def ntri_filter(mesh, condition, trithr):
    msh_pos = deepcopy(mesh)
    con = condition.copy()
    msh_pos.remove_triangles_by_mask(con)
    sub = con[~con]
    tric, ntri, area = msh_pos.cluster_connected_triangles()
    tric = np.asarray(tric)
    ntri = np.asarray(ntri)
    area = np.asarray(area)
    sub[ntri[tric] < trithr] = True
    con[~con] = sub
    return con


def show_mesh(mesh, condition):
    msh_pos = deepcopy(mesh)
    msh_pos.remove_triangles_by_mask(condition)
    o3d.visualization.draw_geometries([msh_pos])
    return


def sphericity_filter(mesh, condition, sphthr):
    pi13 = (np.pi) ** (1 / 3)
    msh_pos = deepcopy(mesh)
    con = condition.copy()
    msh_pos.remove_triangles_by_mask(con)
    sub = con[~con]
    tri_pos = np.asarray(msh_pos.triangles)
    vrt_pos = np.asarray(msh_pos.vertices)
    tric, ntri, area = msh_pos.cluster_connected_triangles()
    tric = np.asarray(tric)
    ntri = np.asarray(ntri)
    for t in trange(tric.max() + 1):
        tt = np.where(tric == t)[0]
        vt = np.unique(tri_pos[tt])
        vv = vrt_pos[vt]
        ch = ConvexHull(vv - vv.mean(axis=0))
        rt = pi13 * (6 * ch.volume) ** (2 / 3) / ch.area
        if rt < sphthr:
            sub[tt] = True
    con[~con] = sub
    return con


def triangle_mesh_divergence(tp, tn, vp, k=25):
    tr = kdtree(tp)
    dd, ii = tr.query(vp, k=1, workers=-1)
    dt = dd.mean()
    vw = dt * 2
    dd, ii = tr.query(vp + np.array([vw, 0, 0]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    nx = np.sum(wi * tn[ii, 0], axis=1) / np.sum(wi, axis=1)
    dd, ii = tr.query(vp - np.array([vw, 0, 0]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    nx -= np.sum(wi * tn[ii, 0], axis=1) / np.sum(wi, axis=1)
    dd, ii = tr.query(vp + np.array([0, vw, 0]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    ny = np.sum(wi * tn[ii, 1], axis=1) / np.sum(wi, axis=1)
    dd, ii = tr.query(vp - np.array([0, vw, 0]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    ny -= np.sum(wi * tn[ii, 1], axis=1) / np.sum(wi, axis=1)
    dd, ii = tr.query(vp + np.array([0, 0, vw]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    nz = np.sum(wi * tn[ii, 2], axis=1) / np.sum(wi, axis=1)
    dd, ii = tr.query(vp - np.array([0, 0, vw]), k=k, workers=-1)
    dd[dd < dt] = dt
    wi = 1.0 / dd
    nz -= np.sum(wi * tn[ii, 2], axis=1) / np.sum(wi, axis=1)
    div = (nx + ny + nz) / vw / 2.0
    return div


def instance_segmentation(
    mesh,
    surface_area_threshold=1e-4,
    sphericity=True,
    sphericity_threshold=0.5,
    fill_holes=True,
    erosion=False,
    erosion_iterations=1,
    dilation=False,
    dilation_iterations=1,
):
    mesh.compute_triangle_normals()
    ntr = np.asarray(mesh.triangle_normals)
    tri = np.asarray(mesh.triangles)
    vrt = np.asarray(mesh.vertices)
    ptr = np.c_[
        vrt[tri, 0].mean(axis=1), vrt[tri, 1].mean(axis=1), vrt[tri, 2].mean(axis=1)
    ]
    mesh.compute_vertex_normals()
    area = mesh.get_surface_area()
    print("total surface area:", area)
    satri = area / tri.shape[0]
    print("average surface area per triangle:", satri)

    # for simplicity we are thresholding at the level of triangles
    trithr = int(surface_area_threshold / satri)
    print("surface area threshold in number of triangles:", trithr)

    print("compute surface normal divergence as mean surface curvature")
    div = triangle_mesh_divergence(ptr, ntr, vrt)

    # initial segmentation condition at the level of vertices
    conv = np.min(div[tri], axis=1) < 0

    # remove large vertical triangles
    print("removing large vertical triangles")
    ztri = vrt[tri, 2]
    conv[ztri.max(axis=1) - ztri.min(axis=1) > 5] = True

    # fill holes that are smaller than our surface area threshold
    conv = ntri_filter(mesh, ~conv, trithr)
    conv = ~conv

    if erosion:
        print("erode potential segments in order to remove bridges")
        for i in range(erosion_iterations):
            conv = erosion_filter(mesh, conv)

    # remove segments that are smaller than our threshold
    conv = ntri_filter(mesh, conv, trithr)

    if sphericity:
        print("filter by sphericity")
        conv = sphericity_filter(mesh, conv, sphericity_threshold)

    if dilation:
        print("dilation, undo erosion of segments")
        for i in range(dilation_iterations):
            conv = dilation_filter(mesh, conv)
            # remove large vertical triangles
            conv[ztri.max(axis=1) - ztri.min(axis=1) > 5] = True

    if fill_holes:
        print("fill topological holes in segments")
        conv = fill_topological_holes_filter(mesh, conv)

    print("final components as segments")
    mesh_pos = deepcopy(mesh)
    mesh_pos.remove_triangles_by_mask(conv)
    tri_pos = np.asarray(mesh_pos.triangles)
    tric, ntri, area = mesh_pos.cluster_connected_triangles()
    tric = np.asarray(tric)
    ntri = np.asarray(ntri)
    tric[ntri[tric] < trithr] = -1
    vla = np.zeros(vrt.shape[0], dtype="int") - 1
    vla[tri_pos] = np.c_[tric, tric, tric]
    tla = np.zeros(conv.shape, dtype="int") - 1
    tla[~conv] = tric

    print("neighborhood graph based segment coloring")
    cmap = graph_coloring(mesh, conv, vla)
    return (mesh_pos, tla, vla, cmap[vla])


def segment_statistics(mesh, seg, colors, write_segments=True):
    vrt = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    pi13 = (np.pi) ** (1 / 3)
    useg = np.unique(seg[0 <= seg])
    csv = {}
    keys = (
        "segment id",
        "segment color",
        "centroid x",
        "centroid y",
        "centroid z",
        "A axis",
        "B axis",
        "C axis",
        "yaw",
        "pitch",
        "roll",
        "red mean",
        "green mean",
        "blue mean",
        "red std",
        "green std",
        "blue std",
        "mesh surface area",
        "convex hull area",
        "convex hull volume",
        "convex hull sphericity",
    )
    for key in keys:
        csv[key] = []

    for i in tqdm(useg):
        sl = np.where(seg == i)[0]
        ivrt = vrt[sl]
        irgb = rgb[sl]

        csv["segment id"].append(i)

        # segment spatial graph coloring
        csv["segment color"].append(colors[sl[0]])

        # centroid of ith segment
        centroid = ivrt.mean(axis=0)
        csv["centroid x"].append(centroid[0])
        csv["centroid y"].append(centroid[1])
        csv["centroid z"].append(centroid[2])
        ivrt -= centroid

        # local segment coordinates
        _, _, R = np.linalg.svd(np.cov(ivrt.T))
        rv = R.dot(ivrt.T).T

        # determine longest axes A, B and C (bbox)
        vmax = rv.max(axis=0)
        vmin = rv.min(axis=0)
        bbox = vmax - vmin
        csv["A axis"].append(bbox[0])
        csv["B axis"].append(bbox[1])
        csv["C axis"].append(bbox[2])

        # Tait-Bryan rotations
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        csv["yaw"].append(yaw * 180 / np.pi)
        csv["pitch"].append(pitch * 180 / np.pi)
        csv["roll"].append(roll * 180 / np.pi)

        # mean and std rgb values
        rgb_mean = irgb.mean(axis=0)
        rgb_std = irgb.std(axis=0)
        csv["red mean"].append(rgb_mean[0])
        csv["green mean"].append(rgb_mean[1])
        csv["blue mean"].append(rgb_mean[2])
        csv["red std"].append(rgb_std[0])
        csv["green std"].append(rgb_std[1])
        csv["blue std"].append(rgb_std[2])

        # surface area
        mesh_seg = deepcopy(mesh)
        mesh_seg.remove_vertices_by_mask(seg != i)
        csv["mesh surface area"].append(mesh_seg.get_surface_area())
        if write_segments:
            o3d.io.write_triangle_mesh("segment_%04d.ply" % i, mesh_seg)

        # convex hull surface area, volume and sphericity
        ch = ConvexHull(ivrt)
        csv["convex hull area"].append(ch.area)
        csv["convex hull volume"].append(ch.volume)
        csv["convex hull sphericity"].append(
            pi13 * (6 * ch.volume) ** (2 / 3) / ch.area
        )

    return csv


def dict_to_csv(fn, csv_dict):
    import pandas as pd

    df = pd.DataFrame(data=csv_dict)
    df.to_csv(fn)