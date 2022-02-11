import numpy as np
import open3d as o3d
def get_cv_raydir(pixelcoords, height, width, focal, rot):
    # pixelcoords: H x W x 2
    if isinstance(focal, float):
        focal = [focal, focal]
    x = (pixelcoords[..., 0] - width / 2.0) / focal[0]
    y = (pixelcoords[..., 1] - height / 2.0) / focal[1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs = np.sum(rot[None,None,:,:] * dirs[...,None], axis=-2) # 1*1*3*3   x   h*w*3*1
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

    return dirs


def get_camera_rotation(eye, center, up):
    nz = center - eye
    nz /= np.linalg.norm(nz)
    x = np.cross(nz, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, nz)
    return np.array([x, y, -nz]).T

#
# def get_blender_raydir(pixelcoords, height, width, focal, rot, dir_norm):
#     ## pixelcoords: H x W x 2
#     x = (pixelcoords[..., 0] - width / 2.0) / focal
#     y = (pixelcoords[..., 1] - height / 2.0) / focal
#     z = np.ones_like(x)
#     dirs = np.stack([x, -y, -z], axis=-1)
#     dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # 32, 32, 3
#     if dir_norm:
#         # print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
#         dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
#     # print("dirs", dirs.shape)
#
#     return dirs


def get_blender_raydir(pixelcoords, height, width, focal, rot, dir_norm):
    ## pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - width / 2.0) / focal
    y = (pixelcoords[..., 1] + 0.5 - height / 2.0) / focal
    z = np.ones_like(x)
    dirs = np.stack([x, -y, -z], axis=-1)
    dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
    if dir_norm:
        # print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
    # print("dirs", dirs.shape)

    return dirs

def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
    # rot is c2w
    ## pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    # dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
    dirs = dirs @ rot[:,:].T #
    if dir_norm:
        # print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
    # print("dirs", dirs.shape)

    return dirs


def get_optix_raydir(pixelcoords, height, width, focal, eye, center, up):
    c2w = get_camera_rotation(eye, center, up)
    return get_blender_raydir(pixelcoords, height, width, focal, c2w)


def flip_z(poses):
    z_flip_matrix = np.eye(4, dtype=np.float32)
    z_flip_matrix[2, 2] = -1.0
    return np.matmul(poses, z_flip_matrix[None,...])


def triangluation_bpa(pnts, test_pnts=None, full_comb=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pnts[:, :3] / np.linalg.norm(pnts[:, :3], axis=-1, keepdims=True))

    # pcd.colors = o3d.utility.Vector3dVector(pnts[:, 3:6] / 255)
    # pcd.normals = o3d.utility.Vector3dVector(pnts[:, 6:9])
    # o3d.visualization.draw_geometries([pcd])

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)


    radius = 3 * avg_dist
    dec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    # dec_mesh = dec_mesh.simplify_quadric_decimation(100000)
    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()

    # vis_lst = [dec_mesh, pcd]
    # vis_lst = [dec_mesh, pcd]
    # o3d.visualization.draw_geometries(vis_lst)
    # if test_pnts is not None :
    #     tpcd = o3d.geometry.PointCloud()
    #     print("test_pnts",test_pnts.shape)
    #     tpcd.points = o3d.utility.Vector3dVector(test_pnts[:, :3])
    #     tpcd.normals = o3d.utility.Vector3dVector(test_pnts[:, :3] / np.linalg.norm(test_pnts[:, :3], axis=-1, keepdims=True))
    #     o3d.visualization.draw_geometries([dec_mesh, tpcd] )
    triangles = np.asarray(dec_mesh.triangles, dtype=np.int32)
    if full_comb:
        q, w, e = triangles[..., 0], triangles[..., 1], triangles[..., 2]
        triangles2 = np.stack([w,q,e], axis=-1)
        triangles3 = np.stack([e,q,w], axis=-1)
        triangles = np.concatenate([triangles, triangles2, triangles3], axis=0)
    return triangles

