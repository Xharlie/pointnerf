import os, torch, cv2, re
import numpy as np

from torch_scatter import scatter_min, segment_coo, scatter_mean
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from functools import partial
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
mse2psnr2 = lambda x : -10. * np.log(x) / np.log(10.)

def get_psnr(imgs_pred, imgs_gt):
    psnrs = []
    for (img,tar) in zip(imgs_pred,imgs_gt):
        psnrs.append(mse2psnr2(np.mean((img - tar.cpu().numpy())**2)))
    return np.array(psnrs)

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]


# Ray helpers
def get_rays_mvs(H, W, intrinsic, c2w, N=1024, isRandom=True, is_precrop_iters=False, chunk=-1, idx=-1):

    device = c2w.device
    if isRandom:
        if is_precrop_iters and torch.rand((1,)) > 0.3:
            xs, ys = torch.randint(W//6, W-W//6, (N,)).float().to(device), torch.randint(H//6, H-H//6, (N,)).float().to(device)
        else:
            xs, ys = torch.randint(0,W,(N,)).float().to(device), torch.randint(0,H,(N,)).float().to(device)
    else:
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        if chunk>0:
            ys, xs = ys[idx*chunk:(idx+1)*chunk], xs[idx*chunk:(idx+1)*chunk]
        ys, xs = ys.to(device), xs.to(device)

    dirs = torch.stack([(xs-intrinsic[0,2])/intrinsic[0,0], (ys-intrinsic[1,2])/intrinsic[1,1], torch.ones_like(xs)], -1) # use 1 instead of -1


    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].clone()
    pixel_coordinates = torch.stack((ys,xs)) # row col
    return rays_o, rays_d, pixel_coordinates

def ndc_2_cam(ndc_xyz, near_far, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3] * (near_far[1] - near_far[0]) + near_far[0]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
    '''
        point_samples [N_rays N_sample 3]
    '''

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    # wrap to ref view
    if w2c_ref is not None:
        R = w2c_ref[:3, :3]  # (3, 3)
        T = w2c_ref[:3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)

    if intrinsic_ref is not None:
        # using projection
        point_samples_pixel =  point_samples @ intrinsic_ref.t()
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(1,3), far.view(1,3)
        point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
    del point_samples

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
        point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

    point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
    return point_samples_pixel

def build_color_volume(point_samples, pose_ref, imgs, img_feat=None, downscale=1.0, with_mask=False):
    '''
    point_world: [N_ray N_sample 3]
    imgs: [N V 3 H W]
    '''

    device = imgs.device
    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)

    C += with_mask
    C += 0 if img_feat is None else img_feat.shape[2]
    colors = torch.empty((*point_samples.shape[:2], V*C), device=imgs.device, dtype=torch.float)
    for i,idx in enumerate(range(V)):

        w2c_ref, intrinsic_ref = pose_ref['w2cs'][idx], pose_ref['intrinsics'][idx].clone()  # assume camera 0 is reference
        point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale)[None]
        grid = point_samples_pixel[...,:2]*2.0-1.0

        grid = grid.to(imgs.dtype)

        data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')
        if img_feat is not None:
            data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)

        if with_mask:
            in_mask = ((grid >-1.0)*(grid < 1.0))
            in_mask = (in_mask[...,0]*in_mask[...,1]).float()
            data = torch.cat((data,in_mask.unsqueeze(1)), dim=1)

        colors[...,i*C:i*C+C] = data[0].permute(1, 2, 0)
        del grid, point_samples_pixel, data

    return colors


def normal_vect(vect, dim=-1):
    return vect / (torch.sqrt(torch.sum(vect**2,dim=dim,keepdim=True))+1e-7)

def index_point_feature(volume_feature, ray_coordinate_ref, chunk=-1):
        ''''
        Args:
            volume_color_feature: [B, G, D, h, w]
            volume_density_feature: [B C D H W]
            ray_dir_world:[3 ray_samples N_samples]
            ray_coordinate_ref:  [3 N_rays N_samples]
            ray_dir_ref:  [3 N_rays]
            depth_candidates: [N_rays, N_samples]
        Returns:
            [N_rays, N_samples]
        '''

        device = volume_feature.device
        H, W = ray_coordinate_ref.shape[-3:-1]


        if chunk != -1:
            features = torch.zeros((volume_feature.shape[1],H,W), device=volume_feature.device, dtype=torch.float, requires_grad=volume_feature.requires_grad)
            grid = ray_coordinate_ref.view(1, 1, 1, H * W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            for i in range(0, H*W, chunk):
                features[:,i:i + chunk] = F.grid_sample(volume_feature, grid[:,:,:,i:i + chunk], align_corners=True, mode='bilinear')[0]
            features = features.permute(1,2,0)
        else:
            grid = ray_coordinate_ref.view(-1, 1, H,  W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')[:,:,0].permute(2,3,0,1).squeeze()#, padding_mode="border"
        return features


def filter_keys(dict):
    if 'N_samples' in dict.keys():
        dict.pop('N_samples')
    if 'ndc' in dict.keys():
        dict.pop('ndc')
    if 'lindisp' in dict.keys():
        dict.pop('lindisp')
    return dict

def sub_selete_data(data_batch, device, idx, filtKey=[], filtIndex=['view_ids_all','c2ws_all','scan','bbox','w2ref','ref2w','light_id','ckpt','idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:,idx].float() if (item not in filtIndex and torch.is_tensor(item) and item.dim()>2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete

def detach_data(dictionary):
    dictionary_new = {}
    for key in dictionary.keys():
        dictionary_new[key] = dictionary[key].detach().clone()
    return dictionary_new

def read_pfm(filename):
    file = open(filename, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


from scipy.interpolate import CubicSpline

#################################################  MVS  helper functions   #####################################
from kornia.utils import create_meshgrid


def homo_warp_nongrid(c2w, w2c, intrinsic, ref_cam_xyz, HD, WD, filter=True, **kwargs):
    # src_grid: B, 3, D*H*W   xyz
    B, M, _ = ref_cam_xyz.shape
    if w2c is not None:
        src_cam_xyz = torch.cat([ref_cam_xyz, torch.ones_like(ref_cam_xyz[:,:,0:1])], dim=-1) @ c2w.transpose(1,2) @ w2c.transpose(1,2)
    else:
        src_cam_xyz = ref_cam_xyz
    src_grid = ((src_cam_xyz[..., :3] / src_cam_xyz[..., 2:3]) @ intrinsic.transpose(1,2))[...,:2]

    mask = torch.prod(torch.cat([torch.ge(src_grid, torch.zeros([1,1,2], device=src_grid.device)), torch.le(src_grid, torch.tensor([[[WD-1,HD-1]]], device=src_grid.device))],dim=-1), dim=-1, keepdim=True, dtype=torch.int8) > 0

    src_grid = src_grid.to(torch.float32)  # grid xy
    hard_id_xy = torch.ceil(src_grid[:,:,:])
    src_grid = torch.masked_select(src_grid, mask).reshape(B, -1, 2) if filter else src_grid
    src_grid[..., 0] = src_grid[..., 0] / ((WD - 1.0) / 2.0) - 1.0  # scale to -1~1
    src_grid[..., 1] = src_grid[..., 1] / ((HD - 1.0) / 2.0) - 1.0  # scale to -1~1
    return src_grid, mask, hard_id_xy


def homo_warp_fg_mask(c2w, w2c, intrinsic, ref_cam_xyz, HD, WD, **kwargs):
    # src_grid: B, 3, D*H*W   xyz
    B, M, _ = ref_cam_xyz.shape
    if w2c is not None:
        src_cam_xyz = torch.cat([ref_cam_xyz, torch.ones_like(ref_cam_xyz[:,:,0:1])], dim=-1) @ c2w.transpose(1,2) @ w2c.transpose(1,2)
    else:
        src_cam_xyz = ref_cam_xyz
    src_grid = ((src_cam_xyz[..., :3] / src_cam_xyz[..., 2:3]) @ intrinsic.transpose(1,2))[...,:2]

    mask = torch.prod(torch.cat([torch.ge(src_grid, torch.zeros([1,1,2], device=src_grid.device)), torch.le(src_grid, torch.tensor([[[WD-1,HD-1]]], device=src_grid.device))],dim=-1), dim=-1, keepdim=True, dtype=torch.int8) > 0

    src_grid = src_grid.to(torch.float32)  # grid xy
    hard_id_xy = torch.ceil(src_grid[:,:,:])[:,mask[0,...,0],:]
    return id2mask(hard_id_xy, HD, WD)

def homo_warp_nongrid_occ(c2w, w2c, intrinsic, ref_cam_xyz, HD, WD, tolerate=0.1, scatter_cpu=True):
    # src_grid: B, 3, D*H*W   xyz
    B, M, _ = ref_cam_xyz.shape
    if w2c is not None:
        src_cam_xyz = torch.cat([ref_cam_xyz, torch.ones_like(ref_cam_xyz[:,:,0:1])], dim=-1) @ c2w.transpose(1,2) @ w2c.transpose(1,2)
    else:
        src_cam_xyz = ref_cam_xyz
    # print("src_cam_xyz",src_cam_xyz.shape, intrinsic.shape)
    src_grid = ((src_cam_xyz[..., :3] / src_cam_xyz[..., 2:3]) @ intrinsic.transpose(1,2))[...,:2]
    # print("src_pix_xy1", src_grid.shape, torch.min(src_grid,dim=-2)[0], torch.max(src_grid,dim=-2)[0])
    mask = torch.prod(torch.cat([torch.ge(src_grid, torch.zeros([1,1,2], device=src_grid.device)), torch.le(torch.ceil(src_grid), torch.tensor([[[WD-1,HD-1]]], device=src_grid.device))],dim=-1), dim=-1, keepdim=True, dtype=torch.int8) > 0
    src_grid = torch.masked_select(src_grid, mask).reshape(B, -1, 2)
    cam_z = torch.masked_select(src_cam_xyz[:,:,2], mask[...,0]).reshape(B, -1)

    src_grid = src_grid.to(torch.float32)  # grid xy
    # print("HD, WD", HD, WD) 512 640
    src_grid_x = src_grid[..., 0:1] / ((WD - 1.0) / 2.0) - 1.0  # scale to -1~1
    src_grid_y = src_grid[..., 1:2] / ((HD - 1.0) / 2.0) - 1.0  # scale to -1~1
    # hard_id_xy: 1, 307405, 2

    hard_id_xy = torch.ceil(src_grid[:,:,:])
    # print("hard_id_xy", hard_id_xy.shape)
    index = (hard_id_xy[...,0] * HD + hard_id_xy[...,1]).long() # 1, 307405
    # print("index", index.shape, torch.min(index), torch.max(index))
    min_depth, argmin = scatter_min(cam_z[:,:].cpu() if scatter_cpu else cam_z[:,:], index[:,:].cpu() if scatter_cpu else index[:,:], dim=1)
    # print("argmin", min_depth.shape, min_depth, argmin.shape)

    queried_depth = min_depth.to(ref_cam_xyz.device)[:, index[0,...]] if scatter_cpu else min_depth[:, index[0,...]]
    block_mask = (cam_z <= (queried_depth + tolerate))
    # print("mask", mask.shape, torch.sum(mask), block_mask.shape, torch.sum(block_mask))
    mask[mask.clone()] = block_mask
    # print("mask", mask.shape, torch.sum(mask), block_mask.shape, torch.sum(block_mask))
    # print("src_grid_x", src_grid_x.shape)
    src_grid_x = torch.masked_select(src_grid_x, block_mask[..., None]).reshape(B, -1, 1)
    src_grid_y = torch.masked_select(src_grid_y, block_mask[..., None]).reshape(B, -1, 1)
    # print("src_grid_x", src_grid_x.shape, src_grid_y.shape, mask.shape)
    return torch.cat([src_grid_x, src_grid_y], dim=-1), mask, hard_id_xy


def id2mask(hard_id_xy, HD, WD):
    mask = torch.zeros([HD, WD], dtype=torch.int8, device=hard_id_xy.device)
    hard_id_xy = hard_id_xy.long()
    mask[hard_id_xy[0,...,1], hard_id_xy[0,...,0]] = 1 # torch.ones_like(hard_id_xy[0,...,0], dtype=mask.dtype)
    return mask



def gen_bg_points(batch):
    plane_pnt, plane_normal = batch["plane_pnt"][0], batch["plane_normal"][0]
    plane_pnt, plane_normal = torch.as_tensor(plane_pnt, dtype=torch.float32, device=batch['campos'].device), torch.as_tensor(plane_normal, dtype=torch.float32, device=batch['campos'].device)
    cross_xyz_world = get_rayplane_cross(batch['campos'], batch['raydir'], plane_pnt[None, None, :], plane_normal[None, None, :])

    return cross_xyz_world

def get_rayplane_cross(cam_pos, raydir, p_co, p_no, epsilon=1e-3):
    """
    cam_pos: 1, 3
    ray_dir: Define the line.  1, 2304, 3
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).    1, 1, 3
        p_no Is a normal vector defining the plane direction;   1, 1, 3
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """
    dot = torch.sum(p_no * raydir, dim=-1) # 1, 2304
    board_mask = dot >= epsilon
    dot_valid = dot[board_mask][None,:]     # 1, 2304
    w = cam_pos[None,:,:] - p_co # torch.Size([1, 1, 3])
    fac = -torch.sum(p_no * w, dim=-1) / dot_valid # 1, 2304
    ray_dir_valid = raydir[:,board_mask[0],:]
    ray_dir_valid = ray_dir_valid * fac[..., None] # 1, 2304, 3
    intersect_world_valid = cam_pos[None,...] + ray_dir_valid # 1, 2304, 3
    intersect_world = torch.zeros_like(raydir)
    intersect_world[:,board_mask[0],:] = intersect_world_valid
    return intersect_world


def extract_from_2d_grid(src_feat, src_grid, mask):
    B, M, _ = src_grid.shape
    warped_src_feat = F.grid_sample(src_feat, src_grid[:, None, ...], mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.permute(0,2,3,1).view(B, M, src_feat.shape[1]).cuda() # 1, 224874, 3
    if mask is not None:
        B, N, _ = mask.shape
        full_src_feat = torch.zeros([B, N, src_feat.shape[1]], device=warped_src_feat.device, dtype=warped_src_feat.dtype)
        full_src_feat[0, mask[0,:,0], :] = warped_src_feat
        warped_src_feat = full_src_feat
    return warped_src_feat


def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad
        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W), X, Y, Z
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)

        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)

        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, H_pad, W_pad, 2)

    B, D, H_pad, W_pad = src_grid.shape[:4]
    src_grid = src_grid.to(src_feat.dtype) # 1, 32, 128, 160
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, H_pad * W_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    return warped_src_feat, src_grid


###############################  render path  ####################################

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from warmup_scheduler import GradualWarmupScheduler


def construct_vox_points(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
        space_max = xyz_mid + space_edge / 2
    else:
        space_edge = space_max - space_min
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = scatter_mean(xyz_val, inv_idx, dim=0)
    min_idx, _ = scatter_min(torch.arange(len(xyz), device=xyz.device), inv_idx, dim=0)
    return xyz_centroid, sparse_grid_idx, min_idx


def construct_vox_points_xyz(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = scatter_mean(xyz_val, inv_idx, dim=0)
    return xyz_centroid


def construct_vox_points_ind(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
        space_max = xyz_mid + space_edge / 2
    else:
        space_edge = space_max - space_min
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    return sparse_grid_idx, inv_idx, space_min, space_max


def construct_vox_points_closest(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
        mask = (xyz_val - space_min[None,...])
        mask *= (space_max[None,...] - xyz_val)
        mask = torch.prod(mask, dim=-1) > 0
        xyz_val = xyz_val[mask, :]
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = scatter_mean(xyz_val, inv_idx, dim=0)
    xyz_centroid_prop = xyz_centroid[inv_idx,:]
    xyz_residual = torch.norm(xyz_val - xyz_centroid_prop, dim=-1)
    print("xyz_residual", xyz_residual.shape)

    _, min_idx = scatter_min(xyz_residual, inv_idx, dim=0)
    print("min_idx", min_idx.shape)
    return xyz_centroid, sparse_grid_idx, min_idx


def transform_points_to_voxels(points, point_cloud_range, voxel_sizes, max_pnts_per_vox, max_voxels, voxel_generator=None):
    voxel_output = voxel_generator.generate(points)
    if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
    else:
        voxels, coordinates, num_points = voxel_output
    return voxels, coordinates, num_points

def alpha_masking(points, alphas, intrinsics, c2ws, w2cs, near_far, opt=None):
    w_xyz1 = torch.cat([points[..., :3], torch.ones_like(points[..., :1])], dim=-1)
    H, W = alphas[0][0].shape
    vishull_mask = None
    range_mask = None
    for i in range(len(alphas)):
        alpha, intrinsic, c2w, w2c = torch.as_tensor(alphas[i][0], dtype=points.dtype, device=points.device), torch.as_tensor(intrinsics[i], dtype=points.dtype, device=points.device), torch.as_tensor(c2ws[i], dtype=points.dtype, device=points.device), torch.as_tensor(w2cs[i], dtype=points.dtype, device=points.device)
        # print("w_xyz1",w_xyz1.shape, w2c.shape, intrinsic.shape, alpha.shape)
        cam_xyz = w_xyz1 @ w2c.t()
        torch.cuda.empty_cache()
        if near_far is not None:
            near_far_mask = torch.logical_and(cam_xyz[...,2]>=(near_far[0]-1.0), cam_xyz[...,2]<=near_far[1])
        cam_xyz = cam_xyz[...,:3] @ intrinsic.t()
        img_xy = torch.floor(cam_xyz[:, :2] / cam_xyz[:, -1:] + 0.0).long()
        del cam_xyz
        torch.cuda.empty_cache()
        if opt is not None and (opt.alpha_range > 0 or opt.inall_img == 0):
            range_mask = torch.logical_and(img_xy >= torch.zeros((1,2), dtype=img_xy.dtype, device=img_xy.device), img_xy < torch.as_tensor([[W,H]], dtype=img_xy.dtype, device=img_xy.device))
            range_mask = torch.prod(range_mask, dim=-1) > 0

        img_xy[..., 0] = torch.clamp(img_xy[..., 0], min=0, max=W-1)
        img_xy[..., 1] = torch.clamp(img_xy[..., 1], min=0, max=H-1)
        mask = alpha[img_xy[..., 1], img_xy[..., 0]]
        if range_mask is not None:
            mask = mask + (~range_mask).to(torch.float32)
        mask = mask > 0.1
        if near_far is not None:
            vishull_mask = (mask*near_far_mask) if vishull_mask is None else vishull_mask*(mask*near_far_mask)
        else:
            vishull_mask=mask if vishull_mask is None else vishull_mask*mask
        del img_xy
        torch.cuda.empty_cache()
    del range_mask
    print("vishull_mask", vishull_mask.shape)
    return vishull_mask > 0