import os
import numpy as np
from numpy import dot
from math import sqrt
import pycuda
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import torch
import pickle
import time
# import cupy
# import open3d.ml.tf as ml3d
# import frnn

from data.load_blender import load_blender_data

# X = torch.cuda.FloatTensor(8)


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()

class lighting_fast_querier():

    def __init__(self, device, opt):

        print("querier device", device, device.index)
        self.gpu = device.index
        self.opt = opt
        drv.init()
        # self.device = drv.Device(gpu)
        self.ctx = drv.Device(self.gpu).make_context()
        self.get_occ_vox, self.near_vox_full, self.insert_vox_points, self.query_along_ray = self.build_cuda()
        
        self.inverse = self.opt.inverse

    def clean_up(self):
        self.ctx.pop()


    def get_hyperparameters(self, h, w, intrinsic, near_depth, far_depth):
        # print("h,w,focal,near,far", h.shape, w.shape, focal.shape, near_depth.shape, far_depth.shape)
        # x_r = w / 2 / focal
        # y_r = h / 2 / focal
        # ranges = np.array([-x_r, -y_r, near_depth, x_r, y_r, far_depth], dtype=np.float32)
        # vdim = np.array([h, w, self.opt.z_depth_dim], dtype=np.int32)
        # vsize = np.array([2 * x_r / vdim[0], 2 * y_r / vdim[1], z_r / vdim[2]], dtype=np.float32)
        x_rl, x_rh = -intrinsic[0, 2] / intrinsic[0, 0], (w - intrinsic[0, 2]) / intrinsic[0, 0]
        y_rl, y_rh = -intrinsic[1, 2] / intrinsic[1, 1], (h - intrinsic[1, 2]) / intrinsic[1, 1],
        z_r = (far_depth - near_depth) if self.inverse == 0 else (1.0 / near_depth - 1.0 / far_depth)
        #  [-0.22929783 -0.1841962   2.125       0.21325193  0.17096843  4.525     ]
        ranges = np.array([x_rl, y_rl, near_depth, x_rh, y_rh, far_depth], dtype=np.float32) if self.inverse == 0 else np.array([x_rl, y_rl, 1.0 / far_depth, x_rh, y_rh, 1.0 / near_depth], dtype=np.float32)
        vdim = np.array([w, h, self.opt.z_depth_dim], dtype=np.int32)

        vsize = np.array([(x_rh - x_rl) / vdim[0], (y_rh - y_rl) / vdim[1], z_r / vdim[2]], dtype=np.float32)

        vscale = np.array(self.opt.vscale, dtype=np.int32)
        scaled_vdim = np.ceil(vdim / vscale).astype(np.int32)
        scaled_vsize = (vsize * vscale).astype(np.float32)
        range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = np_to_gpuarray(ranges, scaled_vsize, scaled_vdim, vscale, np.asarray(self.opt.kernel_size, dtype=np.int32),  np.asarray(self.opt.query_size, dtype=np.int32))
        radius_limit, depth_limit = self.opt.radius_limit_scale * max(vsize[0], vsize[1]), self.opt.depth_limit_scale * vsize[2]


        return radius_limit.astype(np.float32), depth_limit.astype(np.float32), ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu


    def query_points(self, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, h, w, intrinsic, near_depth, far_depth, ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor):

        # print("attr", hasattr(self, "h"), self.opt.feedforward)
        #
        # if not hasattr(self, "h") or self.opt.feedforward > 0 or self.vscale != self.opt.vscale or self.kernel_size != self.opt.kernel_size:
        #     radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.get_hyperparameters(h, w, intrinsic, near_depth, far_depth)
        #     if self.opt.feedforward==0:
        #         self.radius_limit, self.depth_limit, self.ranges, self.vsize, self.vdim, self.scaled_vsize, self.scaled_vdim, self.vscale, self.range_gpu, self.scaled_vsize_gpu, self.scaled_vdim_gpu, self.vscale_gpu, self.kernel_size_gpu, self.kernel_size, self.query_size_gpu = radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, self.opt.kernel_size, query_size_gpu
        #
        # else:
        #     radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.radius_limit, self.depth_limit, self.ranges, self.vsize, self.vdim, self.scaled_vsize, self.scaled_vdim, self.vscale, self.range_gpu, self.scaled_vsize_gpu, self.scaled_vdim_gpu, self.vscale_gpu, self.kernel_size_gpu, self.query_size_gpu

        radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.get_hyperparameters(h, w, intrinsic, near_depth, far_depth)
        self.radius_limit, self.depth_limit, self.ranges, self.vsize, self.vdim, self.scaled_vsize, self.scaled_vdim, self.vscale, self.range_gpu, self.scaled_vsize_gpu, self.scaled_vdim_gpu, self.vscale_gpu, self.c, self.query_size_gpu = radius_limit, depth_limit, ranges, vsize, vdim, scaled_vsize, scaled_vdim, vscale, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu

        sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor = self.query_grid_point_index(h, w,pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, self.opt.SR, self.opt.K, ranges, scaled_vsize, scaled_vdim, vscale, self.opt.max_o, self.opt.P, radius_limit, depth_limit, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kMaxThreadsPerBlock=self.opt.gpu_maxthr)

        self.inverse = self.opt.inverse

        if self.opt.is_train:
            sample_loc_tensor = getattr(self, self.opt.shpnt_jitter, None)(sample_loc_tensor, vsize)

        sample_loc_w_tensor, sample_ray_dirs_tensor = self.pers2w(sample_loc_tensor, cam_rot_tensor, cam_pos_tensor)

        return sample_pidx_tensor, sample_loc_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, ray_mask_tensor, vsize, ranges

    def pers2w(self, point_xyz_pers, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3

        x_pers = point_xyz_pers[..., 0] * point_xyz_pers[..., 2]
        y_pers = point_xyz_pers[..., 1] * point_xyz_pers[..., 2]
        z_pers = point_xyz_pers[..., 2]
        xyz_c = torch.stack([x_pers, y_pers, z_pers], dim=-1)
        xyz_w_shift = torch.sum(xyz_c[...,None,:] * camrotc2w, dim=-1)
        # print("point_xyz_pers[..., 0, 0]", point_xyz_pers[..., 0, 0].shape, point_xyz_pers[..., 0, 0])
        ray_dirs = xyz_w_shift / (torch.linalg.norm(xyz_w_shift, dim=-1, keepdims=True) + 1e-7)

        xyz_w = xyz_w_shift + campos[:, None, :]
        return xyz_w, ray_dirs

    def gaussian(self, input, vsize):
        B, R, SR, _ = input.shape
        jitters = torch.normal(mean=torch.zeros([B, R, SR], dtype=torch.float32, device=input.device), std=torch.full([B, R, SR], vsize[2] / 4, dtype=torch.float32, device=input.device))
        input[..., 2] = input[..., 2] + torch.clamp(jitters, min=-vsize[2]/2, max=vsize[2]/2)
        return input

    def uniform(self, input, vsize):
        B, R, SR, _ = input.shape
        jitters = torch.rand([B, R, SR], dtype=torch.float32, device=input.device) - 0.5
        input[..., 2] = input[..., 2] + jitters * vsize[2]
        return input


    def build_cuda(self):

        mod = SourceModule(
            """
            #define KN  """ + str(self.opt.K)
            + """ 
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <algorithm>
            #include <vector>
            #include <stdio.h>
            #include <math.h>
            #include <stdlib.h>
            #include <curand_kernel.h>
            namespace cuda {          
    
                static __device__ inline uint8_t atomicAdd(uint8_t *address, uint8_t val) {
                    size_t offset = (size_t)address & 3;
                    uint32_t *address_as_ui = (uint32_t *)(address - offset);
                    uint32_t old = *address_as_ui;
                    uint32_t shift = offset * 8;
                    uint32_t old_byte;
                    uint32_t newval;
                    uint32_t assumed;
    
                    do {
                      assumed = old;
                      old_byte = (old >> shift) & 0xff;
                      // preserve size in initial cast. Casting directly to uint32_t pads
                      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
                      newval = static_cast<uint8_t>(val + old_byte);
                      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
                      old = atomicCAS(address_as_ui, assumed, newval);
                    } while (assumed != old);
                    return __byte_perm(old, 0, offset);   // need validate
                }
    
                static __device__ inline char atomicAdd(char* address, char val) {
                    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
                    size_t long_address_modulo = (size_t) address & 3;
                    // the 32-bit address that overlaps the same memory
                    auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
                    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
                    // The "4" signifies the position where the first byte of the second argument will end up in the output.
                    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
                    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
                    unsigned int selector = selectors[long_address_modulo];
                    unsigned int long_old, long_assumed, long_val, replacement;
    
                    long_old = *base_address;
    
                    do {
                        long_assumed = long_old;
                        // replace bits in long_old that pertain to the char address with those from val
                        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
                        replacement = __byte_perm(long_old, long_val, selector);
                        long_old = atomicCAS(base_address, long_assumed, replacement);
                    } while (long_old != long_assumed);
                    return __byte_perm(long_old, 0, long_address_modulo);
                }            
    
                static __device__ inline int8_t atomicAdd(int8_t *address, int8_t val) {
                    return (int8_t)cuda::atomicAdd((char*)address, (char)val);
                }
    
                static __device__ inline short atomicAdd(short* address, short val)
                {
    
                    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);
    
                    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
    
                    unsigned int long_old = ::atomicAdd(base_address, long_val);
    
                    if((size_t)address & 2) {
                        return (short)(long_old >> 16);
                    } else {
    
                        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
    
                        if (overflow)
    
                            atomicSub(base_address, overflow);
    
                        return (short)(long_old & 0xffff);
                    }
                }
    
                static __device__ float cas(double *addr, double compare, double val) {
                   unsigned long long int *address_as_ull = (unsigned long long int *) addr;
                   return __longlong_as_double(atomicCAS(address_as_ull,
                                                 __double_as_longlong(compare),
                                                 __double_as_longlong(val)));
                }
    
                static __device__ float cas(float *addr, float compare, float val) {
                    unsigned int *address_as_uint = (unsigned int *) addr;
                    return __uint_as_float(atomicCAS(address_as_uint,
                                           __float_as_uint(compare),
                                           __float_as_uint(val)));
                }
    
    
    
                static __device__ inline uint8_t atomicCAS(uint8_t * const address, uint8_t const compare, uint8_t const value)
                {
                    uint8_t const longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
                    uint32_t *const baseAddress  = reinterpret_cast< uint32_t * >( address - longAddressModulo );
                    uint32_t constexpr byteSelection[] = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
                    uint32_t const byteSelector = byteSelection[ longAddressModulo ];
                    uint32_t const longCompare = compare;
                    uint32_t const longValue = value;
                    uint32_t longOldValue = * baseAddress;
                    uint32_t longAssumed;
                    uint8_t oldValue;
                    do {
                        // Select bytes from the old value and new value to construct a 32-bit value to use.
                        uint32_t const replacement = __byte_perm( longOldValue, longValue,   byteSelector );
                        uint32_t const comparison  = __byte_perm( longOldValue, longCompare, byteSelector );
    
                        longAssumed  = longOldValue;
                        // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
                        longOldValue = ::atomicCAS( baseAddress, comparison, replacement );
                        // Grab the 8-bit portion we care about from the old value at address.
                        oldValue     = ( longOldValue >> ( 8 * longAddressModulo )) & 0xFF;
                    } while ( compare == oldValue and longAssumed != longOldValue ); // Repeat until other three 8-bit values stabilize.
                    return oldValue;
                }
            }
            
    
    
            extern "C" {
    
                __global__ void get_occ_vox(
                    const float* in_data,   // B * N * 3
                    const int* in_actual_numpoints, // B 
                    const int B,
                    const int N,
                    const float *d_coord_shift,     // 3
                    const float *d_voxel_size,      // 3
                    const int *d_grid_size,       // 3
                    const int *kernel_size,       // 3
                    const int pixel_size,
                    const int grid_size_vol,
                    uint8_t *coor_occ,  // B * 400 * 400 * 400
                    int8_t *loc_coor_counter,  // B * 400 * 400 * 400
                    int *near_depth_id_tensor,  // B * 400 * 400
                    int *far_depth_id_tensor,  // B * 400 * 400 
                    const int inverse
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / N;  // index of batch
                    if (i_batch >= B) { return; }
                    int i_pt = index - N * i_batch;
                    if (i_pt < in_actual_numpoints[i_batch]) {
                        int coor[3];
                        const float *p_pt = in_data + index * 3;
                        coor[0] = floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
                        if (coor[0] < 0 || coor[0] >= d_grid_size[0]) { return; }
                        coor[1] = floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
                        if (coor[1] < 0 || coor[1] >= d_grid_size[1]) { return; }
                        float z = p_pt[2];
                        if (inverse > 0){ z = 1.0 / z;}
                        coor[2] = floor((z - d_coord_shift[2]) / d_voxel_size[2]);
                        if (coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
                        
                        int frust_id_b, coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
                        if (loc_coor_counter[coor_indx_b] < (int8_t)0 || cuda::atomicAdd(loc_coor_counter + coor_indx_b, (int8_t)-1) < (int8_t)0) { return; }
    
                        for (int coor_x = max(0, coor[0] - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
                            for (int coor_y = max(0, coor[1] - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                                for (int coor_z = max(0, coor[2] - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                                    frust_id_b = i_batch * pixel_size + coor_x * d_grid_size[1] + coor_y;
                                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                                    if (coor_occ[coor_indx_b] > (uint8_t)0) { continue; }
                                    cuda::atomicCAS(coor_occ + coor_indx_b, (uint8_t)0, (uint8_t)1);
                                    atomicMin(near_depth_id_tensor + frust_id_b, coor_z);
                                    atomicMax(far_depth_id_tensor + frust_id_b, coor_z);
                                }
                            }
                        }   
                    }
                }
    
                __global__ void near_vox_full(
                    const int B,
                    const int SR,
                    const int *pixel_idx,
                    const int R,
                    const int *vscale,
                    const int *d_grid_size,
                    const int pixel_size,
                    const int grid_size_vol,
                    const int *kernel_size,      // 3
                    uint8_t *pixel_map,
                    int8_t *ray_mask,     // B * R
                    const uint8_t *coor_occ,  // B * 400 * 400 * 400
                    int8_t *loc_coor_counter,    // B * 400 * 400 * 400
                    const int *near_depth_id_tensor,  // B * 400 * 400
                    const int *far_depth_id_tensor,  // B * 400 * 400 
                    short *voxel_to_coorz_idx  // B * 400 * 400 * SR 
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / R;  // index of batch
                    if (i_batch >= B) { return; }
                    int vx_id = pixel_idx[index*2] / vscale[0], vy_id = pixel_idx[index*2 + 1] / vscale[1];
                    int i_xyvox_id = i_batch * pixel_size + vx_id * d_grid_size[1] + vy_id;
                    int near_id = near_depth_id_tensor[i_xyvox_id], far_id = far_depth_id_tensor[i_xyvox_id];
                    ray_mask[index] = far_id > 0 ? (int8_t)1 : (int8_t)0;
                    if (pixel_map[i_xyvox_id] > (uint8_t)0 || cuda::atomicCAS(pixel_map + i_xyvox_id, (uint8_t)0, (uint8_t)1) > (uint8_t)0) { return; }
                    int counter = 0;
                    for (int depth_id = near_id; depth_id <= far_id; depth_id++) {
                        if (coor_occ[i_xyvox_id * d_grid_size[2] + depth_id] > (uint8_t)0) {
                            voxel_to_coorz_idx[i_xyvox_id * SR + counter] = (short)depth_id;
                            // if (i_xyvox_id>81920){
                            //    printf("   %d %d %d %d %d %d %d %d %d %d    ", pixel_idx[index*2], vscale[0], i_batch, vx_id, vy_id, i_xyvox_id * SR + counter, i_xyvox_id, SR, counter, d_grid_size[1]);
                            // }
                            for (int coor_x = max(0, vx_id - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], vx_id + (kernel_size[0] + 1) / 2); coor_x++)  {
                                for (int coor_y = max(0, vy_id - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], vy_id + (kernel_size[1] + 1) / 2); coor_y++)   {
                                    for (int coor_z = max(0, depth_id - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], depth_id + (kernel_size[2] + 1) / 2); coor_z++)    {
                                        int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                                        // cuda::atomicCAS(loc_coor_counter + coor_indx_b, (int8_t)-1, (int8_t)1);
                                        int8_t loc = loc_coor_counter[coor_indx_b];
                                        if (loc < (int8_t)0) {
                                            loc_coor_counter[coor_indx_b] = (int8_t)1;
                                        }
                                    }
                                }
                            }
                            if (counter >= SR - 1) { return; }
                            counter += 1;
                        }
                    }
                }
    
    
                __global__ void insert_vox_points(        
                    float* in_data,   // B * N * 3
                    int* in_actual_numpoints, // B 
                    const int B,
                    const int N,
                    const int P,
                    const int max_o,
                    const int pixel_size,
                    const int grid_size_vol,
                    const float *d_coord_shift,     // 3
                    const int *d_grid_size,
                    const float *d_voxel_size,      // 3
                    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
                    short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
                    int *voxel_to_pntidx,      // B * pixel_size * max_o * P
                    unsigned long seconds,
                    const int inverse
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / N;  // index of batch
                    if (i_batch >= B) { return; }
                    if (index - i_batch * N < in_actual_numpoints[i_batch]) {
                        const float *p_pt = in_data + index * 3;
                        int coor_x = (p_pt[0] - d_coord_shift[0]) / d_voxel_size[0];
                        int coor_y = (p_pt[1] - d_coord_shift[1]) / d_voxel_size[1];
                        float z = p_pt[2];
                        if (inverse > 0){ z = 1.0 / z;}
                        int coor_z = (z - d_coord_shift[2]) / d_voxel_size[2];
                        int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
                        int coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
                        if (coor_x < 0 || coor_x >= d_grid_size[0] || coor_y < 0 || coor_y >= d_grid_size[1] || coor_z < 0 || coor_z >= d_grid_size[2] || loc_coor_counter[coor_indx_b] < (int8_t)0) { return; }
                        int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[coor_indx_b];
                        //printf("voxel_indx_b, %d  ||   ", voxel_indx_b);
                        int voxel_pntid = (int) cuda::atomicAdd(voxel_pnt_counter + voxel_indx_b, (short)1);
                        if (voxel_pntid < P) {
                            voxel_to_pntidx[voxel_indx_b * P + voxel_pntid] = index;
                        } else {
                            curandState state;
                            curand_init(index+seconds, 0, 0, &state);
                            int insrtidx = ceilf(curand_uniform(&state) * (voxel_pntid+1)) - 1;
                            if(insrtidx < P){
                                voxel_to_pntidx[voxel_indx_b * P + insrtidx] = index;
                            }
                        }
                    }
                }                        
    
    
                __global__ void query_rand_along_ray(
                    const float* in_data,   // B * N * 3
                    const int B,
                    const int SR,               // num. samples along each ray e.g., 128
                    const int R,               // e.g., 1024
                    const int max_o,
                    const int P,
                    const int K,                // num.  neighbors
                    const int pixel_size,                
                    const int grid_size_vol,
                    const float radius_limit2,
                    const float depth_limit2,
                    const float *d_coord_shift,     // 3
                    const int *d_grid_size,
                    const float *d_voxel_size,      // 3
                    const float *d_ray_voxel_size,      // 3
                    const int *vscale,      // 3
                    const int *kernel_size,
                    const int *pixel_idx,               // B * R * 2
                    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
                    const short *voxel_to_coorz_idx,            // B * 400 * 400 * SR 
                    const short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
                    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
                    int *sample_pidx,       // B * R * SR * K
                    float *sample_loc,       // B * R * SR * K
                    unsigned long seconds,
                    const int NN,
                    const int inverse
                ) {
                    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / (R * SR);  // index of batch
                    int ray_idx_b = index / SR;  
                    if (i_batch >= B || ray_idx_b >= B * R) { return; }
    
                    int ray_sample_loc_idx = index - ray_idx_b * SR;
                    int frustx = pixel_idx[ray_idx_b * 2] / vscale[0];
                    int frusty = pixel_idx[ray_idx_b * 2 + 1] / vscale[1];
                    int vxy_ind_b = i_batch * pixel_size + frustx * d_grid_size[1] + frusty;
                    int frustz = (int) voxel_to_coorz_idx[vxy_ind_b * SR + ray_sample_loc_idx];
                    float centerx = d_coord_shift[0] + frustx * d_voxel_size[0] + (pixel_idx[ray_idx_b * 2] % vscale[0] + 0.5) * d_ray_voxel_size[0];
                    float centery = d_coord_shift[1] + frusty * d_voxel_size[1] + (pixel_idx[ray_idx_b * 2 + 1] % vscale[1] + 0.5) * d_ray_voxel_size[1];
                    float centerz = d_coord_shift[2] + (frustz + 0.5) * d_voxel_size[2];
                    if (inverse > 0){ centerz = 1.0 / centerz;}
                    sample_loc[index * 3] = centerx;
                    sample_loc[index * 3 + 1] = centery;
                    sample_loc[index * 3 + 2] = centerz;
                    if (frustz < 0) { return; }
                    int coor_indx_b = vxy_ind_b * d_grid_size[2] + frustz;
                    int raysample_startid = index * K;
                    int kid = 0;
                    curandState state;
                    for (int coor_x = max(0, frustx - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], frustx + (kernel_size[0] + 1) / 2); coor_x++) {
                        for (int coor_y = max(0, frusty - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], frusty + (kernel_size[1] + 1) / 2); coor_y++) {
                            int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
                            for (int coor_z = max(0, frustz - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], frustz + (kernel_size[2] + 1) / 2); coor_z++) {
                                int shift_coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
                                if(loc_coor_counter[shift_coor_indx_b] < (int8_t)0) {continue;}
                                int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[shift_coor_indx_b];
                                for (int g = 0; g < min(P, (int) voxel_pnt_counter[voxel_indx_b]); g++) {
                                    int pidx = voxel_to_pntidx[voxel_indx_b * P + g];
                                    if ((radius_limit2 == 0 || (in_data[pidx*3]-centerx) * (in_data[pidx*3]-centerx) + (in_data[pidx*3 + 1]-centery) * (in_data[pidx*3 + 1]-centery) <= radius_limit2) && (depth_limit2==0 || (in_data[pidx*3 + 2]-centerz) * (in_data[pidx*3 + 2]-centerz) <= depth_limit2)) { 
                                        if (kid++ < K) {
                                            sample_pidx[raysample_startid + kid - 1] = pidx;
                                        }
                                        else {
                                            curand_init(index+seconds, 0, 0, &state);
                                            int insrtidx = ceilf(curand_uniform(&state) * (kid)) - 1;
                                            if (insrtidx < K) {
                                                sample_pidx[raysample_startid + insrtidx] = pidx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                
                __global__ void query_neigh_along_ray_layered(
                    const float* in_data,   // B * N * 3
                    const int B,
                    const int SR,               // num. samples along each ray e.g., 128
                    const int R,               // e.g., 1024
                    const int max_o,
                    const int P,
                    const int K,                // num.  neighbors
                    const int pixel_size,                
                    const int grid_size_vol,
                    const float radius_limit2,
                    const float depth_limit2,
                    const float *d_coord_shift,     // 3
                    const int *d_grid_size,
                    const float *d_voxel_size,      // 3
                    const float *d_ray_voxel_size,      // 3
                    const int *vscale,      // 3
                    const int *kernel_size,
                    const int *pixel_idx,               // B * R * 2
                    const int8_t *loc_coor_counter,    // B * 400 * 400 * 400
                    const short *voxel_to_coorz_idx,            // B * 400 * 400 * SR 
                    const short *voxel_pnt_counter,      // B * 400 * 400 * max_o 
                    const int *voxel_to_pntidx,      // B * pixel_size * max_o * P
                    int *sample_pidx,       // B * R * SR * K
                    float *sample_loc,       // B * R * SR * K
                    unsigned long seconds,
                    const int NN,
                    const int inverse
                ) {
                    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / (R * SR);  // index of batch
                    int ray_idx_b = index / SR;  
                    if (i_batch >= B || ray_idx_b >= B * R) { return; }
    
                    int ray_sample_loc_idx = index - ray_idx_b * SR;
                    int frustx = pixel_idx[ray_idx_b * 2] / vscale[0];
                    int frusty = pixel_idx[ray_idx_b * 2 + 1] / vscale[1];
                    int vxy_ind_b = i_batch * pixel_size + frustx * d_grid_size[1] + frusty;
                    int frustz = (int) voxel_to_coorz_idx[vxy_ind_b * SR + ray_sample_loc_idx];
                    float centerx = d_coord_shift[0] + frustx * d_voxel_size[0] + (pixel_idx[ray_idx_b * 2] % vscale[0] + 0.5) * d_ray_voxel_size[0];
                    float centery = d_coord_shift[1] + frusty * d_voxel_size[1] + (pixel_idx[ray_idx_b * 2 + 1] % vscale[1] + 0.5) * d_ray_voxel_size[1];
                    float centerz = d_coord_shift[2] + (frustz + 0.5) * d_voxel_size[2];
                    if (inverse > 0){ centerz = 1.0 / centerz;}
                    sample_loc[index * 3] = centerx;
                    sample_loc[index * 3 + 1] = centery;
                    sample_loc[index * 3 + 2] = centerz;
                    if (frustz < 0) { return; }
                    // int coor_indx_b = vxy_ind_b * d_grid_size[2] + frustz;
                    int raysample_startid = index * K;
                    // curandState state;
                    
                    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
                    float far2 = 0.0;
                    float xyz2Buffer[KN];
                    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){
                        int zlayer = min((kernel_size[2]+1)/2-1, layer);
                        
                        for (int x = max(-frustx, -layer); x < min(d_grid_size[0] - frustx, layer+1); x++) {
                            for (int y = max(-frusty, -layer); y < min(d_grid_size[1] - frusty, layer+1); y++) {                              
                                coor_y = frusty + y;
                                coor_x = frustx + x;
                                int pixel_indx_b = i_batch * pixel_size  + coor_x * d_grid_size[1] + coor_y;
                                for (int z =  max(-frustz, -zlayer); z < min(d_grid_size[2] - frustz, zlayer + 1); z++) {
                                    //  if (max(abs(x),abs(y)) != layer || abs(z) != zlayer) continue;
                                    if (max(abs(x),abs(y)) != layer && ((zlayer == layer) ? (abs(z) != zlayer) : 1)) continue;
                                    // if (max(abs(x),abs(y)) != layer) continue;
                                    coor_z = z + frustz;
                                    
                                    int shift_coor_indx_b = pixel_indx_b * d_grid_size[2] + coor_z;
                                    if(loc_coor_counter[shift_coor_indx_b] < (int8_t)0) {continue;}
                                    int voxel_indx_b = pixel_indx_b * max_o + (int)loc_coor_counter[shift_coor_indx_b];                  
                                    for (int g = 0; g < min(P, (int) voxel_pnt_counter[voxel_indx_b]); g++) {
                                        int pidx = voxel_to_pntidx[voxel_indx_b * P + g];
                                        float x_v = (NN < 2) ? (in_data[pidx*3]-centerx) : (in_data[pidx*3] * in_data[pidx*3+2]-centerx*centerz) ;
                                        float y_v = (NN < 2) ? (in_data[pidx*3+1]-centery) : (in_data[pidx*3+1] * in_data[pidx*3+2]-centery*centerz) ;
                                        float xy2 = x_v * x_v + y_v * y_v;
                                        float z2 = (in_data[pidx*3 + 2]-centerz) * (in_data[pidx*3 + 2]-centerz);
                                        float xyz2 = xy2 + z2;
                                        if ((radius_limit2 == 0 || xy2 <= radius_limit2) && (depth_limit2==0 || z2 <= depth_limit2)){
                                            if (kid++ < K) {
                                                sample_pidx[raysample_startid + kid - 1] = pidx;
                                                xyz2Buffer[kid-1] = xyz2;
                                                if (xyz2 > far2){
                                                    far2 = xyz2;
                                                    far_ind = kid - 1;
                                                }
                                            } else {
                                                if (xyz2 < far2) {
                                                    sample_pidx[raysample_startid + far_ind] = pidx;
                                                    xyz2Buffer[far_ind] = xyz2;
                                                    far2 = xyz2;
                                                    for (int i = 0; i < K; i++) {
                                                        if (xyz2Buffer[i] > far2) {
                                                            far2 = xyz2Buffer[i];
                                                            far_ind = i;
                                                        }
                                                    }
                                                } 
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        """, no_extern_c=True)
        get_occ_vox = mod.get_function("get_occ_vox")
        near_vox_full = mod.get_function("near_vox_full")
        insert_vox_points = mod.get_function("insert_vox_points")

        query_along_ray = mod.get_function("query_neigh_along_ray_layered") if self.opt.NN > 0 else mod.get_function("query_rand_along_ray")
        return get_occ_vox, near_vox_full, insert_vox_points, query_along_ray


    def switch_pixel_id(self, pixel_idx_tensor, h):
        pixel_id = torch.cat([pixel_idx_tensor[..., 0:1], h - 1 - pixel_idx_tensor[..., 1:2]], dim=-1)
        # print("pixel_id", pixel_id.shape, torch.min(pixel_id, dim=-2)[0], torch.max(pixel_id, dim=-2)[0])
        return pixel_id


    def query_grid_point_index(self, h, w, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kMaxThreadsPerBlock = 1024):

        device = point_xyz_pers_tensor.device
        B, N = point_xyz_pers_tensor.shape[0], point_xyz_pers_tensor.shape[1]
        pixel_size = scaled_vdim[0] * scaled_vdim[1]
        grid_size_vol = pixel_size * scaled_vdim[2]
        d_coord_shift = range_gpu[:3]
        # ray_vsize_gpu = (vsize_gpu / vscale_gpu).astype(np.float32)

        pixel_idx_cur_tensor = pixel_idx_tensor.reshape(B, -1, 2).clone()
        R = pixel_idx_cur_tensor.shape[1]

        # print("kernel_size_gpu {}, SR {}, K {}, ranges {}, scaled_vsize {}, scaled_vdim {}, vscale {}, max_o {}, P {}, radius_limit {}, depth_limit {}, range_gpu {}, scaled_vsize_gpu {}, scaled_vdim_gpu {}, vscale_gpu {} ".format(kernel_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, pixel_idx_cur_tensor.shape))

        # print("point_xyz_pers_tensor", ranges, scaled_vdim_gpu, torch.min(point_xyz_pers_tensor, dim=-2)[0], torch.max(point_xyz_pers_tensor, dim=-2)[0])


        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        coor_occ_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], scaled_vdim[2]], dtype=torch.uint8, device=device)
        loc_coor_counter_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], scaled_vdim[2]], dtype=torch.int8, device=device)
        near_depth_id_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1]], scaled_vdim[2], dtype=torch.int32, device=device)
        far_depth_id_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1]], -1, dtype=torch.int32, device=device)

        self.get_occ_vox(
            Holder(point_xyz_pers_tensor),
            Holder(actual_numpoints_tensor),
            np.int32(B),
            np.int32(N),
            d_coord_shift,
            scaled_vsize_gpu,
            scaled_vdim_gpu,
            query_size_gpu,
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            Holder(coor_occ_tensor),
            Holder(loc_coor_counter_tensor),
            Holder(near_depth_id_tensor),
            Holder(far_depth_id_tensor),
            np.int32(self.inverse),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))

        # torch.cuda.synchronize()
        # print("near_depth_id_tensor", torch.min(near_depth_id_tensor), torch.max(far_depth_id_tensor),torch.max(loc_coor_counter_tensor), torch.max(torch.sum(coor_occ_tensor, dim=-1)), B*scaled_vdim[0]* scaled_vdim[1]*SR, pixel_size, scaled_vdim, vscale, scaled_vdim_gpu)

        gridSize = int((B * R + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        voxel_to_coorz_idx_tensor = torch.full([B, scaled_vdim[0], scaled_vdim[1], SR], -1, dtype=torch.int16, device=device)
        pixel_map_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1]], dtype=torch.uint8, device=device)
        ray_mask_tensor = torch.zeros([B, R], dtype=torch.int8, device=device)
        self.near_vox_full(
            np.int32(B),
            np.int32(SR),
            # Holder(self.switch_pixel_id(pixel_idx_cur_tensor,h)),
            Holder(pixel_idx_cur_tensor),
            np.int32(R),
            vscale_gpu,
            scaled_vdim_gpu,
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            query_size_gpu,
            Holder(pixel_map_tensor),
            Holder(ray_mask_tensor),
            Holder(coor_occ_tensor),
            Holder(loc_coor_counter_tensor),
            Holder(near_depth_id_tensor),
            Holder(far_depth_id_tensor),
            Holder(voxel_to_coorz_idx_tensor),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))

        # torch.cuda.synchronize()
        # print("voxel_to_coorz_idx_tensor max", torch.max(torch.sum(voxel_to_coorz_idx_tensor > -1, dim=-1)))
        # print("scaled_vsize_gpu",scaled_vsize_gpu, scaled_vdim_gpu)
        # print("ray_mask_tensor",ray_mask_tensor.shape, torch.min(ray_mask_tensor), torch.max(ray_mask_tensor))
        # print("pixel_idx_cur_tensor",pixel_idx_cur_tensor.shape, torch.min(pixel_idx_cur_tensor), torch.max(pixel_idx_cur_tensor))

        pixel_id_num_tensor = torch.sum(ray_mask_tensor, dim=-1)
        pixel_idx_cur_tensor = torch.masked_select(pixel_idx_cur_tensor, (ray_mask_tensor > 0)[..., None].expand(-1, -1, 2)).reshape(1, -1, 2)
        del coor_occ_tensor, near_depth_id_tensor, far_depth_id_tensor, pixel_map_tensor

        R = torch.max(pixel_id_num_tensor).cpu().numpy()
        # print("loc_coor_counter_tensor",loc_coor_counter_tensor.shape)
        loc_coor_counter_tensor = (loc_coor_counter_tensor > 0).to(torch.int8)
        loc_coor_counter_tensor = loc_coor_counter_tensor * torch.cumsum(loc_coor_counter_tensor, dtype=torch.int8, dim=-1) - 1

        if max_o is None:
            max_o = torch.max(loc_coor_counter_tensor).cpu().numpy().astype(np.int32) + 1
        # print("max_o", max_o)

        voxel_pnt_counter_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], max_o], dtype=torch.int16, device=device)
        voxel_to_pntidx_tensor = torch.zeros([B, scaled_vdim[0], scaled_vdim[1], max_o, P], dtype=torch.int32, device=device)
        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        ray_vsize_gpu = (scaled_vsize_gpu / vscale_gpu).astype(np.float32)

        seconds = time.time()
        self.insert_vox_points(
            Holder(point_xyz_pers_tensor),
            Holder(actual_numpoints_tensor),
            np.int32(B),
            np.int32(N),
            np.int32(P),
            np.int32(max_o),
            np.int32(pixel_size),
            np.int32(grid_size_vol),
            d_coord_shift,
            scaled_vdim_gpu,
            scaled_vsize_gpu,
            Holder(loc_coor_counter_tensor),
            Holder(voxel_pnt_counter_tensor),
            Holder(voxel_to_pntidx_tensor),
            np.uint64(seconds),
            np.int32(self.inverse),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))

        # torch.cuda.synchronize()
        # print("loc_coor_counter_tensor",loc_coor_counter_tensor.shape, torch.min(loc_coor_counter_tensor), torch.max(loc_coor_counter_tensor))
        # print("voxel_pnt_counter_tensor",voxel_pnt_counter_tensor.shape, torch.min(voxel_pnt_counter_tensor), torch.max(voxel_pnt_counter_tensor))
        # print("voxel_to_pntidx_tensor",voxel_to_pntidx_tensor.shape, torch.min(voxel_to_pntidx_tensor), torch.max(voxel_to_pntidx_tensor))

        sample_pidx_tensor = torch.full([B, R, SR, K], -1, dtype=torch.int32, device=device)
        sample_loc_tensor = torch.full([B, R, SR, 3], 0.0, dtype=torch.float32, device=device)
        gridSize = int((R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        seconds = time.time()


        # print(point_xyz_pers_tensor.shape, B, SR, R ,max_o, P, K, pixel_size, grid_size_vol, radius_limit, depth_limit, d_coord_shift, scaled_vdim_gpu, scaled_vsize_gpu, ray_vsize_gpu, vscale_gpu, kernel_size_gpu, pixel_idx_cur_tensor.shape, loc_coor_counter_tensor.shape, voxel_to_coorz_idx_tensor.shape, voxel_pnt_counter_tensor.shape, voxel_to_pntidx_tensor.shape, sample_pidx_tensor.shape, sample_loc_tensor.shape, gridSize)
        if R > 0:
            self.query_along_ray(
                Holder(point_xyz_pers_tensor),
                np.int32(B),
                np.int32(SR),
                np.int32(R),
                np.int32(max_o),
                np.int32(P),
                np.int32(K),
                np.int32(pixel_size),
                np.int32(grid_size_vol),
                np.float32(radius_limit ** 2),
                np.float32(depth_limit ** 2),
                d_coord_shift,
                scaled_vdim_gpu,
                scaled_vsize_gpu,
                ray_vsize_gpu,
                vscale_gpu,
                kernel_size_gpu,
                # Holder(self.switch_pixel_id(pixel_idx_cur_tensor,h)),
                Holder(pixel_idx_cur_tensor),
                Holder(loc_coor_counter_tensor),
                Holder(voxel_to_coorz_idx_tensor),
                Holder(voxel_pnt_counter_tensor),
                Holder(voxel_to_pntidx_tensor),
                Holder(sample_pidx_tensor),
                Holder(sample_loc_tensor),
                np.uint64(seconds),
                np.int32(self.opt.NN),
                np.int32(self.inverse),
                block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))


        # torch.cuda.synchronize()
        # print("max_o", max_o)
        # print("voxel_pnt_counter", torch.max(voxel_pnt_counter_tensor))
        # print("sample_pidx_tensor", torch.max(torch.sum(sample_pidx_tensor >= 0, dim=-1)))
        # print("sample_pidx_tensor min max", torch.min(sample_pidx_tensor), torch.max(sample_pidx_tensor))

        # print("sample_pidx_tensor", sample_pidx_tensor.shape, sample_pidx_tensor[0,80,3], sample_pidx_tensor[0,80,6], sample_pidx_tensor[0,80,9])


        # print("sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor", sample_pidx_tensor.shape, sample_loc_tensor.shape, pixel_idx_cur_tensor.shape, ray_mask_tensor.shape)


        return sample_pidx_tensor, sample_loc_tensor, pixel_idx_cur_tensor, ray_mask_tensor


def load_pnts(point_path, point_num):
    with open(point_path, 'rb') as f:
        print("point_file_path################", point_path)
        all_infos = pickle.load(f)
        point_xyz = all_infos["point_xyz"]
    print(len(point_xyz), point_xyz.dtype, np.mean(point_xyz, axis=0), np.min(point_xyz, axis=0),
          np.max(point_xyz, axis=0))
    np.random.shuffle(point_xyz)
    return point_xyz[:min(len(point_xyz), point_num), :]


def np_to_gpuarray(*args):
    result = []
    for x in args:
        if isinstance(x, np.ndarray):
            result.append(pycuda.gpuarray.to_gpu(x))
        else:
            print("trans",x)
    return result


def try_build(point_file, point_dir, ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, split=["train"], imgidx=0, gpu=0):
    point_path = os.path.join(point_dir, point_file)
    point_xyz = load_pnts(point_path, 819200000)  # 81920   233872
    imgs, poses, _, hwf, _ = load_blender_data(
        os.path.expandvars("${nrDataRoot}") + "/nerf/nerf_synthetic/{}".format(obj), split, half_res=False, testskip=1)
    H, W, focal = hwf
    plt.figure()
    plt.imshow(imgs[imgidx])
    point_xyz_pers = w2img(point_xyz, poses[imgidx], focal)
    point_xyz_tensor = torch.as_tensor(point_xyz, device="cuda:{}".format(gpu))[None, ...]
    # plt.show()
    point_xyz_pers_tensor = torch.as_tensor(point_xyz_pers, device="cuda:{}".format(gpu))[None, ...]
    actual_numpoints_tensor = torch.ones([1], device=point_xyz_tensor.device, dtype=torch.int32) * len(point_xyz)
    scaled_vsize = (vsize * vscale).astype(np.float32)
    scaled_vdim = np.ceil(vdim / vscale).astype(np.int32)
    print("vsize", vsize, "vdim", vdim, "scaled_vdim", scaled_vdim)
    range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, kernel_size_gpu = np_to_gpuarray(ranges, scaled_vsize, scaled_vdim, vscale, kernel_size)
    pixel_idx_tensor = torch.as_tensor(pixel_idx, device="cuda:{}".format(gpu), dtype=torch.int32)[None, ...]
    sample_pidx_tensor, pixel_idx_cur_tensor = build_grid_point_index(pixel_idx_tensor, point_xyz_pers_tensor, actual_numpoints_tensor, kernel_size_gpu, SR, K, ranges, scaled_vsize, scaled_vdim, vscale, max_o, P, radius_limit, depth_limit, range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, gpu=gpu)

    save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor,
                        pixel_idx_cur_tensor, vdim, vsize, ranges)


def w2img(point_xyz, transform_matrix, focal):
    camrot = transform_matrix[:3, :3]  # world 2 cam
    campos = transform_matrix[:3, 3]  #
    point_xyz_shift = point_xyz - campos[None, :]
    # xyz = np.sum(point_xyz_shift[:,None,:] * camrot.T, axis=-1)
    xyz = np.sum(camrot[None, ...] * point_xyz_shift[:, :, None], axis=-2)
    # print(xyz.shape, np.sum(camrot[None, None, ...] * point_xyz_shift[:,:,None], axis=-2).shape)
    xper = xyz[:, 0] / -xyz[:, 2]
    yper = xyz[:, 1] / xyz[:, 2]
    x_pixel = np.round(xper * focal + 400).astype(np.int32)
    y_pixel = np.round(yper * focal + 400).astype(np.int32)
    print("focal", focal, np.tan(.5 * 0.6911112070083618))
    print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel), np.min(y_pixel))
    print("per xmax xmin:", np.max(xper), np.min(xper), "per ymax ymin:", np.max(yper), np.min(yper), "per zmax zmin:",
          np.max(xyz[:, 2]), np.min(xyz[:, 2]))
    print("min perx", -400 / focal, "max perx", 400 / focal)
    background = np.ones([800, 800, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .2

    plt.figure()
    plt.imshow(background)

    return np.stack([xper, yper, -xyz[:, 2]], axis=-1)





def render_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    plt.figure()
    plt.imshow(background)


def save_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    image_dir = os.path.join(self.opt.checkpoints_dir, opt.name, 'images')
    image_file = os.path.join(image_dir)


def render_pixel_mask(pixel_xy_inds, w, h):
    y_pixel, x_pixel = pixel_xy_inds[0, :, 1], pixel_xy_inds[0, :, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .0
    plt.figure()
    plt.imshow(background)


def save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor,
                        pixel_idx_cur_tensor, vdim, vsize, ranges):
    B, R, SR, K = sample_pidx_tensor.shape
    # pixel_inds = torch.as_tensor([3210, 3217,3218,3219,3220, 3221,3222,3223,3224,3225,3226,3227,3228,3229,3230, 3231,3232,3233,3234,3235, 3236,3237,3238,3239,3240], device=sample_pidx_tensor.device, dtype=torch.int64)
    point_inds = sample_pidx_tensor[0, :, :, :]
    # point_inds = sample_pidx_tensor[0, pixel_inds, :, :]
    mask = point_inds > -1
    point_inds = torch.masked_select(point_inds, mask).to(torch.int64)
    queried_point_xyz_tensor = point_xyz_tensor[0, point_inds, :]
    queried_point_xyz = queried_point_xyz_tensor.cpu().numpy()
    print("queried_point_xyz.shape", B, R, SR, K, point_inds.shape, queried_point_xyz_tensor.shape,
          queried_point_xyz.shape)
    print("pixel_idx_cur_tensor", pixel_idx_cur_tensor.shape)
    render_pixel_mask(pixel_idx_cur_tensor.cpu().numpy(), vdim[0], vdim[1])

    render_mask_pers_points(point_xyz_pers_tensor[0, point_inds, :].cpu().numpy(), vsize, ranges, vdim[0], vdim[1])

    plt.show()



if __name__ == "__main__":
    obj = "lego"
    point_file = "{}.pkl".format(obj)
    point_dir = os.path.expandvars("${nrDataRoot}/nerf/nerf_synthetic_points/")
    r = 0.36000002589322094
    ranges = np.array([-r, -r, 2., r, r, 6.], dtype=np.float32)
    vdim = np.array([800, 800, 400], dtype=np.int32)
    vsize = np.array([2 * r / vdim[0], 2 * r / vdim[1], 4. / vdim[2]], dtype=np.float32)
    vscale = np.array([2, 2, 1], dtype=np.int32)
    SR = 24
    P = 16
    kernel_size = np.array([5, 5, 1], dtype=np.int32)
    radius_limit = 0  # r / 400 * 5 #r / 400 * 5
    depth_limit = 0  # 4. / 400 * 1.5 # r / 400 * 2
    max_o = None
    K = 32

    xrange = np.arange(0, 800, 1, dtype=np.int32)
    yrange = np.arange(0, 800, 1, dtype=np.int32)
    xv, yv = np.meshgrid(xrange, yrange, sparse=False, indexing='ij')
    pixel_idx = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # 20000 * 2
    gpu = 0
    imgidx = 3
    split = ["train"]

    if gpu < 0:
        import pycuda.autoinit
    else:
        drv.init()
        dev1 = drv.Device(gpu)
        ctx1 = dev1.make_context()
    try_build(point_file, point_dir, ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, split=split, imgidx=imgidx, gpu=0)