#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


std::vector<torch::Tensor> woord_query_grid_point_index_cuda(
        torch::Tensor pixel_idx_tensor,
        torch::Tensor raypos_tensor,
        torch::Tensor point_xyz_w_tensor,
        torch::Tensor actual_numpoints_tensor,
        torch::Tensor kernel_size,
        torch::Tensor query_size,
        const int SR,
        const int K,
        const int R,
        const int D,
        torch::Tensor scaled_vdim,
        const int max_o,
        const int P,
        const float radius_limit,
        torch::Tensor ranges,
        torch::Tensor scaled_vsize,
        const int kMaxThreadsPerBlock,
        const int NN);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> woord_query_grid_point_index(
        torch::Tensor pixel_idx_tensor,
        torch::Tensor raypos_tensor,
        torch::Tensor point_xyz_w_tensor,
        torch::Tensor actual_numpoints_tensor,
        torch::Tensor kernel_size,
        torch::Tensor query_size,
        const int SR,
        const int K,
        const int R,
        const int D,
        torch::Tensor scaled_vdim,
        const int max_o,
        const int P,
        const float radius_limit,
        torch::Tensor ranges,
        torch::Tensor scaled_vsize,
        const int kMaxThreadsPerBlock,
        const int NN){
//  CHECK_INPUT(pixel_idx_tensor);
//  CHECK_INPUT(raypos_tensor);
//  CHECK_INPUT(point_xyz_w_tensor);
//  CHECK_INPUT(actual_numpoints_tensor);
  return woord_query_grid_point_index_cuda(
        pixel_idx_tensor,
        raypos_tensor,
        point_xyz_w_tensor,
        actual_numpoints_tensor,
        kernel_size,
        query_size,
        SR,
        K,
        R,
        D,
        scaled_vdim,
        max_o,
        P,
        radius_limit,
        ranges,
        scaled_vsize,
        kMaxThreadsPerBlock,
        NN);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("woord_query_grid_point_index", &woord_query_grid_point_index, "woord_query_grid_point_index world coordinate");
}


