#include "../kernel_header.h"

__global__ void __kernel_permutate_r16(
    RTensor16 X, 
    RTensor16 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}

extern "C" void launch_perumutate_r16(
  RTensor16 X, RTensor16 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_r16),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_permutate_c16(
    CTensor16 X, 
    CTensor16 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}


extern "C" void launch_permutate_c16(
  CTensor16 X, CTensor16 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_c16),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}
#include "../kernel_header.h"

__global__ void __kernel_permutate_r32(
    RTensor32 X, 
    RTensor32 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}

extern "C" void launch_perumutate_r32(
  RTensor32 X, RTensor32 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_r32),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_permutate_c32(
    CTensor32 X, 
    CTensor32 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}


extern "C" void launch_permutate_c32(
  CTensor32 X, CTensor32 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_c32),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}
#include "../kernel_header.h"

__global__ void __kernel_permutate_r64(
    RTensor64 X, 
    RTensor64 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}

extern "C" void launch_perumutate_r64(
  RTensor64 X, RTensor64 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_r64),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_permutate_c64(
    CTensor64 X, 
    CTensor64 Y, 
    Permutation P
  ) {

  auto grid = cg::this_grid();

  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  len_t x_index[MAX_DIMS];

  len_t rem = tid;
  for (len_t i = 0; i < X.dims; ++i) {
    const len_t index = rem / X.strides[i];
    rem -= index * rem;
    x_index[i] = index;
  }

  len_t idx = 0;

  for (len_t i = 0; i < X.dims; ++i) {
    idx += x_index[P.order[i]] * Y.strides[i];
  }

  grid.sync();

  Y.values[idx] = X.values[tid];
}


extern "C" void launch_permutate_c64(
  CTensor64 X, CTensor64 Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_c64),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}
