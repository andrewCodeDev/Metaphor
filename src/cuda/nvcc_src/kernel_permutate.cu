#include "../kernel_header.h"

__global__ void __kernel_permutate_naive_RScalar(
    RTensor X, 
    RTensor Y, 
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

extern "C" void launch_perumutate_RScalar(
  RTensor X, RTensor Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_naive_RScalar),
    dim3(DIMPAD(X.len, 128)),
    dim3(128),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_permutate_naive_CScalar(
    CTensor X, 
    CTensor Y, 
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


extern "C" void launch_permutate_naive_CScalar(
  CTensor X, CTensor Y, Permutation P
) {

  void* args[] = { 
    (void*)&X, (void*)&Y, (void*)&P
  };
  
  cudaLaunchCooperativeKernel(
    (void*)(__kernel_permutate_naive_CScalar),
    dim3(GRID_1D(X.len)),
    dim3(32),
    args
  );

  CUDA_ASSERT(cudaDeviceSynchronize());
}
