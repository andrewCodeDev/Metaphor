#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_fill_RScalar(
  RScalar *dev_a,
  RScalar value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] = value;
}

extern "C" void launch_fill_RScalar(
  RScalar* dev_a,
  RScalar value, 
  len_t N
) {
  __kernel_fill_RScalar<<<GRID_1D(N), 32>>>(dev_a, value, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_fill_CScalar(
  CScalar *dev_a,
  CScalar value,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r = value.r;
    dev_a[tid].i = value.i;
  }
}

extern "C" void launch_fill_CScalar(
  CScalar* dev_a,
  CScalar value, 
  len_t N
) {
  __kernel_fill_CScalar<<<GRID_1D(N), 32>>>(dev_a, value, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}


