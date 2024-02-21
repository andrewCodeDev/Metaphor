#include "../kernel_header.h"

// CUDA Kernel for Vector subtraction
__global__ void __kernel_subtraction_RScalar(
  const RScalar *dev_a,
  const RScalar *dev_b,
  RScalar *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] - dev_b[tid];
}

extern "C" void launch_subtraction_RScalar(
  Stream stream,
  const RScalar* a,
  const RScalar* b, 
  RScalar* c, 
  len_t N
) {
  __kernel_subtraction_RScalar<<<GRID_1D(N), dim3(32), 0, getStream(stream)>>>(a, b, c, N);
}

__global__ void __kernel_subtraction_CScalar(
  const CScalar *dev_a,
  const CScalar *dev_b,
  CScalar *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid].r = dev_a[tid].r - dev_b[tid].r;
    dev_c[tid].i = dev_a[tid].i - dev_b[tid].i;
  }
}

extern "C" void launch_subtraction_CScalar(
  Stream stream,
  const CScalar* a,
  const CScalar* b, 
  CScalar* c, 
  len_t N
) {
  __kernel_subtraction_CScalar<<<GRID_1D(N), dim3(32), 0, getStream(stream)>>>(a, b, c, N);
}
