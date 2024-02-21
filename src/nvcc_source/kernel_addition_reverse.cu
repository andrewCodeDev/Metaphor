#include "../kernel_header.h"

__global__ void __kernel_addition_reverse_RScalar(
  RScalar *dev_a,
  const RScalar *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_a[tid] += dev_b[tid];
}

extern "C" void launch_addition_reverse_RScalar(
  Stream stream,
  RScalar* a, 
  const RScalar* b, 
  len_t N
) {
  __kernel_addition_reverse_RScalar<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}

__global__ void __kernel_addition_reverse_CScalar(
  CScalar *dev_a,
  const CScalar *dev_b,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_a[tid].r += dev_b[tid].r;
    dev_a[tid].i += dev_b[tid].i;
  }
}

extern "C" void launch_addition_reverse_CScalar(
  Stream stream,
  CScalar* a, 
  const CScalar* b, 
  len_t N
) {
  __kernel_addition_reverse_CScalar<<<1, GRID_1D(N), 32, getStream(stream)>>>(a, b, N);
}
