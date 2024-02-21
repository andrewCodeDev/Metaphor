#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard_reverse
__global__ void __kernel_hadamard_reverse_RScalar(
  RScalar *grads_a,
  const RScalar *value_b,
  const RScalar *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    grads_a[tid] += value_b[tid] * grads_c[tid];
}

extern "C" void launch_hadamard_reverse_RScalar(
  Stream stream,
  RScalar *grads_a,
  const RScalar *value_b,
  const RScalar *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_RScalar<<<1, GRID_1D(N), 32, getStream(stream)>>>(
      grads_a, value_b, grads_c, N
  );
}

__global__ void __kernel_hadamard_reverse_CScalar(
  CScalar *grads_a,
  const CScalar *value_b,
  const CScalar *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    grads_a[tid].r = (value_b[tid].r * grads_c[tid].r - value_b[tid].i * grads_c[tid].i);
    grads_a[tid].i = (value_b[tid].r * grads_c[tid].i + value_b[tid].i * grads_c[tid].r);
  }
}

extern "C" void launch_hadamard_reverse_CScalar(
  Stream stream,
  CScalar *grads_a,
  const CScalar *value_b,
  const CScalar *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_CScalar<<<1, GRID_1D(N), 32, getStream(stream)>>>(
    grads_a, value_b, grads_c, N
  );
}

