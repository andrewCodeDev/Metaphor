#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard_reverse
__global__ void __kernel_hadamard_reverse_r16(
  r16 *grads_a,
  const r16 *value_b,
  const r16 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    grads_a[tid] += value_b[tid] * grads_c[tid];
}

extern "C" void launch_hadamard_reverse_r16(
  r16 *grads_a,
  const r16 *value_b,
  const r16 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_r16<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_reverse_c16(
  c16 *grads_a,
  const c16 *value_b,
  const c16 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    grads_a[tid].r = (value_b[tid].r * grads_c[tid].r - value_b[tid].i * grads_c[tid].i);
    grads_a[tid].i = (value_b[tid].r * grads_c[tid].i + value_b[tid].i * grads_c[tid].r);
  }
}

extern "C" void launch_hadamard_reverse_c16(
  c16 *grads_a,
  const c16 *value_b,
  const c16 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_c16<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard_reverse
__global__ void __kernel_hadamard_reverse_r32(
  r32 *grads_a,
  const r32 *value_b,
  const r32 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    grads_a[tid] += value_b[tid] * grads_c[tid];
}

extern "C" void launch_hadamard_reverse_r32(
  r32 *grads_a,
  const r32 *value_b,
  const r32 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_r32<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_reverse_c32(
  c32 *grads_a,
  const c32 *value_b,
  const c32 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    grads_a[tid].r = (value_b[tid].r * grads_c[tid].r - value_b[tid].i * grads_c[tid].i);
    grads_a[tid].i = (value_b[tid].r * grads_c[tid].i + value_b[tid].i * grads_c[tid].r);
  }
}

extern "C" void launch_hadamard_reverse_c32(
  c32 *grads_a,
  const c32 *value_b,
  const c32 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_c32<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard_reverse
__global__ void __kernel_hadamard_reverse_r64(
  r64 *grads_a,
  const r64 *value_b,
  const r64 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    grads_a[tid] += value_b[tid] * grads_c[tid];
}

extern "C" void launch_hadamard_reverse_r64(
  r64 *grads_a,
  const r64 *value_b,
  const r64 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_r64<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

__global__ void __kernel_hadamard_reverse_c64(
  c64 *grads_a,
  const c64 *value_b,
  const c64 *grads_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    grads_a[tid].r = (value_b[tid].r * grads_c[tid].r - value_b[tid].i * grads_c[tid].i);
    grads_a[tid].i = (value_b[tid].r * grads_c[tid].i + value_b[tid].i * grads_c[tid].r);
  }
}

extern "C" void launch_hadamard_reverse_c64(
  c64 *grads_a,
  const c64 *value_b,
  const c64 *grads_c,
  len_t N
) {
  __kernel_hadamard_reverse_c64<<<GRID_1D(N), 32>>>(grads_a, value_b, grads_c, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

