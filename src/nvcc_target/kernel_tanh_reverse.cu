#include "../kernel_header.h"

__global__ void __kernel_tanh_reverse_r16(
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const r16 one = 1.0;
    const r16 b = b_value[tid];
    a_grads[tid] += (one - (b * b)) * b_grads[tid];
  }
}

extern "C" void launch_tanh_reverse_r16(
        r16 *a_grads,
  const r16 *b_value,
  const r16 *b_grads,
  len_t N
) {
  __kernel_tanh_reverse_r16<<<GRID_1D(N), 32>>>(a_grads, b_value, b_grads, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

__global__ void __kernel_tanh_reverse_r32(
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const r32 one = 1.0;
    const r32 b = b_value[tid];
    a_grads[tid] += (one - (b * b)) * b_grads[tid];
  }
}

extern "C" void launch_tanh_reverse_r32(
        r32 *a_grads,
  const r32 *b_value,
  const r32 *b_grads,
  len_t N
) {
  __kernel_tanh_reverse_r32<<<GRID_1D(N), 32>>>(a_grads, b_value, b_grads, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

#include "../kernel_header.h"

__global__ void __kernel_tanh_reverse_r64(
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const r64 one = 1.0;
    const r64 b = b_value[tid];
    a_grads[tid] += (one - (b * b)) * b_grads[tid];
  }
}

extern "C" void launch_tanh_reverse_r64(
        r64 *a_grads,
  const r64 *b_value,
  const r64 *b_grads,
  len_t N
) {
  __kernel_tanh_reverse_r64<<<GRID_1D(N), 32>>>(a_grads, b_value, b_grads, N);

  CUDA_ASSERT(cudaDeviceSynchronize());
}

