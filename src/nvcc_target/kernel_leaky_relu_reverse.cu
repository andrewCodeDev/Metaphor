#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_reverse_r16(
  const r16 *a_value,
        r16 *a_grads,
  const r16 *b_grads,
        r16 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  // prevents ambiguous < lookup
  const r16 zero = 0.0;

  if (tid < N) {
    a_grads[tid] += (a_value[tid] < zero) ? coef * b_grads[tid] : zero;
  }
}

extern "C" void launch_relu_leaky_reverse_r16(
  const r16 *a_value,
        r16 *a_grads,
  const r16 *b_grads,
        r16 coef,
  len_t N
) {
  __kernel_leaky_relu_reverse_r16<<<GRID_1D(N), 32>>>(a_value, a_grads, b_grads, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}
#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_reverse_r32(
  const r32 *a_value,
        r32 *a_grads,
  const r32 *b_grads,
        r32 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  // prevents ambiguous < lookup
  const r32 zero = 0.0;

  if (tid < N) {
    a_grads[tid] += (a_value[tid] < zero) ? coef * b_grads[tid] : zero;
  }
}

extern "C" void launch_relu_leaky_reverse_r32(
  const r32 *a_value,
        r32 *a_grads,
  const r32 *b_grads,
        r32 coef,
  len_t N
) {
  __kernel_leaky_relu_reverse_r32<<<GRID_1D(N), 32>>>(a_value, a_grads, b_grads, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}
#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_reverse_r64(
  const r64 *a_value,
        r64 *a_grads,
  const r64 *b_grads,
        r64 coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  // prevents ambiguous < lookup
  const r64 zero = 0.0;

  if (tid < N) {
    a_grads[tid] += (a_value[tid] < zero) ? coef * b_grads[tid] : zero;
  }
}

extern "C" void launch_relu_leaky_reverse_r64(
  const r64 *a_value,
        r64 *a_grads,
  const r64 *b_grads,
        r64 coef,
  len_t N
) {
  __kernel_leaky_relu_reverse_r64<<<GRID_1D(N), 32>>>(a_value, a_grads, b_grads, coef, N);
  CUDA_ASSERT(cudaDeviceSynchronize());
}
