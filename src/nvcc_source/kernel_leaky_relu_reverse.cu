#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_reverse_RScalar(
  const RScalar *a_value,
        RScalar *a_grads,
  const RScalar *b_grads,
        RScalar coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  // prevents ambiguous < lookup
  const RScalar zero = 0.0;

  if (tid < N) {
    a_grads[tid] += (a_value[tid] < zero) ? coef * b_grads[tid] : zero;
  }
}

extern "C" void launch_relu_leaky_reverse_RScalar(
  Stream stream,
  const RScalar *a_value,
        RScalar *a_grads,
  const RScalar *b_grads,
        RScalar coef,
  len_t N
) {
  __kernel_leaky_relu_reverse_RScalar<<<1, GRID_1D(N), 32, getStream(stream)>>>(
    a_value, a_grads, b_grads, coef, N
  );
}
