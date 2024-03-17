#include "../kernel_header.h"

__global__ void __kernel_logistic_reverse_RScalar(
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    const RScalar one = RScalar(1.0f);
    const RScalar b = b_value[tid];
    a_grads[tid] += b * (one - b) * b_grads[tid];
  }
}

extern "C" void launch_logistic_reverse_RScalar(
  StreamCtx stream,
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  __kernel_logistic_reverse_RScalar<<<GRID_1D(N), dim3(32), 0, getCtx(stream)>>>(
    a_grads, b_value, b_grads, N
  );
}

