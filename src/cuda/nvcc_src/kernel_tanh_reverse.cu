#include "../kernel_header.h"

__global__ void __kernel_tanh_reverse_RScalar(
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    a_grads[tid] += (RScalar(1.0) - rsqr(b_value[0])) * b_grads[tid];
  }
}

extern "C" void launch_tanh_reverse_RScalar(
  StreamCtx stream,
        RScalar *a_grads,
  const RScalar *b_value,
  const RScalar *b_grads,
  len_t N
) {
  __kernel_tanh_reverse_RScalar<<<DIMPAD(N, 1024), 1024, 0, getCtx(stream)>>>(
    a_grads, b_value, b_grads, N
  );
}

