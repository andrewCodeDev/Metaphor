#include "../kernel_header.h"

__global__ void __kernel_sigmoid_RScalar(
  const RScalar *a_value,
        RScalar *b_value,
  len_t N
) {
  // TODO: this function needs to be rewritten
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    b_value[tid] = rtanh(a_value[tid]);
  }
}

extern "C" void launch_sigmoid_RScalar(
  StreamCtx stream,
  const RScalar* a,
        RScalar* b, 
  len_t N
) {
  // TODO: implement block limiting
  __kernel_sigmoid_RScalar<<<DIMPAD(N, 1024), 1024, 0, getCtx(stream)>>>(a, b, N);
}
