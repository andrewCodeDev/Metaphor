#include "../kernel_header.h"

__global__ void __kernel_leaky_relu_RScalar(
  const RScalar *a_value,
        RScalar *b_value,
        RScalar coef,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // prevents ambiguous < lookup
  const RScalar zero = 0.0;
     
  if (tid < N) {
    const RScalar u = a_value[tid];
    b_value[tid] = (u < zero) ? u * coef : u;
  }
}

extern "C" void launch_leaky_relu_RScalar(
  StreamCtx stream,
  const RScalar* a,
        RScalar* b, 
        RScalar coef,
  len_t N
) {
  __kernel_leaky_relu_RScalar<<<GRID_1D(N), dim3(32), 0, getCtx(stream)>>>(a, b, coef, N);
}
