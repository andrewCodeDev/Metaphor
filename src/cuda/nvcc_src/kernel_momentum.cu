#include "../kernel_header.h"

__global__ void __kernel_momentum_RScalar(
        RScalar *a_value,
  const RScalar *a_grads,
        RScalar *mtm,
  RScalar rate,  // learning rate
  RScalar alpha, // momentum coeficient
  RScalar lower, // clip lower
  RScalar upper, // clip upper
  len_t n
) {
  // TODO: update this kernel with block limiting
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < n) {
    const RScalar g = ClipOP::apply(a_grads[tid], lower, upper);
    const RScalar m = mtm[tid];
    const RScalar upd = alpha * m + g;
    a_value[tid] -= rate * upd; 
    mtm[tid] = upd;
  }
}

extern "C" void launch_momentum_RScalar(
  StreamCtx stream,
        RScalar* a_value,
  const RScalar* a_grads, 
        RScalar* mtm,
  RScalar rate,
  RScalar alpha,
  RScalar lower,
  RScalar upper,
  len_t n
) {
  // TODO: update this kernel with block limiting
  __kernel_momentum_RScalar<<<DIMPAD(n, 1024), dim3(1024), 0, getCtx(stream)>>>(
    a_value, a_grads, mtm, rate, alpha, lower, upper, n
  );
}
