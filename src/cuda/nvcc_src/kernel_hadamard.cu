#include "../kernel_header.h"

// CUDA Kernel for Vector hadamard
__global__ void __kernel_hadamard_RScalar(
  const RScalar *dev_a,
  const RScalar *dev_b,
  RScalar *dev_c,
  len_t N
) {
  // test comment
  
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N)
    dev_c[tid] = dev_a[tid] * dev_b[tid];
}

extern "C" void launch_hadamard_RScalar(
  StreamCtx stream,
  const RScalar* a,
  const RScalar* b, 
  RScalar* c, 
  len_t N
) {
  __kernel_hadamard_RScalar<<<DIMPAD(N, 1024), 1024, 0, getCtx(stream)>>>(a, b, c, N);
}

__global__ void __kernel_hadamard_CScalar(
  const CScalar *dev_a,
  const CScalar *dev_b,
  CScalar *dev_c,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {    
    dev_c[tid].r = (dev_a[tid].r * dev_b[tid].r - dev_a[tid].i * dev_b[tid].i);
    dev_c[tid].i = (dev_a[tid].r * dev_b[tid].i + dev_a[tid].i * dev_b[tid].r);
  }
}

extern "C" void launch_hadamard_CScalar(
  StreamCtx stream,
  const CScalar* a,
  const CScalar* b, 
  CScalar* c, 
  len_t N
) {
  __kernel_hadamard_CScalar<<<DIMPAD(N, 1024), 1024, 0, getCtx(stream)>>>(a, b, c, N);
}

