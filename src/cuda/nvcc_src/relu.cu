#include "../kernel_header.h"

__global__ void __kernel_relu(
  const Scalar *a,
        Scalar *b,
  unsigned n
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n) {
    const Scalar value = a[tid];    
    b[tid] = (value >= Scalar(0.0)) ? value : Scalar(0.0);
  }
}

extern "C" void launch_relu_Scalar(
  const void* a,
        void* b, 
  len_t n,
  StreamCtx stream
) {
  __kernel_relu<<<DIMPAD(n, 1024), dim3(1024), 0, getCtx(stream)>>>(
    static_cast<const Scalar*>(a),
    static_cast<Scalar*>(b),
    static_cast<unsigned>(n)
  );
}
