#include "../kernel_header.h"

__global__ void __kernel_dilate_reverse(
  const Scalar *dev_a,
  Scalar value,
        Scalar *dev_b,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_b[tid] += dev_a[tid] * value;
  }
}

extern "C" void launch_dilate_reverse_Scalar(
  const void* a,
  double value, 
        void* b, 
  len_t N,
  StreamCtx stream
) {
  __kernel_dilate_reverse<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(
    static_cast<const Scalar*>(a), 
    static_cast<Scalar>(value), 
    static_cast<Scalar*>(b),
    static_cast<unsigned>(N)
  );
}
