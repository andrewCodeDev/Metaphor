#include "../kernel_header.h"

__global__ void __kernel_dilate(
  const Scalar *dev_a,
  Scalar value,
        Scalar *dev_c,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
  if (tid < N) {
    dev_c[tid] = dev_a[tid] * value;
  }
}

extern "C" void launch_dilate_Scalar(
  const void* a,
  double value, 
        void* c, 
  len_t N,
  StreamCtx stream
) {
  __kernel_dilate<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(
    static_cast<const Scalar*>(a), 
    static_cast<Scalar>(value), 
    static_cast<Scalar*>(c),
    static_cast<unsigned>(N)
  );
}
