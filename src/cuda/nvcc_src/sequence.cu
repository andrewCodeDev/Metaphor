
#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_sequence(
  Scalar *dev_a,
  Scalar init,
  Scalar step,
  unsigned N
) {
  const unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    dev_a[tid] = init + (step * Scalar(tid));
  }
}

extern "C" void launch_sequence_Scalar(
  void* dev_a,
  double init,
  double step,
  len_t N,
  StreamCtx stream
) {
  __kernel_sequence<<<GRID_1D(N), dim3(1024), 0, getCtx(stream)>>>(
    static_cast<Scalar*>(dev_a),
    static_cast<Scalar>(init),
    static_cast<Scalar>(step),
    static_cast<unsigned>(N)
  );
}
