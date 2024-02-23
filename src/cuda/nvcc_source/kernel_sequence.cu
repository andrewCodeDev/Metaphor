
#include "../kernel_header.h"

// CUDA Kernel for Vector fill
__global__ void __kernel_sequence_RScalar(
  RScalar *dev_a,
  RScalar init,
  RScalar step,
  len_t N
) {
  const len_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < N) {
    dev_a[tid] = init + (step * RScalar(tid));
  }
}

extern "C" void launch_sequence_RScalar(
  Stream stream,
  RScalar* dev_a,
  RScalar init,
  RScalar step,
  len_t N
) {
  __kernel_sequence_RScalar<<<GRID_1D(N), dim3(1024), 0, getStream(stream)>>>(
    dev_a, init, step, N
  );
}
