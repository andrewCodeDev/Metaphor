#include "../kernel_header.h"

__global__ void __kernel_setup_sort_pairs_i_RScalar(
  const RScalar* src, 
  SortPair_RScalar* pairs, 
  len_t n
){
  for (unsigned offset = 0; offset < n; offset += blockDim.x) {
      if (offset + threadIdx.x < n) {
          pairs[threadIdx.x + offset].val = src[threadIdx.x + offset];
          pairs[threadIdx.x + offset].key = offset + threadIdx.x;
      }
  }
}

extern "C" void launch_setup_sort_pairs_i_RScalar(
  StreamCtx stream,
  const RScalar* src, 
  SortPair_RScalar* pairs, 
  len_t n
){
  __kernel_setup_sort_pairs_i_RScalar<<<1, 1024, 0, getCtx(stream)>>>(src, pairs, n);
}


