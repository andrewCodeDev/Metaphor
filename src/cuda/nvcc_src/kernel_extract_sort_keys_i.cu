#include "../kernel_header.h"

__global__ void __kernel_extract_sort_keys_i_RScalar(
  const SortPair_RScalar* pairs, 
  unsigned* keys, 
  len_t n
){
  for (unsigned offset = 0; offset < n; offset += blockDim.x) {
      if (offset + threadIdx.x < n) {
          keys[threadIdx.x + offset] = pairs[offset + threadIdx.x].key;
      }
  }
}

extern "C" void launch_extract_sort_keys_i_RScalar(
  StreamCtx stream, 
  const SortPair_RScalar* pairs, 
  unsigned* keys, 
  len_t n
) {
    __kernel_extract_sort_keys_i_RScalar<<<1, 1024, 0, getCtx(stream)>>>(pairs, keys, n);
}
