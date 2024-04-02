#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

struct MaxIndex_RScalar {
  RScalar val; unsigned idx;
};

// CUDA Kernel for Vector fill
__global__ void __kernel_max_key_ij_j_RScalar(
  const RScalar* __restrict src, // input
       unsigned* __restrict keys, // output
  len_t m,
  len_t n
) {
  using prc = precision<RScalar>;

  // this loop moves our blocks down the rows
  for (len_t m_step = 0; m_step < m; m_step += blockDim.y)
  {
    // find the max index for each row
    RScalar old_max = -Init<RScalar>::infinity();
    len_t max_idx = 0;

    // check if we're still in bounds for the rows
    const len_t m_pos = threadIdx.y + (blockIdx.y * blockDim.y) + m_step;

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const unsigned n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        const auto val = src[m_pos * n + n_pos];
        if(old_max < val){
          old_max = val;
          max_idx = n_pos;
        }
      }
    }

    const auto new_max = warpReduce<MaxOP>(old_max);

    // TODO: 
    //  This can assign to keys multiple times if
    //  the same max value exists in the tensor.
    //  Warp reduce doesn't like user types, so
    //  we may have to sum up shared memory instead.
    if (new_max == old_max) {
      keys[m_pos] = static_cast<unsigned>(max_idx);
    }
  }
}

extern "C" void launch_max_key_ij_j_RScalar(
  StreamCtx stream, 
  const RScalar* src, 
       unsigned* keys, // output
  len_t m,
  len_t n
) {
  // TODO: search for hyper parameters
  const dim3 grid_block(1, std::min(12ul, DIMPAD(m, WARP_SIZE)));
  const dim3 thread_block(WARP_SIZE, WARP_SIZE);
  __kernel_max_key_ij_j_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(src, keys, m, n);
}


