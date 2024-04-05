
#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

__global__ void __kernel_broadcast_i_ij_RScalar(
  const RScalar* __restrict src, // input
        RScalar* __restrict dst, // output
        RScalar alpha,
  len_t m,
  len_t n
) {
  using prc = precision<RScalar>;

  RScalar __attribute__((unused)) tmp;

  // this loop moves our blocks down the rows
  for (len_t m_step = 0; m_step < m; m_step += (gridDim.y * blockDim.y))
  {
    // check if we're still in bounds for the rows
    const len_t m_pos = threadIdx.y + (blockIdx.y * blockDim.y) + m_step;

    if (m_pos < m)
      tmp = src[m_pos];

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        dst[m_pos * n + n_pos] = tmp + alpha * dst[m_pos * n + n_pos];
      }
    }
  }
}

extern "C" void launch_broadcast_i_ij_RScalar(
  StreamCtx stream, 
  const RScalar* src, 
        RScalar* dst, 
        RScalar alpha,
  len_t m,
  len_t n
) {
  // TODO: search for hyper parameters
  const dim3 grid_block(1ul, std::min(12ul, DIMPAD(m, WARP_SIZE)));
  const dim3 thread_block(WARP_SIZE, WARP_SIZE);
  __kernel_broadcast_i_ij_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(src, dst, alpha, m, n);
}


