#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

// CUDA Kernel for Vector fill
__global__ void __kernel_norm_l2_ij_j_RScalar(
  const RScalar* __restrict src, // input
        RScalar* __restrict dst, // output
  len_t m,
  len_t n
) {
  using prc = precision<RScalar>;

  // this loop moves our blocks down the rows
  for (len_t m_step = 0; m_step < m; m_step += gridDim.y)
  {
    prc::type sum = 0.0;

    // check if we're still in bounds for the rows
    const len_t m_pos = threadIdx.y + (blockIdx.y * blockDim.y) + m_step;

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        sum += prc::cast(rsqr(src[m_pos * n + n_pos]));
      }
    }

    sum = warpReduce<AddOP>(sum);

    const prc::type norm = 1.0f / std::sqrt(sum);

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        dst[m_pos * n + n_pos] = RScalar(prc::cast(src[m_pos * n + n_pos]) * norm);
      }
    }
  }
}

extern "C" void launch_norm_l2_ij_j_RScalar(
  StreamCtx stream, 
  const RScalar* src, 
        RScalar* dst, 
  len_t m,
  len_t n
) {
  // TODO: search for hyper parameters
  const dim3 grid_block(1, 12);
  const dim3 thread_block(WARP_SIZE, WARP_SIZE);
  __kernel_norm_l2_ij_j_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(src, dst, m, n);
}


