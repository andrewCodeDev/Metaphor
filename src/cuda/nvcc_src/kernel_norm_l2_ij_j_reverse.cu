#include "../kernel_header.h"

// This is experimental. I'm trying a "precision" type to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

__device__ __inline__ RScalar __grad_calc_RScalar(
  precision<RScalar>::type src_val, // current value
  precision<RScalar>::type dst_grd, // incoming gradient
  precision<RScalar>::type s_sum, // sum of squares
  precision<RScalar>::type g_sum, // sum of src * grd
  precision<RScalar>::type denom // sum of squares to 1.5 power
){
  return static_cast<RScalar>((dst_grd * (s_sum - rsqr(src_val)) - src_val * (g_sum - src_val * g_sum)) * denom);
}

// CUDA Kernel for Vector fill
__global__ void __kernel_norm_l2_ij_j_reverse_RScalar(
  const RScalar* __restrict src_value, // input
        RScalar* __restrict src_grads, // input
  const RScalar* __restrict dst_grads, // output
  len_t m,
  len_t n
) {
  using prc = precision<RScalar>;

  // this loop moves our blocks down the rows
  for (len_t m_step = 0; m_step < m; m_step += gridDim.y)
  {
    prc::type s_sum = 0.0;
    prc::type g_sum = 0.0;

    const len_t m_pos = threadIdx.y + (blockIdx.y * blockDim.y) + m_step;

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        const RScalar src_val = src_value[m_pos * n + n_pos];
        const RScalar dst_grd = dst_grads[m_pos * n + n_pos];
        s_sum += prc::cast(src_val * src_val);
        g_sum += prc::cast(src_val * dst_grd);
      }
    }

    // I regret putting thread sync in warp reduce now
    s_sum = warpReduce<AddOP>(s_sum);
    g_sum = warpReduce<AddOP>(g_sum);
    const prc::type denom = 1.0f / std::pow(s_sum, 1.5f);

    // this loop moves our blocks across the columns
    for (len_t n_step = 0; n_step < n; n_step += blockDim.x) {

      const len_t n_pos = threadIdx.x + n_step;

      if ((m_pos < m) && (n_pos < n)) {
        const RScalar src_val = src_value[m_pos * n + n_pos];
        const RScalar dst_grd = dst_grads[m_pos * n + n_pos];
        src_grads[m_pos * n + n_pos] = __grad_calc_RScalar(src_val, dst_grd, s_sum, g_sum, denom);
      }
    }
  }
}

extern "C" void launch_norm_l2_ij_j_reverse_RScalar(
  StreamCtx stream,
  const RScalar* src_value, // input
        RScalar* src_grads, // output
  const RScalar* dst_grads, // input
  len_t m,
  len_t n
) {
  // TODO: search for hyper parameters
  const dim3 grid_block(1, 12);
  const dim3 thread_block(WARP_SIZE, WARP_SIZE);
  __kernel_norm_l2_ij_j_reverse_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(
      src_value, src_grads, dst_grads, m, n
  );
}

