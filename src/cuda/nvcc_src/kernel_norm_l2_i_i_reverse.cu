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
__global__ void __kernel_norm_l2_i_i_reverse_RScalar(
  const RScalar* src_value, // input
        RScalar* src_grads, // input
  const RScalar* dst_grads, // output
  len_t n
) {
  const len_t block_step = blockDim.x * 4;

  // recover raw norm and sum of squares  
  precision<RScalar>::type s_sum = 0.0;
  precision<RScalar>::type g_sum = 0.0;

  for (len_t step = 0; step < n; step += block_step) {

    const len_t ofs = (step + threadIdx.x) * 4;

    if ((ofs + 4) < n) {
      // sum the gradient times each raw value...
      const auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src_value[ofs]);
      const auto g = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&dst_grads[ofs]);      
      s_sum += precision<RScalar>::cast(rsqr(g.w));
      s_sum += precision<RScalar>::cast(rsqr(g.x));
      s_sum += precision<RScalar>::cast(rsqr(g.y));
      s_sum += precision<RScalar>::cast(rsqr(g.z));
      g_sum += precision<RScalar>::cast(g.w * u.w);
      g_sum += precision<RScalar>::cast(g.x * u.x);
      g_sum += precision<RScalar>::cast(g.y * u.y);
      g_sum += precision<RScalar>::cast(g.z * u.z);
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        s_sum += precision<RScalar>::cast(rsqr(src_value[i]));  
        g_sum += precision<RScalar>::cast(src_value[i] * dst_grads[i]);  
      }
    }
  }
  s_sum = blockReduce<AddOP, WARP_SIZE>(
    s_sum, (threadIdx.y / WARP_SIZE), (threadIdx.x % WARP_SIZE)
  );
  g_sum = blockReduce<AddOP, WARP_SIZE>(
    g_sum, (threadIdx.y / WARP_SIZE), (threadIdx.x % WARP_SIZE)
  );
  const precision<RScalar>::type denom = 1.0 / std::pow(s_sum, 1.5);

  for (len_t step = 0; step < n; step += block_step) {
    const len_t ofs = (step + threadIdx.x) * 4;

    if ((ofs + 4) < n) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src_value[ofs]);
      auto g = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&dst_grads[ofs]);
      u.w = __grad_calc_RScalar(u.w, g.w, s_sum, g_sum, denom);
      u.x = __grad_calc_RScalar(u.x, g.x, s_sum, g_sum, denom);
      u.y = __grad_calc_RScalar(u.y, g.w, s_sum, g_sum, denom);
      u.z = __grad_calc_RScalar(u.z, g.z, s_sum, g_sum, denom);
      *reinterpret_cast<coalesce<RScalar>::ptr>(&src_grads[ofs]) = u;
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        src_grads[i] = __grad_calc_RScalar(src_value[i], dst_grads[i], s_sum, g_sum, denom);
      }
    }
  }
}

extern "C" void launch_norm_l2_i_i_reverse_RScalar(
  StreamCtx stream, 
  const RScalar* src_value,
        RScalar* src_grads, 
  const RScalar* dst_grads, 
  len_t n
) {
  // TODO: search for hyper parameters
  __kernel_norm_l2_i_i_reverse_RScalar<<<1, 1024, 0, getCtx(stream)>>>(
    src_value, src_grads, dst_grads, n
  );
}


