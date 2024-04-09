#include "../kernel_header.h"

// This is experimental. I'm trying a "precision" type to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

__device__ __inline__ RScalar __grad_calc_RScalar(
  precision<RScalar>::type x_val, // current value
  precision<RScalar>::type y_grd, // incoming gradient
  precision<RScalar>::type s_sum, // sum of squares
  precision<RScalar>::type g_sum, // sum of grads
  precision<RScalar>::type denom // sum of src * grd
){
  return static_cast<RScalar>((y_grd * s_sum - x_val * g_sum) / denom);
}

// CUDA Kernel for Vector fill
__global__ void __kernel_norm_l2_i_i_reverse_RScalar(
  const RScalar* src_value, // input
        RScalar* src_grads, // input
  const RScalar* dst_grads, // output
  len_t n
) {
  using prc = precision<RScalar>;
  
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
      s_sum += prc::cast(rsqr(g.w));
      s_sum += prc::cast(rsqr(g.x));
      s_sum += prc::cast(rsqr(g.y));
      s_sum += prc::cast(rsqr(g.z));
      g_sum += prc::cast(g.w * u.w);
      g_sum += prc::cast(g.x * u.x);
      g_sum += prc::cast(g.y * u.y);
      g_sum += prc::cast(g.z * u.z);
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        s_sum += prc::cast(rsqr(src_value[i]));  
        g_sum += prc::cast(src_value[i] * dst_grads[i]);  
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
      u.w = __grad_calc_RScalar(u.w, g.w, s_sum - prc::cast(rsqr(u.w)), g_sum - prc::cast(u.w * g.w), denom);
      u.x = __grad_calc_RScalar(u.x, g.x, s_sum - prc::cast(rsqr(u.x)), g_sum - prc::cast(u.x * g.x), denom);
      u.y = __grad_calc_RScalar(u.y, g.y, s_sum - prc::cast(rsqr(u.y)), g_sum - prc::cast(u.y * g.y), denom);
      u.z = __grad_calc_RScalar(u.z, g.z, s_sum - prc::cast(rsqr(u.z)), g_sum - prc::cast(u.z * g.z), denom);
      *reinterpret_cast<coalesce<RScalar>::ptr>(&src_grads[ofs]) = u;
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        const RScalar u = src_value[i];
        const RScalar g = dst_grads[i];
        src_grads[i] = __grad_calc_RScalar(u, g, s_sum - prc::cast(rsqr(u)), g_sum - prc::cast(u * g), denom);
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


