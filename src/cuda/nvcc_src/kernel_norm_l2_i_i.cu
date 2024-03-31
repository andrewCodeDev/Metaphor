#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

// CUDA Kernel for Vector fill
__global__ void __kernel_norm_l2_i_i_RScalar(
  const RScalar* src, // input
        RScalar* dst, // output
  len_t n
) {
  const len_t block_step = blockDim.x * 4;

  precision<RScalar>::type sum = 0.0;
  
  for (len_t step = 0; step < n; step += block_step) {

    const len_t ofs = step + (threadIdx.x * 4);

    if ((ofs + 4) < n) {
      const auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src[ofs]);
      sum += precision<RScalar>::cast(rsqr(u.w));
      sum += precision<RScalar>::cast(rsqr(u.x));
      sum += precision<RScalar>::cast(rsqr(u.y));
      sum += precision<RScalar>::cast(rsqr(u.z));    
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        sum += precision<RScalar>::cast(rsqr(src[i]));  
      }
    }
  }

  sum = blockReduce<AddOP, WARP_SIZE>(
    sum, (threadIdx.y / WARP_SIZE), (threadIdx.x % WARP_SIZE)
  );

  const RScalar norm = 1.0 / std::sqrt(sum);

  for (len_t step = 0; step < n; step += block_step) {

    const len_t ofs = step + (threadIdx.x * 4);

    if ((ofs + 4) < n) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src[ofs]);
      u.w = precision<RScalar>::cast(u.w * norm);
      u.x = precision<RScalar>::cast(u.x * norm);
      u.y = precision<RScalar>::cast(u.y * norm);
      u.z = precision<RScalar>::cast(u.z * norm);    
      *reinterpret_cast<coalesce<RScalar>::ptr>(&dst[ofs]) = u;
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        dst[i] = precision<RScalar>::cast(src[i] * norm);
      }
    }
  }
}

extern "C" void launch_norm_l2_i_i_RScalar(
  StreamCtx stream, 
  const RScalar* src, 
        RScalar* dst, 
  len_t n
) {
  // TODO: search for hyper parameters
  __kernel_norm_l2_i_i_RScalar<<<1, 1024, 0, getCtx(stream)>>>(src, dst, n);
}


