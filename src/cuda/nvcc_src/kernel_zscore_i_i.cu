#include "../kernel_header.h"


#include "../kernel_header.h"

// This is experimental. I'm casting up to double to increase the precision
// of the calculation. This could have negative effects in terms of performance
// and may cause overflow/underflow? If the calculation was going to flow,
// I'd be surprised that it wouldn't with the original raw types.

// CUDA Kernel for Vector fill
__global__ void __kernel_zscore_i_i_RScalar(
  const RScalar* src, // input
        RScalar* dst, // output
  len_t n
) {
  using prc = precision<RScalar>;
  
  const len_t block_step = blockDim.x * 4;

  prc::type sum_raw = 0.0;
  prc::type sum_sqr = 0.0;
  
  for (len_t step = 0; step < n; step += block_step) {

    const len_t ofs = step + (threadIdx.x * 4);

    if ((ofs + 4) < n) {
      const auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src[ofs]);
      sum_sqr += prc::cast(rsqr(u.w));
      sum_sqr += prc::cast(rsqr(u.x));
      sum_sqr += prc::cast(rsqr(u.y));
      sum_sqr += prc::cast(rsqr(u.z));    
      sum_raw += prc::cast(u.w);
      sum_raw += prc::cast(u.x);
      sum_raw += prc::cast(u.y);
      sum_raw += prc::cast(u.z);
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        sum_sqr += prc::cast(rsqr(src[i]));  
        sum_raw += prc::cast(src[i]);  
      }
    }
  }

  sum_sqr = blockReduce<AddOP, WARP_SIZE>(
    sum_sqr, (threadIdx.y / WARP_SIZE), (threadIdx.x % WARP_SIZE)
  );

  sum_raw = blockReduce<AddOP, WARP_SIZE>(
    sum_raw, (threadIdx.y / WARP_SIZE), (threadIdx.x % WARP_SIZE)
  );

  const RScalar stddev = 1.0 / std::sqrt(sum_sqr);
  const RScalar mean = sum_raw / prc::cast(n);

  for (len_t step = 0; step < n; step += block_step) {

    const len_t ofs = step + (threadIdx.x * 4);

    if ((ofs + 4) < n) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src[ofs]);
      u.w = (u.w - mean) / stddev;
      u.x = (u.x - mean) / stddev;
      u.y = (u.y - mean) / stddev;
      u.z = (u.z - mean) / stddev;    
      *reinterpret_cast<coalesce<RScalar>::ptr>(&dst[ofs]) = u;
    }
    else if (ofs < n) {
      for (len_t i = ofs; i < n; ++i) {
        dst[i] = (src[i] - mean) / stddev;
      }
    }
  }
}

extern "C" void launch_zscore_i_i_RScalar(
  StreamCtx stream, 
  const RScalar* src, 
        RScalar* dst, 
  len_t n
) {
  // TODO: search for hyper parameters
  __kernel_zscore_i_i_RScalar<<<1, 1024, 0, getCtx(stream)>>>(src, dst, n);
}


