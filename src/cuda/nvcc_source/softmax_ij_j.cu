#include "../kernel_header.h"

template<int cols_per_thread>
__global__ void __kernel_softmax_ij_j_small_vec4_RScalar(
    const RScalar* A, 
          RScalar* B, 
    len_t m, 
    len_t n
) {
  constexpr int num_packs = (cols_per_thread + 3) / 4; // pack_size = 4, k/32 = cols_per_thread, num_packs = k/32/4
  coalesce<float> buf[num_packs];

  auto inp = reinterpret_cast<coalesce<RScalar>::c_ptr>(A);
  auto out = reinterpret_cast<coalesce<RScalar>::ptr>(B);

  const int m_idx = blockIdx.x * blockDim.y + threadIdx.y; // blockDim.y=4=thread_group_per_block
  const int tid = threadIdx.x;

  for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {

    const int64_t row_offset = row * (n >> 2);
    coalesce<RScalar>::c_ptr row_x = inp + row_offset;
    coalesce<RScalar>::ptr   row_y = out + row_offset;

    // float for size/precision considerations
    float local_max[1] = { -std::numeric_limits<float>::infinity() };

#pragma unroll
    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = pack_id * blockDim.x + tid;
     // row_y[col] = row_x[col];
      if (col < n/4) {
        buf[pack_id] = {
          static_cast<float>(row_x[col].w),
          static_cast<float>(row_x[col].x),
          static_cast<float>(row_x[col].y),
          static_cast<float>(row_x[col].z)
        };
        local_max[0] = max(
            local_max[0], max(max(buf[pack_id].w, buf[pack_id].x), max(buf[pack_id].y, buf[pack_id].z))
        );
      } else {
        buf[pack_id].x = -std::numeric_limits<float>::infinity();
        buf[pack_id].y = -std::numeric_limits<float>::infinity();
        buf[pack_id].z = -std::numeric_limits<float>::infinity();
        buf[pack_id].w = -std::numeric_limits<float>::infinity();
      }
    }
    warpReduce<MaxOP, 1>(local_max); //cal the actual max among cols

    float local_sum[1] = {0.0f};
#pragma unroll
    for (int i = 0; i < num_packs; ++i) {
      buf[i].w = exp(buf[i].w - local_max[0]);
      buf[i].x = exp(buf[i].x - local_max[0]);
      buf[i].y = exp(buf[i].y - local_max[0]);
      buf[i].z = exp(buf[i].z - local_max[0]);
      local_sum[0] += buf[i].w;
      local_sum[0] += buf[i].x;
      local_sum[0] += buf[i].y;
      local_sum[0] += buf[i].z;
    }
    warpReduce<AddOP, 1>(local_sum);

    for (int i = 0; i < num_packs; ++i) {
      const int col = i * blockDim.x + tid;
      if (col < n / 4) {
        row_y[col] = { 
            static_cast<RScalar>(buf[i].x / local_sum[0]), 
            static_cast<RScalar>(buf[i].y / local_sum[0]), 
            static_cast<RScalar>(buf[i].z / local_sum[0]), 
            static_cast<RScalar>(buf[i].w / local_sum[0]) 
        };
      }
    }
  }
} 

extern "C" void launch_softmax_ij_j_RScalar(
  StreamCtx stream,
  const RScalar* A, 
        RScalar* B, 
  len_t m, 
  len_t n
) {
  // TODO: Finish softmax implementations

  //const len_t packs = n / pack_size;

  //if (packs > 32) {

  //  if (DIVISIBLE_BY_FOUR(m)) {
  //    __kernel_softmax_ij_j_small_vec4_RScalar<
  //        <<<dim3(m/packs), dim3(32, 4), 0, getCtx(stream)>>> (A, B, m, n);
  //    
  //  }

  //  
  //}

  
}
