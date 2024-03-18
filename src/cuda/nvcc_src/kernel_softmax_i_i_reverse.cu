#include "../kernel_header.h"

__global__ void __kernel_softmax_i_i_reverse_RScalar(
        RScalar* a_grads, 
  const RScalar* b_value, 
  const RScalar* b_grads,
        RScalar* scratch,
  len_t m
) {
  auto grid = cg::this_grid();

  const len_t idx = grid.thread_rank();

  // since we're coalescing by 4's for every thread in the
  // kernel across the grid, we need to adjust up by 4 times
  // the grid every time we make a shift over to the right.
  const len_t g_step = (WARP_SIZE * WARP_SIZE * gridDim.x) * 4;

  RScalar grid_sum = RScalar(0.0f);
    
  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = (step + idx) * 4;

    if ((ofs + 4) < m) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&b_value[ofs]);
      auto v = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&b_grads[ofs]);
      grid_sum += (u.w * v.w) + (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        grid_sum += b_value[i] * b_grads[i];
      }
    }
  }

  grid_sum = blockReduce<AddOP, WARP_SIZE>(
    grid_sum, threadIdx.y, threadIdx.x
  );

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    scratch[blockIdx.x] = grid_sum;
  }

  grid.sync();

  grid_sum = RScalar(0.0f);

  if (idx < WARP_SIZE) { // get only first warp

    for (len_t step = 0; step < gridDim.x; step += (WARP_SIZE * 4)) {

      const len_t ofs = (step + idx) * 4;

      if ((ofs + 4) < gridDim.x) {
        auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&scratch[ofs]);
        grid_sum += u.w + u.x + u.y + u.z;
      }
      else if (ofs < gridDim.x) {
        for (len_t i = ofs; i < gridDim.x; ++i) {
          grid_sum += scratch[i];  
        }
      }
    }
  }

  grid_sum = warpReduce<AddOP>(grid_sum);
  
  if (idx == 0) {
    scratch[0] = grid_sum;
  }

  grid.sync();

  grid_sum = scratch[0];
  
  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = (step + idx) * 4;

    if ((ofs + 4) < m) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&b_value[ofs]);
      auto v = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&b_grads[ofs]);
      auto g4 = reinterpret_cast<coalesce<RScalar>::ptr>(&a_grads[ofs]);
      u.w = u.w * (v.w - grid_sum);
      u.x = u.x * (v.x - grid_sum);
      u.y = u.y * (v.y - grid_sum);
      u.z = u.z * (v.z - grid_sum);
      *g4 = u;
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        a_grads[i] = b_value[i] * (b_grads[i] - grid_sum);  
      }
    }
  }
} 

extern "C" void launch_softmax_i_i_reverse_RScalar(
  StreamCtx stream,
        RScalar* a_grads, 
  const RScalar* b_value, 
  const RScalar* b_grads,
        RScalar* scratch,
  len_t m
) {

  dim3 grid (
    DIMPAD(m, (WARP_SIZE * WARP_SIZE * 4)), 1
  );

  dim3 block (
    WARP_SIZE, 
    WARP_SIZE
  );

  void* args[] = { 
    (void*)&a_grads,
    (void*)&b_value,
    (void*)&b_grads,
    (void*)&scratch, 
    (void*)&m
  };

  CUDA_ASSERT(cudaLaunchCooperativeKernel(
    (void*)(__kernel_softmax_i_i_reverse_RScalar), grid, block, args, 0, getCtx(stream)
  ));
}
