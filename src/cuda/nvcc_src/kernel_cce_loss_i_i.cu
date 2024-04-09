#include "../kernel_header.h"

__global__ void __kernel_cce_loss_i_i_RScalar(
    const RScalar* src_value, 
          RScalar* src_grads, 
          unsigned trg,
          RScalar* scratch,
          float* redux, // scalar
    len_t m
) {
  auto grid = cg::this_grid();

  const len_t idx = grid.thread_rank();

  // since we're coalescing by 4's for every thread in the
  // kernel across the grid, we need to adjust up by 4 times
  // the grid every time we make a shift over to the right.
  const len_t g_step = (blockDim.x * gridDim.x) * 4;

  RScalar grid_sum = RScalar(0.0f);
    
  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = step + idx * 4;

    if ((ofs + 4) < m) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src_value[ofs]);
      // calculate the loss
      const RScalar w = (ofs + 0 == trg) ? -rlog(MaxOP::apply(u.w, Init<RScalar>::epsilon())) : RScalar(0.0f);
      const RScalar x = (ofs + 1 == trg) ? -rlog(MaxOP::apply(u.x, Init<RScalar>::epsilon())) : RScalar(0.0f);
      const RScalar y = (ofs + 2 == trg) ? -rlog(MaxOP::apply(u.y, Init<RScalar>::epsilon())) : RScalar(0.0f);
      const RScalar z = (ofs + 3 == trg) ? -rlog(MaxOP::apply(u.z, Init<RScalar>::epsilon())) : RScalar(0.0f);
      grid_sum += static_cast<double>(w + x + y + z);
      // calculate the dydx
      if (src_value != src_grads) {
        auto s = reinterpret_cast<coalesce<RScalar>::ptr>(&src_grads[ofs]);
        u.w -= RScalar((ofs + 0 == trg) ? 1.0f : 0.0f);
        u.x -= RScalar((ofs + 1 == trg) ? 1.0f : 0.0f);
        u.y -= RScalar((ofs + 2 == trg) ? 1.0f : 0.0f);
        u.z -= RScalar((ofs + 3 == trg) ? 1.0f : 0.0f);
        *s = u;
      }
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        const RScalar x = src_value[i];
        const RScalar y = (i == trg) ? -rlog(MaxOP::apply(x, Init<RScalar>::epsilon())) : RScalar(0.0f);
        grid_sum += y;
        // calculate the dydx
        if (src_value != src_grads) {
          src_grads[i] = x - RScalar((i == trg) ? 1.0f : 0.0f);
        }
      }
    }
  }

  // continue calculating sum
  if (redux != nullptr)
  {
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
      *redux = grid_sum;
    }
  }
} 

extern "C" void launch_cce_loss_i_i_RScalar(
  StreamCtx stream,
  const RScalar* src_value, 
        RScalar* src_grads, 
        unsigned trg,
        RScalar* scratch,
        float* redux, // scalar
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
    (void*)&src_value, 
    (void*)&src_grads, 
    (void*)&trg, 
    (void*)&scratch, 
    (void*)&redux, 
    (void*)&m
  };

  CUDA_ASSERT(cudaLaunchCooperativeKernel(
    (void*)(__kernel_cce_loss_i_i_RScalar), grid, block, args, 0, getCtx(stream)
  ));
}
