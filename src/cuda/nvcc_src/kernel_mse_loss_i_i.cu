#include "../kernel_header.h"

__global__ void __kernel_mse_loss_i_i_RScalar(
    const RScalar* src_value, 
          RScalar* src_grads, 
    const RScalar* trg_value, 
          RScalar* scratch,
          float*  redux, // scalar
    len_t m
) {
  auto grid = cg::this_grid();

  const len_t idx = grid.thread_rank();

  // since we're coalescing by 4's for every thread in the
  // kernel across the grid, we need to adjust up by 4 times
  // the grid every time we make a shift over to the right.
  const len_t g_step = (WARP_SIZE * WARP_SIZE * gridDim.x) * 4;

  const RScalar coef = 1.0 / (static_cast<double>(m));
    
  RScalar grid_sum = RScalar(0.0f);

  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = (step + idx) * 4;

    if ((ofs + 4) < m) {
      coalesce<RScalar> u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src_value[ofs]);
      coalesce<RScalar> v = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&trg_value[ofs]);      
      u.w = (u.w - v.w);
      u.x = (u.x - v.x);
      u.y = (u.y - v.y);
      u.z = (u.z - v.z);
      grid_sum += rsqr(u.w) + rsqr(u.x) + rsqr(u.y) + rsqr(u.z);
      // calculate the dydx
      if (src_value != src_grads) {
        auto s = reinterpret_cast<coalesce<RScalar>::ptr>(&src_grads[ofs]);
        u.w *= coef;
        u.x *= coef;
        u.y *= coef;
        u.z *= coef;
        *s = u;
      }
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        const RScalar dif = src_value[i] - trg_value[i];
        grid_sum += rsqr(dif);
        // calculate the dydx
        if (src_value != src_grads) {
          src_grads[i] = dif * coef;
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
      *redux = grid_sum * coef;
    }
  }
} 

extern "C" void launch_mse_loss_i_i_RScalar(
  StreamCtx stream,
  const RScalar* src_value, 
        RScalar* src_grads, 
  const RScalar* trg_value, 
        RScalar* scratch,
        float*  redux, // scalar
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
    (void*)&trg_value, 
    (void*)&scratch, 
    (void*)&redux, 
    (void*)&m
  };

  CUDA_ASSERT(cudaLaunchCooperativeKernel(
    (void*)(__kernel_mse_loss_i_i_RScalar), grid, block, args, 0, getCtx(stream)
  ));
}
