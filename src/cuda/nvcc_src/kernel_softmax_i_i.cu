#include "../kernel_header.h"

__global__ void __kernel_softmax_i_i_RScalar(
    const RScalar* A, 
          RScalar* B, 
          RScalar* scratch,
    len_t m
) {
  auto grid = cg::this_grid();

  const len_t idx = grid.thread_rank();

  // since we're coalescing by 4's for every thread in the
  // kernel across the grid, we need to adjust up by 4 times
  // the grid every time we make a shift over to the right.
  const len_t g_step = (WARP_SIZE * WARP_SIZE * gridDim.x) * 4;

  // find the max value of the whole vector
  RScalar grid_max = -Init<RScalar>::infinity();
    
  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = (step + idx) * 4;

    if ((ofs + 4) < m) {
      const auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&A[ofs]);
    
      const RScalar tmp_max = MaxOP::apply(
        u.w, MaxOP::apply(u.x, MaxOP::apply(u.y, u.z))
      );      
      grid_max = MaxOP::apply(grid_max, tmp_max);
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        grid_max = MaxOP::apply(grid_max, A[i]);  
      }
    }
  }

  grid_max = blockReduce<MaxOP, WARP_SIZE>(
    grid_max, threadIdx.y, threadIdx.x
  );

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    scratch[blockIdx.x] = grid_max;
  }

  grid.sync();

  if (idx < WARP_SIZE) { // get only first warp

    for (len_t step = 0; step < gridDim.x; step += (WARP_SIZE * 4)) {

      const len_t ofs = (step + idx) * 4;

      if ((ofs + 4) < gridDim.x) {
        auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&scratch[ofs]);

        const RScalar tmp_max = MaxOP::apply(
          u.w, MaxOP::apply(u.x, MaxOP::apply(u.y, u.z))
        );      
        grid_max = MaxOP::apply(grid_max, tmp_max);
      }
      else if (ofs < gridDim.x) {
        for (len_t i = ofs; i < gridDim.x; ++i) {
          grid_max = MaxOP::apply(grid_max, scratch[i]);  
        }
      }
    }
  }

  grid_max = warpReduce<MaxOP>(grid_max);

  if (idx == 0) {
    scratch[blockIdx.x] = grid_max;
  }

  grid.sync();

  grid_max = scratch[0];

  RScalar grid_sum = RScalar(0.0f);
    
  for (len_t step = 0; step < m; step += g_step) {

    const len_t ofs = (step + idx) * 4;

    if ((ofs + 4) < m) {
      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&A[ofs]);
      auto b4 = reinterpret_cast<coalesce<RScalar>::ptr>(&B[ofs]);
      u.w = rexp(u.w - grid_max);
      u.x = rexp(u.x - grid_max);
      u.y = rexp(u.y - grid_max);
      u.z = rexp(u.z - grid_max);
      grid_sum += u.w + u.x + u.y + u.z;
      *b4 = u;
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        RScalar tmp = rexp(A[i] - grid_max);
        grid_sum += tmp;
        B[i] = tmp;
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
      auto b4 = reinterpret_cast<coalesce<RScalar>::ptr>(&B[ofs]);
      auto u = *b4;
      u.w /= grid_sum;
      u.x /= grid_sum;
      u.y /= grid_sum;
      u.z /= grid_sum;
      *b4 = u;
    }
    else if (ofs < m) {
      for (len_t i = ofs; i < m; ++i) {
        B[i] = B[i] / grid_sum;  
      }
    }
  }
} 

extern "C" void launch_softmax_i_i_RScalar(
  StreamCtx stream,
  const RScalar* A, 
        RScalar* B, 
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
    (void*)&A, (void*)&B, (void*)&scratch, (void*)&m
  };

  CUDA_ASSERT(cudaLaunchCooperativeKernel(
    (void*)(__kernel_softmax_i_i_RScalar), grid, block, args, 0, getCtx(stream)
  ));
}
