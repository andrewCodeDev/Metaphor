#include "../kernel_header.h"

__global__ void __kernel_cce_loss_ij_j_RScalar(
    const RScalar* src_value, 
          RScalar* src_grads, 
    const len_t* trg,
          double* redux, // scalar
    len_t m
) {
//  __shared__ RScalar redux[WARP_SIZE];
//
//  // find our tile row and column
//  len_t t_row = threadIdx.y;
//  len_t t_col = threadIdx.x;
//
//  A += (blockIdx.y * WARP_SIZE) + t_row;
//  B += (blockIdx.y * WARP_SIZE) + t_row;
//  const len_t m_bound = (blockIdx.y * WARP_SIZE) + t_row;
//
//  // these are mobile pointers
//  auto A_tmp = A;
//  auto B_tmp = B;
//
//  //////////////////////////////////
//  /// Grid Max /////////////////////
//
//  RScalar grid_max = -Init<RScalar>::infinity();
//    
//  for (len_t step = 0; step < n; step += WARP_SIZE) {
//
//    if ((m_bound < m) && (step + t_row) < n) {    
//      grid_max = MaxOP::apply(grid_max, A_tmp[t_col]);
//    }
//    A_tmp += WARP_SIZE;
//  }
//
//  grid_max = warpReduce<MaxOP>(grid_max);
//
//  if (m_bound < m && t_col == 0) {
//    redux[t_row] = grid_max;
//  }
//
//  __syncthreads();
//
//  if (m_bound < m) {
//    grid_max = redux[t_row];
//  }
//
//  //////////////////////////////////
//  /// Grid Sum /////////////////////
//
//  RScalar grid_sum = RScalar(0.0f);
//
//  A_tmp = A;
//    
//  for (len_t step = 0; step < n; step += WARP_SIZE) {
//
//    if ((m_bound < m) && (step + t_row) < n) {    
//      const RScalar u = rexp(A_tmp[t_col] - grid_max);
//      B_tmp[t_col] = u;
//      grid_sum += u;
//    }
//    A_tmp += WARP_SIZE;
//    B_tmp += WARP_SIZE;
//  }
//
//  grid_sum = warpReduce<AddOP>(grid_sum);
//
//  if (m_bound < m && t_col == 0) {
//    redux[t_row] = grid_sum;
//  }
//
//  __syncthreads();
//
//  if (m_bound < m) {
//    grid_sum = redux[t_row];
//  }
//  
//  for (len_t step = 0; step < n; step += WARP_SIZE) {
//
//    if ((m_bound < m) && (step + t_row) < n) {    
//      B[t_col] /= grid_sum;
//    }
//    // move A along the columns
//    B += WARP_SIZE;
//  }
//
//
//
//  
//      auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&src_value[ofs]);
//      // calculate the loss
//      const RScalar w = (ofs + 0 == trg) ? -rlog(MaxOP::apply(u.w, Init<RScalar>::epsilon())) : RScalar(0.0f);
//      const RScalar x = (ofs + 1 == trg) ? -rlog(MaxOP::apply(u.x, Init<RScalar>::epsilon())) : RScalar(0.0f);
//      const RScalar y = (ofs + 2 == trg) ? -rlog(MaxOP::apply(u.y, Init<RScalar>::epsilon())) : RScalar(0.0f);
//      const RScalar z = (ofs + 3 == trg) ? -rlog(MaxOP::apply(u.z, Init<RScalar>::epsilon())) : RScalar(0.0f);
//      grid_sum += static_cast<double>(w + x + y + z);
//      // calculate the dydx
//      if (src_value != src_grads) {
//        auto s = reinterpret_cast<coalesce<RScalar>::ptr>(&src_grads[ofs]);
//        u.w = u.w - RScalar((ofs + 0 == trg) ? 1.0f : 0.0f);
//        u.x = u.x - RScalar((ofs + 1 == trg) ? 1.0f : 0.0f);
//        u.y = u.y - RScalar((ofs + 2 == trg) ? 1.0f : 0.0f);
//        u.z = u.z - RScalar((ofs + 3 == trg) ? 1.0f : 0.0f);
//        *s = u;
//      }
//    }
//    else if (ofs < m) {
//      for (len_t i = ofs; i < m; ++i) {
//        const RScalar x = src_value[i];
//        const RScalar y = (i == trg) ? -rlog(MaxOP::apply(x, Init<RScalar>::epsilon())) : RScalar(0.0f);
//        grid_sum += y;
//        // calculate the dydx
//        if (src_value != src_grads) {
//          src_grads[i] = x - RScalar((i == trg) ? 1.0f : 0.0f);
//        }
//      }
//    }
//  }
//
//  // continue calculating sum
//  if (redux != nullptr)
//  {
//    grid_sum = blockReduce<AddOP, WARP_SIZE>(
//      grid_sum, threadIdx.y, threadIdx.x
//    );
//
//    if (threadIdx.x == 0 && threadIdx.y == 0) {
//      scratch[blockIdx.x] = grid_sum;
//    }
//
//    grid.sync();
//
//    grid_sum = RScalar(0.0f);
//
//    if (idx < WARP_SIZE) { // get only first warp
//
//      for (len_t step = 0; step < gridDim.x; step += (WARP_SIZE * 4)) {
//
//        const len_t ofs = (step + idx) * 4;
//
//        if ((ofs + 4) < gridDim.x) {
//          auto u = *reinterpret_cast<coalesce<RScalar>::c_ptr>(&scratch[ofs]);
//          grid_sum += u.w + u.x + u.y + u.z;
//        }
//        else if (ofs < gridDim.x) {
//          for (len_t i = ofs; i < gridDim.x; ++i) {
//            grid_sum += scratch[i];  
//          }
//        }
//      }
//    }
//
//    grid_sum = warpReduce<AddOP>(grid_sum);
//    
//    if (idx == 0) {
//      *redux = grid_sum;
//    }
//  }
} 
extern "C" void launch_cce_loss_ij_j_RScalar(
  StreamCtx stream,
  const RScalar* src_value, 
        RScalar* src_grads, 
  const len_t* trg,
        double* redux, // scalar
  len_t m,
  len_t n
) {

  //dim3 grid (
  //  1, DIMPAD(n, WARP_SIZE)
  //);

  //dim3 block (
  //  WARP_SIZE, 
  //  WARP_SIZE
  //);

  //__kernel_cce_loss_ij_j_RScalar<<<grid, block, 0, getCtx(stream)>>>(
  //  src_value, src_grads, trg, m, n
  //);
}
