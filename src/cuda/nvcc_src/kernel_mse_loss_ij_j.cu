#include "../kernel_header.h"

__global__ void __kernel_mse_loss_ij_j_RScalar(
    const RScalar* src_value, 
          RScalar* src_grads, 
    const RScalar* trg_value,
          RScalar* scratch,
          float* redux, // scalar
    len_t m,
    len_t n
) {
    using prc = precision<RScalar>;
  
      auto grid = cg::this_grid();
      
      // find our tile row and column
      len_t t_row = threadIdx.y;
      len_t t_col = threadIdx.x;

      const len_t row_offset = n * ((blockIdx.y * WARP_SIZE) + t_row);
      src_value += row_offset;
      src_grads += row_offset;

      // each row of m is handled by one thread warp (m / 32)
      const len_t m_pos = (blockIdx.y * WARP_SIZE) + t_row;

      //////////////////////////////////
      /// Grid Sum /////////////////////

      prc::type grid_sum = 0.0;

      for (len_t step = 0; step < n; step += WARP_SIZE) {

            if ((m_pos < m) && ((step + t_col) < n)) {    
                  const RScalar x = src_value[t_col];    
                  const RScalar y = trg_value[t_col];    
                  const auto diff = x - y;
                  grid_sum += 0.5f * prc::cast(rsqr(diff));
                  // calculate the dydx
                  if (src_value != src_grads) {
                      src_grads[t_col] = diff;
                  }
            }
            src_value += WARP_SIZE;
            src_grads += WARP_SIZE;
        }

      // no sum to be further calculated
      if (redux == nullptr)
            return;

      grid_sum = blockReduce<AddOP, WARP_SIZE>(
          grid_sum, t_row, t_col
      );

      if ((t_row == 0) && (t_col == 0)) {
          scratch[blockIdx.y] = grid_sum;
      }

      grid.sync();

      grid_sum = 0.0;

      // now we change our thinking to make the block do the reduction
      if ((blockIdx.y == 0) && (m_pos < m)) {

            // flatten the matrix indexing
            const len_t idx = (t_row * WARP_SIZE) + t_col;

            for (len_t step = 0; step < gridDim.y; step += (WARP_SIZE * WARP_SIZE)) {

                  if ((step + idx) < gridDim.y) {
                      grid_sum += prc::cast(scratch[step + idx]);
                  }
            }
      }
      __syncthreads();

      grid_sum = blockReduce<AddOP, WARP_SIZE>(
            grid_sum, t_row, t_col
      );

      if ((blockIdx.y == 0) && (t_row == 0) && (t_col == 0)) {
          const double denom = static_cast<float>(m * n);
          *redux = static_cast<float>(grid_sum) / denom;
      }
} 

extern "C" void launch_mse_loss_ij_j_RScalar(
  StreamCtx stream,
  const RScalar* src_value, 
        RScalar* src_grads, 
  const len_t* trgs,
        RScalar* scratch,
        float* redux, // scalar
  len_t m,
  len_t n
) {
      // TODO: implement block limiting
      dim3 grid (
        1, DIMPAD(m, WARP_SIZE)
      );

      dim3 block (
        WARP_SIZE, 
        WARP_SIZE
      );

      void* args[] = { 
        (void*)&src_value, 
        (void*)&src_grads, 
        (void*)&trgs, 
        (void*)&scratch, 
        (void*)&redux, 
        (void*)&m,
        (void*)&n
      };

      CUDA_ASSERT(cudaLaunchCooperativeKernel(
        (void*)(__kernel_mse_loss_ij_j_RScalar), grid, block, args, 0, getCtx(stream)
      ));
}
