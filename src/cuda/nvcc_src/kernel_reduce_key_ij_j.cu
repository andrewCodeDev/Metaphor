#include "../kernel_header.h"

__global__ void __kernel_reduce_key_ij_j_RScalar(
    const RScalar* src,
          RScalar* dst,
    const unsigned* keys,
          RScalar* scratch,
    unsigned src_col,
    unsigned key_len
){
    auto grid = cg::this_grid();
  
    // track our reduction pairs
    const unsigned lower = blockIdx.y * 2;
    const unsigned upper = lower + 1;

    // TODO: consider moving initial reduction to shared memory

    // reduce all key pairs into 
    for (unsigned key_offset = 0; key_offset < ((key_len + 1) / 2); key_offset += gridDim.y) {
           
        if ((lower + key_offset) < key_len) {
            // the position that we will get from lower index src
            const unsigned lower_src = keys[lower + key_offset] * src_col;

            if ((upper + key_offset) < key_len) {
                // the position that we will get from the upper index src
                const unsigned upper_src = keys[upper + key_offset] * src_col;


                // reduce x + y to blockIdx.y
                for (unsigned offset = 0; offset < src_col; offset += blockDim.x) {
                    // traverse 2 src_column in chunks and sum into scratch
                    if (offset + blockIdx.x < src_col) {
                        scratch[(blockIdx.y * src_col) + threadIdx.x + offset] += 
                          src[lower_src + threadIdx.x + offset] + src[upper_src + threadIdx.x + offset];
                    }
                }
            } else {
                // only reduce x to blockIdx.y
                for (unsigned offset = 0; offset < src_col; offset += blockDim.x) {
                    // traverse 2 src_column in chunks and sum into scratch
                    if (offset + blockIdx.x < src_col) {
                        scratch[(blockIdx.y * src_col) + threadIdx.x + offset] += src[lower_src + threadIdx.x + offset];
                    }
                }
            }
        }
    }

    //////////////////////////////////////////////
    // all of our values are now in scratch memory

    for (unsigned limit = gridDim.y; limit > 1; limit = (limit + 1) / 2) {

        // the position that we will get from lower index src
        for (unsigned offset = 0; offset < src_col; offset += blockDim.x) {

            grid.sync();

            // prevents data race for columns reading and writing
            RScalar __attribute__((unused)) lower_value = RScalar(0.0f); 
            RScalar __attribute__((unused)) upper_value = RScalar(0.0f); 

            if ((lower < limit) && ((offset + threadIdx.x) < src_col)) {                    
                lower_value = scratch[(lower * src_col) + threadIdx.x + offset];
            }
            if ((upper < limit) && ((offset + threadIdx.x) < src_col)) {
                upper_value = scratch[(upper * src_col) + threadIdx.x + offset];
            }
                        
            grid.sync();

            if ((lower < limit) && (offset + threadIdx.x < src_col)) {
                scratch[(blockIdx.y * src_col) + threadIdx.x + offset] = lower_value + upper_value;
            } 
        }
    }
    
    if (blockIdx.y == 0) {
      for (unsigned offset = 0; offset < src_col; offset += blockDim.x) {
          if (offset + threadIdx.x < src_col) {
              dst[threadIdx.x + offset] = scratch[threadIdx.x + offset];
          }
      }
    }
}

extern "C" void launch_reduce_key_ij_j_RScalar(
    StreamCtx stream,
    const RScalar* src,
          RScalar* dst,
    const unsigned* keys,
          RScalar* scratch,
    len_t src_col,
    len_t key_len
) {
    // launch the exact number of blocks required
    const dim3 grid(1, DIMPAD(key_len, 2));
    const dim3 block(std::min(1024ul, src_col));
    const unsigned _src_col = static_cast<unsigned>(src_col);
    const unsigned _key_len = static_cast<unsigned>(key_len);

    void* args[] = { 
      (void*)&src, (void*)&dst, (void*)&keys, (void*)&scratch, (void*)&_src_col, (void*)&_key_len
    };

    CUDA_ASSERT(cudaLaunchCooperativeKernel(
      (void*)(__kernel_reduce_key_ij_j_RScalar), grid, block, args, 0, getCtx(stream)
    ));
}
