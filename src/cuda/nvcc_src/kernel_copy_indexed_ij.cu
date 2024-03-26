#include "../kernel_header.h"

__global__ void __kernel_copy_indexed_ij_kj_RScalar(
    const RScalar* src,
          RScalar* dst,
    const len_t* idxs,
    uint src_row,
    uint src_col,
    uint out_row
){
    // we have less indices than the out_row, so we need to advance through the
    // out rows, dedicating each block in our grid to copying one row at a time
    for (uint idx_offset = 0; idx_offset < out_row; idx_offset += gridDim.y) {

        // since each block deals with a single row, we have to be in bounds
        if (idx_offset + blockIdx.y >= out_row)
            break;

        // TODO: place checks for row values outside of src_row boundary

        // get the index specified by the next value in the idxs array
        const uint src_offset = static_cast<uint>(idxs[idx_offset + blockIdx.y] * src_col);
        const uint dst_offset = blockIdx.y * src_col;

        // advance through the targeted column and copy elementwise
        for (uint col_offset = 0; col_offset < src_col; col_offset += blockDim.x)
        {
            if (col_offset + threadIdx.x < src_col)
                dst[dst_offset + threadIdx.x] = src[src_offset + threadIdx.x];
        }
    }
}

extern "C" void launch_copy_indexed_ij_kj_RScalar(
    StreamCtx stream,
    const RScalar* src,
          RScalar* dst,
    const len_t* idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
) {
    // TODO: search for better hyper-parameter
    constexpr len_t MAX_Y_BLOCKS = 50;
  
    const dim3 grid_block(
        1, std::min(MAX_Y_BLOCKS, DIMPAD(out_row, 2))
    );

    // flat indexing per block
    const dim3 thread_block(std::min(1024ul, src_col));

    __kernel_copy_indexed_ij_kj_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(
        src, dst, idxs, 
        static_cast<uint>(src_row), 
        static_cast<uint>(src_col), 
        static_cast<uint>(out_row)
    );
}

__global__ void __kernel_copy_indexed_ij_kj_buffered_RScalar(
    const RScalar* src,
          RScalar* dst,
    UintBuffer idxs,
    uint src_row,
    uint src_col,
    uint out_row
){
    // TODO: place checks for row values outside of src_row boundary

    // get the index specified by the next value in the idxs array
    const uint row_offset = idxs.items[blockIdx.y] * src_col;
    const uint out_offset = blockIdx.y * src_col;

    // advance through the targeted column and copy elementwise
    for (uint col_offset = 0; col_offset < src_col; col_offset += blockDim.x)
    {
        if (col_offset + threadIdx.x < src_col)
            dst[out_offset + threadIdx.x] = src[row_offset + threadIdx.x];
    }
}

extern "C" void launch_copy_indexed_ij_kj_buffered_RScalar(
    StreamCtx stream,
    const RScalar* src,
          RScalar* dst,
    UintBuffer idxs,
    len_t src_row,
    len_t src_col,
    len_t out_row
) {
    // launch the exact number of blocks required
    const dim3 grid_block(1, idxs.used);

    // flat indexing per block
    const dim3 thread_block(std::min(1024ul, src_col));

    __kernel_copy_indexed_ij_kj_buffered_RScalar<<<grid_block, thread_block, 0, getCtx(stream)>>>(
        src, dst, idxs, 
        static_cast<uint>(src_row), 
        static_cast<uint>(src_col), 
        static_cast<uint>(out_row)
    );
}
