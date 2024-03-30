#include "../kernel_header.h"

/*
    This is a brutal sorting algorithm. We're caught between two hard
    choices - leave the value vector untouched or suffer massive cache
    incoherence. It may seem counter-intuitive, but we're going to
    coalesce the indices with the values and sort both simultaneously
    so that all memory can remain local to each chunk.

    The index will not be used to access the value, but they'll be
    bussed together to prevent essentially random access patterns.
*/

__global__ void __kernel_merge_sort_key_i_RScalar(
      const SortPair_RScalar* src_pairs, // pairs for reading value and index
            SortPair_RScalar* dst_pairs, // pairs for writing value and index
      unsigned per_thread, // number of elements per thrad
      unsigned n
) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned left = tid * per_thread;

    if (left < n) {

        unsigned right = left + per_thread;

        const unsigned mid = ((right - left) / 2) + left;

        // this check sees if we're in a partition that
        // is the remainder of the vector. In this case,
        // we need to just copy the tail and return.
        if (mid >= n) {

            for (unsigned i = left; i < n; ++i)  
                dst_pairs[i] = src_pairs[i];

        } else {

            // this thread assumes that it has per_thread items to merge. However,
            // we may have a total length that isn't divisible by per_threads, we
            // need to adjust down the right hand boundary to accommodate.
            if (n < right) {
                right = n;
            }

            unsigned l_idx = left;
            unsigned r_idx = mid;
            unsigned w_idx = left;
    
            // Keep going until either (left or right) side is exhausted.
            while (l_idx < mid && r_idx < right) {
                if (dst_pairs[l_idx].val <= dst_pairs[r_idx].val) {
                    dst_pairs[w_idx] = src_pairs[l_idx];
                    ++l_idx;
                    ++w_idx;
                }
                else {
                    dst_pairs[w_idx] = src_pairs[r_idx];
                    ++r_idx;
                    ++w_idx;
                }
            }

            // Write remaining left side
            while (l_idx < mid) {
                dst_pairs[w_idx] = src_pairs[l_idx];
                ++l_idx;
                ++w_idx;
            }

            // Write remaining right side
            while (r_idx < right) {
                dst_pairs[w_idx] = src_pairs[r_idx];
                ++r_idx;
                ++w_idx;
            }
        }
    }
}

template <unsigned CacheSize>
__global__ void __kernel_merge_sort_key_cached_i_RScalar(
  SortPair_RScalar* pairs, // pairs containing value and index
  unsigned per_thread, // number of elements per thrad
  unsigned n // total length of vector
){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned src_idx = tid * per_thread;

    if (src_idx < n) {

        const unsigned mid = per_thread / 2;

        unsigned right = per_thread;

        // this thread assumes that it has per_thread items to merge. However,
        // we may have a total length that isn't divisible by per_threads, we
        // need to adjust down the right hand boundary to accommodate.
        if (n < (src_idx + per_thread)) {
            right = (n - src_idx);
        }

        SortPair_RScalar cache[CacheSize];

        // fill our cache to reduce non-local reads
        for (unsigned i = 0, j = src_idx; i < per_thread; ++i, ++j) {
            cache[i] = pairs[j];
        }

        unsigned l_cache = 0;
        unsigned r_cache = mid;

        // Keep going until either (left or right) side is exhausted.
        while (l_cache < mid && r_cache < right) {
            if (cache[l_cache].val <= cache[r_cache].val) {
                pairs[src_idx] = cache[l_cache];
                ++l_cache;
                ++src_idx;
            }
            else {
                pairs[src_idx] = cache[r_cache];
                ++r_cache;
                ++src_idx;
            }
        }

        // Write remaining left side
        while (l_cache < mid) {
            pairs[src_idx] = cache[l_cache];
            ++l_cache;
            ++src_idx;
        }

        // Write remaining right side
        while (r_cache < right) {
            pairs[src_idx] = cache[r_cache];
            ++r_cache;
            ++src_idx;
        }
    }
}

// Sorts the array in src.
// Returns a pointer to src.
extern "C" void launch_kernel_sort_key_i_RScalar(
  StreamCtx stream,
  SortPair_RScalar* gpu_p1,
  unsigned* per_thread_remaining,
  len_t n
){
    const unsigned total = static_cast<unsigned>(n);

    // we always start off with floor division as
    // the only possible remainders are 0 and 1. In
    // both cases, the remainder is already  sorted
    unsigned merges = total / 2;

     //////////////////////////////////////////////////////
    ///// PHASE 1: Small Buffer Kernel ///////////////////

    const unsigned gpu_threads = 8;

    // these decrease by ceiling divisions of 2
    unsigned thread_blocks = DIMPAD(merges, gpu_threads);

    // these increase in powers of 2
    unsigned per_thread = 2;

    // Try to exploit our cache version and merge
    // until we've hit the hard cache limit (512)
    while (per_thread <= 512) {

        if (DIMPAD(per_thread, total - 1) > 2) {
            merges = 0;
            break;
        }

        __kernel_merge_sort_key_cached_i_RScalar<512>
            <<<thread_blocks, gpu_threads, 0, getCtx(stream)>>>(gpu_p1, per_thread, n);

        per_thread *= 2;

        merges = DIMPAD(total, per_thread);

        thread_blocks = DIMPAD(merges, gpu_threads);
    }

    // only if n >= 2^18 power (262144)
    // this branch never reaches merges == 0
    if (merges > 256) {
        
        // On the host side, we check if we need to expand beyond cache
        // memory. In that case, we've allocated the gpu_p1 to be two
        // times the size of the required value to act as a swap buffer.
        auto gpu_p2 = gpu_p1 + (2 * n);

        // remember our unswapped position
        auto memo = gpu_p1;

        while (merges > 256) {
                        
            __kernel_merge_sort_key_i_RScalar
                <<<thread_blocks, gpu_threads, 0, getCtx(stream)>>>(gpu_p1, gpu_p2, per_thread, n);

            per_thread *= 2;
            merges = DIMPAD(total, per_thread);
            thread_blocks = DIMPAD(merges, gpu_threads);

            std::swap(gpu_p1, gpu_p2);
        }

        // make sure we keep our scratch intact
        if (gpu_p1 != memo) {
            cuMemcpyDtoDAsync(
                reinterpret_cast<CUdeviceptr>(gpu_p1),
                reinterpret_cast<CUdeviceptr>(gpu_p2),
                n, getCtx(stream)
            );            
        }
    }

    ////////////////////////////////////////////////
    // Return back to Zig for potential finalization

    *per_thread_remaining = per_thread;
}
