
pub fn callSortKey(
    stream: Stream,
    src: anytype,  
    keys: []Key,
) void {

    const T = Child(@TypeOf(src));

    // TODO: 
    //  It is possible to do dimensional sort (sorting rows of a matrix, for instance)
    //  but I can't think of a use case at this moment. Currently, sort works on rank-1
    //  tensors and the key vector needs to be the same length.
    std.debug.assert(src.sizes().len == 1);

    // It's worth mentioning that the keys get sorted, not the tensor itself.
    std.debug.assert(src.len() == keys.len);

    if (src.len() < 2) return;

    /////////////////////////////////////////////////////
    // Once we get to merges <= 256, we switch to the
    // cpu to finish the job. The cached merge sort can
    // knock down 9 powers of 2, leaving a remainder
    // for arrays keys.len <= 512.
    //
    // Between the cache array and the cpu, we could have
    // a large number of merges left though. In that case,
    // an auxiliary protocol kicks in and reduces until we
    // get merges <= 256. That secondary protocol requires
    // it's own buffer. To accommodate, we allocate 2*len
    // for sorting pairs and use the second half as the
    // swap buffer. 
    //
    // Any length above 2^18 (262144) will kick off the
    // auxiliary protocol and needs 2*len memory.

    const gpu_pairs = stream.scratch(
        SortPair(T), if (keys.len <= 262144) keys.len else (2 * keys.len)
    );

    overloads.kernel_setup_sort_pairs_i.call(.{
        stream.context, src.values().ptr, gpu_pairs.ptr, keys.len
    });

    var per_thread_remaining: u32 = 0;

    overloads.kernel_sort_key_i.call(.{
        stream.context, gpu_pairs.ptr, &per_thread_remaining, keys.len
    });

    ////////////////////////////////////////////
    // finish our work on the cpu for remainder.
    // remember that the gpu_pairs could be 2x
    // the size of the keys.

    if (per_thread_remaining < keys.len) {        
        mergeRemainderCPU(stream, gpu_pairs[0..keys.len], per_thread_remaining)
            catch @panic("Failed to merge on cpu");
    }

    overloads.kernel_extract_sort_keys_i.call(.{
        stream.context, gpu_pairs.ptr, keys.ptr, keys.len
    });
}


//////////////////////////////////////////////////////////////
///// INTERNAL LIBRARY FUNCTIONS /////////////////////////////

// This function finishes off large merge sorts on the CPU
// because single CPU threads are faster than GPU threads
fn mergeRemainderCPU(
    stream: Stream,
    gpu_pairs: anytype,
    per_thread_remaining: u32,
) !void {
    // T now contains SortPair
    const T = Child(@TypeOf(gpu_pairs));
    
    // continue from remainder of GPU
    const total: u32 = @intCast(gpu_pairs.len);

    var per_thread = per_thread_remaining;

    //////////////////////////////////
    // Hyper parameter for cpu sorting

    const cpu_threads: u32 = 8;

    var threads = std.BoundedArray(std.Thread, cpu_threads).init(0) catch unreachable;

    //////////////////////////////////
    // Setup double length host buffer

    // create host vectors on the cpu...
    var cpu_p1 = try std.heap.c_allocator.alloc(T, 2 * gpu_pairs.len);

    defer {
        cpu_p1.len *= 2; // necessary?
        std.heap.c_allocator.free(cpu_p1);
    }
    // copy over our work from the gpu...
    DU.copyFromDevice(gpu_pairs, cpu_p1[0..gpu_pairs.len], stream);

    // use second half as swap buffer...
    var cpu_p2 = cpu_p1[gpu_pairs.len..];

    // reset length to buffer size
    cpu_p1.len = gpu_pairs.len;

    DU.synchronizeStream(stream);

    //////////////////////////////////
    // Thread-friendly merge sort ///

    const cpu_merge = struct {

        pub fn call_merge(   
           src: []const T, // buffer to read from
           dst: []T, // buffer to write to
           left: u32, // start of subsection
           right_assumed: u32, // assumed boundary
        ) void {

            // compute the original division between the two boundaries    
            const mid = left + ((right_assumed - left) / 2);

            // this check sees if we're in a partition that
            // is the remainder of the vector. In this case,
            // we need to just copy the tail and return.
            if (mid >= src.len) {
                for (left..src.len) |i| dst[i] = src[i];
                return;
            }

            // adjust boundary for non-power of 2 sizing
            const right = if (src.len < right_assumed)
                @as(u32, @intCast(src.len)) else right_assumed; 

            // mobile indices for reading and writing
            var l_head = left;
            var r_head = mid;
            var w_head = left;
        
            while (l_head < mid and r_head < right) {
                if (src[l_head].val <= src[r_head].val) {
                    dst[w_head] = src[l_head];
                    l_head += 1;
                    w_head += 1;
                }
                else {
                    dst[w_head] = src[r_head];
                    r_head += 1;
                    w_head += 1;
                }
            }

            // Write remaining left side
            while (l_head < mid) {
                dst[w_head] = src[l_head];
                l_head += 1;
                w_head += 1;
            }

            // Write remaining right side
            while (r_head < right) {
                dst[w_head] = src[r_head];
                r_head += 1;
                w_head += 1;
            }
        }

    }.call_merge;

    const memo = cpu_p1.ptr;
    
    while (UT.dimpad(per_thread, total - 1) <= 2) : (per_thread *= 2){
        
        var left: u32 = 0;

        while (left < total) : (left += per_thread) {

            threads.appendAssumeCapacity(
                try std.Thread.spawn(.{}, cpu_merge, .{ cpu_p1, cpu_p2, left, (left + per_thread) })
            ); 
            if (threads.len == threads.capacity()) {
                for (threads.slice()) |*t| t.join();
                threads.resize(0) catch unreachable;
            }
        }

        if (threads.len > 0) {
            for (threads.slice()) |*t| t.join();
            threads.resize(0) catch unreachable;
        }

        UT.swap(&cpu_p1, &cpu_p2);
    }

    if (cpu_p1.ptr == memo) {
        // cpu_p1 was swapped and preparing to be the destionation
        DU.copyToDevice(cpu_p1, gpu_pairs, stream);
    } else {
        // cpu_p1 wrote its values into cpu_p2 as the destination
        DU.copyToDevice(cpu_p2, gpu_pairs, stream);
    }
}

fn SortPair(comptime T: type) type {
    return switch(T) {
        SC.r16 => C.SortPair_r16,
        SC.r32 => C.SortPair_r32,
        SC.r64 => C.SortPair_r64,
    };
}

