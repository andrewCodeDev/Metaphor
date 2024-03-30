//////////////////////////////////////
/// Algorithm intefaces //////////////

// Algorithms differ from operations as
// they do not effect the gradient or
// provide a reversal function.

const std = @import("std");
const math = std.math;
const SC = @import("scalar.zig");
const UT = @import("utility.zig");
const CB = @import("callback_builder.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");
const Child = UT.Child;

const overloads = @import("kernel_overloads.zig");
const Parser = @import("expression_parsing.zig");
const Stream = DU.Stream;
const TenOps = @import("tensor_ops.zig");
const C = @import("cimport.zig").C;

// key types are used for selective indexing
// they can choose columns, rows, or represent
// elements in a sorting algorithm
pub const Key = u32;


// TODO: make a dispatch map and create column-key reduce

pub fn reduceKey_ij_j(
    stream: Stream,
    x: anytype,  
    y: anytype,
    keys: []const Key,
    alpha: f32,
) void {
    const T = Child(@TypeOf(x));
    const x_sizes = x.sizes();
    std.debug.assert(x_sizes.len == 2);
    std.debug.assert(x_sizes[1] == y.len());
    std.debug.assert(x_sizes[0] >= keys.len);
    std.debug.assert(0 < keys.len);

    // rows are reduced two at a time, so we only need approximately
    // half the number of blocks as we have keys to reduce rows
    const blocks = (keys.len + 1) / 2;

    // each block has it's own row in scratch memory the size of the column
    const scratch = stream.getScratch(T, blocks * x_sizes[1]);

    // TODO: consider making this unnecessary - would need kernel update
    TenOps.fillSlice(T, scratch, 0.0, stream);

    overloads.kernel_reduce_key_ij_j.call(.{
        stream.context, x.values().ptr, y.values().ptr, keys.ptr, SC.asScalar(T, alpha), scratch.ptr, x_sizes[1], keys.len        
    });
}

const reduce_key_map = std.ComptimeStringMap(@TypeOf(reduceKey_ij_j), .{
    .{ "ij->j", reduceKey_ij_j },
});

pub fn callReduceKey(
    stream: Stream,
    x: anytype,  
    y: anytype,
    keys: []const Key,
    alpha: f32,
    comptime expression: [] const u8,
) void {
    if (comptime reduce_key_map.get(expression)) |redux| {
        redux(stream, x, y, keys, alpha);                
    } else {
        @compileError("TODO: Declare General Permutation Kernel: " ++ expression);
    }
}

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

    const gpu_pairs = stream.getScratch(
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

// <>--------------------------------------------------------<>

pub fn sequence(tensor: anytype, init: anytype, step: anytype) void {
    const T = UT.Child(@TypeOf(tensor));
    const _init = SC.asScalar(T, init);
    const _step = SC.asScalar(T, step);
    const values = tensor.values();

    overloads.kernel_sequence.call(.{ tensor.ptr.stream.context, values.ptr, _init, _step, values.len });
}

// <>--------------------------------------------------------<>

pub fn randomize(x: anytype) void {
    //TODO: replace this with a kernel call...?
    //      really though, how often is this called?
    var backing = std.rand.DefaultPrng.init(22);
    var random = backing.random();

    const mem = std.heap.c_allocator.alloc(@TypeOf(x).DataType, x.len()) catch @panic("randomize out of memory");
    defer std.heap.c_allocator.free(mem);

    for (0..x.len()) |i| {
        mem[i] = random.float(@TypeOf(x).DataType);
    }

    DU.copyToDevice(mem, x.values(), x.stream());
    DU.synchronizeStream(x.stream());
}

// <>--------------------------------------------------------<>

pub fn fillSlice(
    comptime T: type,
    x_slice: []T,
    value: anytype,
    stream: Stream,
) void {
    overloads.kernel_fill.call(.{ stream.context, x_slice.ptr, SC.asScalar(T, value), x_slice.len });
    // TODO:
    //   consider removing this synchronize call - the graph requires
    //   that the gradient exists before reversing and that's where this
    //   synchronize call is useful for. Consider different approach?
    DU.synchronizeStream(stream);
}

pub fn fill(x: anytype, value: anytype) void {
    fillSlice(Child(@TypeOf(x)), x.values(), value, x.stream());
}

// <>--------------------------------------------------------<>

pub fn copySlice(
    comptime T: type,
    src_slice: []T,
    dst_slice: []T,
    stream: Stream,
) void {
    std.debug.assert(src_slice.len == dst_slice.len);
    overloads.kernel_copy.call(.{ stream.context, src_slice.ptr, dst_slice.ptr, src_slice.len });
}

pub fn copy(src: anytype, dst: anytype) void {
    copySlice(Child(@TypeOf(src)), src.values(), dst.values(), dst.stream());
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

    std.log.info("cpu_p1.len: {}", .{cpu_p1.len});

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
                std.log.info("Copying from: {}", .{ left });
                for (left..src.len) |i| dst[i] = src[i];
                return;
            }

            // adjust boundary for non-power of 2 sizing
            const right = if (src.len < right_assumed)
                @as(u32, @intCast(src.len)) else right_assumed; 

            std.log.info("left: {}, mid: {}, right: {}", .{ left, mid, right });

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
    
    while (UT.dimpad(per_thread, total) <= 2) : (per_thread *= 2){
        
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
        else => @compileError("Invalid type for sorting pair")
    };
}
