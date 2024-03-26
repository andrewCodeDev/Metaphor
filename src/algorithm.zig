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

// key types are used for selective indexing
// they can choose columns, rows, or represent
// elements in a sorting algorithm
pub const Key = u32;

// helper function to create populated key vector
pub fn allocKeys(n: usize, stream: Stream) []Key {

    std.debug.assert(n < std.math.maxInt(Key));
    
    const keys_cpu = std.heap.c_allocator.alloc(Key, n) catch @panic("Failed to allocate key buffer");
    defer std.heap.c_allocator.free(keys_cpu);

    var key: Key = 0;
    for (0..n) |i| {
        keys_cpu[i] = key;
        key += 1;
    }

    const keys = DU.alloc(Key, n, stream);

    DU.copyToDevice(keys_cpu, keys, stream);

    return keys;
}

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
    x: anytype,  
    keys: []Key,
) void {

    // TODO: 
    //  It is possible to do dimensional sort (sorting rows of a matrix, for instance)
    //  but I can't think of a use case at this moment. Currently, sort works on rank-1
    //  tensors and the key vector needs to be the same length.
    std.debug.assert(x.sizes().len == 1);

    // It's worth mentioning that the keys get sorted, not the tensor itself.
    std.debug.assert(x.len() == keys.len);

    if (x.len() < 2) return;
    
    _ = &stream;

}
