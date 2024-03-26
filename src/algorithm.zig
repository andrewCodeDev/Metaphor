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
const NoArg = CB.NoArg;
const TenOps = @import("tensor_ops.zig");

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
) void {
    const x_sizes = x.sizes();
    std.debug.assert(x_sizes.len == 2);
    std.debug.assert(x_sizes[1] == y.len());
    std.debug.assert(x_sizes[0] >= keys.len);
    std.debug.assert(0 < keys.len);
    const blocks = (keys.len + 1) / 2;
    const scratch = stream.getScratch(Child(@TypeOf(x)), blocks * x_sizes[1]);
    TenOps.fillSlice(Child(@TypeOf(x)), scratch, 0.0, stream);
    overloads.kernel_reduce_key_ij_j.call(.{
        stream.context, x.values().ptr, y.values().ptr, keys.ptr, scratch.ptr, x_sizes[1], keys.len        
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
    comptime expression: [] const u8,
) void {
    if (comptime reduce_key_map.get(expression)) |redux| {
        redux(stream, x, y, keys);                
    } else {
        @compileError("TODO: Declare General Permutation Kernel: " ++ expression);
    }
}
