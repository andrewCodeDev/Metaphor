//////////////////////////////////////
/// Algorithm intefaces //////////////

// Algorithms differ from operations as
// they do not effect the gradient or
// provide a reversal function.

const std = @import("std");
const math = std.math;
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;

// key types are used for selective indexing
// they can choose columns, rows, or represent
// elements in a sorting algorithm
// pub const Key = u32;

// <>--------------------------------------------------------<>

pub fn fill(x: Tensor, value: f64) void {
    core.kernels.fill[core.dkey(x)](x.data_ptr(), value, x.len(), x.stream());
}

// <>--------------------------------------------------------<>

pub fn sequence(x: Tensor, init: f64, step: f64) void {    
    core.kernels.sequence[core.dkey(x)](x.data_ptr(), init, step, x.len(), x.stream());
}

// <>--------------------------------------------------------<>

pub fn copy(src: Tensor, dst: Tensor) void {
    std.debug.assert(src.dtype() == dst.dtype());
    core.kernels.copy[core.dkey(dst)](src.data_ptr(), dst.data_ptr(), dst.stream());
}

// <>--------------------------------------------------------<>

//const RandomizeMode = enum { gauss, uniform };
//
//pub fn randomizeSlice(
//    comptime T: type,
//     dst: []T,
//     mode: RandomizeMode,
//     stream: Stream,
//) void {
//
//    //TODO: replace this with a kernel call...?
//    //      really though, how often is this called?
//    var backing = std.rand.DefaultPrng.init(22);
//    const random = backing.random();
//
//    const mem = std.heap.c_allocator.alloc(T, dst.len) catch @panic("randomize out of memory");
//    defer std.heap.c_allocator.free(mem);
//
//    switch (mode) {
//        .uniform, => {
//            for (0..dst.len) |i| mem[i] = 2.0 * SC.asScalar(T, random.float(f32)) - 1.0; 
//        },
//        .gauss => { 
//            for (0..dst.len) |i| mem[i] = SC.asScalar(T, random.floatNorm(f32)); 
//        }
//    }
//
//    DU.copyToDevice(mem, dst, stream);
//}
//
//pub fn randomize(dst: anytype, mode: RandomizeMode) void {
//    randomizeSlice(Child(@TypeOf(dst)), dst.values(), mode, dst.stream());
//}

// <> -------------------------------------------------------------- <>

//pub fn reduceKey_ij_j(
//    stream: Stream,
//    x: anytype,  
//    y: anytype,
//    keys: []const Key,
//    alpha: f32,
//) void {
//    const T = Child(@TypeOf(x));
//    const x_sizes = x.sizes();
//    std.debug.assert(x_sizes.len == 2);
//    std.debug.assert(x_sizes[1] == y.len());
//    std.debug.assert(x_sizes[0] >= keys.len);
//    std.debug.assert(0 < keys.len);
//
//    // rows are reduced two at a time, so we only need approximately
//    // half the number of blocks as we have keys to reduce rows
//    const blocks = (keys.len + 1) / 2;
//
//    // each block has it's own row in scratch memory the size of the column
//    const scratch = stream.getScratch(T, blocks * x_sizes[1]);
//
//    // TODO: consider making this unnecessary - would need kernel update
//    fillSlice(T, scratch, 0.0, stream);
//
//    overloads.kernel_reduce_key_ij_j.call(.{
//        stream.context, x.values().ptr, y.values().ptr, keys.ptr, SC.asScalar(T, alpha), scratch.ptr, x_sizes[1], keys.len        
//    });
//}
//
//const reduce_key_map = std.ComptimeStringMap(@TypeOf(reduceKey_ij_j), .{
//    .{ "ij->j", reduceKey_ij_j },
//});
//
//pub fn callReduceKey(
//    stream: Stream,
//    x: anytype,  
//    y: anytype,
//    keys: []const Key,
//    alpha: f32,
//    comptime expression: [] const u8,
//) void {
//    if (comptime reduce_key_map.get(expression)) |redux| {
//        redux(stream, x, y, keys, alpha);                
//    } else {
//        @compileError("Unknown reduce key expression: " ++ expression);
//    }
//}
//
//// <>--------------------------------------------------------<>
//
//pub fn maxKey_ij_j(
//    stream: Stream,
//    x: anytype,  
//    keys: []Key,
//) void {
//    const x_sizes = x.sizes();
//    std.debug.assert(x_sizes.len == 2);
//    std.debug.assert(x_sizes[0] == keys.len);
//    std.debug.assert(0 < keys.len);
//
//    overloads.kernel_max_key_ij_j.call(.{
//        stream.context, x.values().ptr, keys.ptr, x_sizes[0], x_sizes[1]
//    });
//}
//
//const max_key_map = std.ComptimeStringMap(@TypeOf(maxKey_ij_j), .{
//    .{ "ij->j", maxKey_ij_j },
//});
//
//pub fn callMaxKey(
//    stream: Stream,
//    src: anytype,  
//    keys: []Key,
//    comptime expression: [] const u8,
//) void {
//    if (comptime max_key_map.get(expression)) |max| {
//        max(stream, src, keys);                
//    } else {
//        @compileError("Unknown max key expression: " ++ expression);
//    }
//}
// <>--------------------------------------------------------<>
//
