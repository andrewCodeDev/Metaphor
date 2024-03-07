const mp = @import("metaphor.zig");
const std = @import("std");
const ops = @import("tensor_ops.zig");
const Parser = @import("expression_parsing.zig");
const DU = @import("device_utils.zig");

pub fn copyAndPrintFlat(
    name: []const u8,
    src: anytype,
    dst: anytype,
    stream: anytype
) void {
    
    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\n{s}: {any} " , .{ name, dst });
}

pub fn cpu_print_matrix(
    name: []const u8, 
    dst: anytype, 
    row: usize,
    col: usize,
) void {    
    std.debug.assert(dst.len == row * col);
    
    std.debug.print("\nName: {s}:\n", .{ name });

    for (0..row) |i| {
        for (0..col) |j| {
            std.debug.print("{d:.1} ", .{ dst[i * col + j]});
        }
        std.debug.print("\n", .{});
    }
}

pub fn randomize(
    x: anytype,
    stream: DU.Stream
) void {
    // testing function - should use cuda support in the future.
    // perhaps thrust namespace functions?
    
    var backing = std.rand.DefaultPrng.init(42);

    var random = backing.random();
    
    const mem = std.heap.c_allocator.alloc(@TypeOf(x).DataType, x.len())
        catch @panic("randomize out of memory");

        defer std.heap.c_allocator.free(mem);

    for (0..x.len()) |i|
        mem[i] = random.float(@TypeOf(x).DataType);

    DU.copyToDevice(mem, x.values(), stream);

    DU.synchronizeStream(stream);
}

pub fn copyAndPrintMatrix(
    name: []const u8, 
    src: anytype, 
    dst: anytype, 
    row: usize,
    col: usize,
    stream: anytype
) void {    
    std.debug.assert(src.len == dst.len);
    std.debug.assert(src.len == row * col);
    
    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\nName: {s}:\n", .{ name });

    for (0..row) |i| {
        for (0..col) |j| {
            std.debug.print("{d:.1} ", .{ dst[i * col + j] });
        }
        std.debug.print("\n", .{});
    }
}


pub fn cpu_matmul_2D(
    x: anytype,
    y: @TypeOf(x),
    z: @TypeOf(x),
    M: usize,
    N: usize,
    K: usize
) void {
    std.debug.assert(x.len == M * N);
    std.debug.assert(y.len == N * K);
    std.debug.assert(z.len == M * K);

    for (0..M) |m| {

        for (0..K) |k| {

            var tmp: f32 = 0;

            for (0..N) |n| {
                tmp += x[m * N + n] * y[n * K + k];
            }
            z[m * K + k] = tmp;
        }
    }
}

pub fn verify_restuls(
    name: []const u8,
    x: anytype,
    y: @TypeOf(x),
) void {
    std.debug.assert(x.len == y.len);

    var good: bool = true;

    for (x, y) |a, b| {
        if (@abs(a - b) > 0.0001) {
            std.log.err("Verification FAILURE: {s}", .{name});
            good = false;
            break;
        }
    }
    if (good) {    
        std.log.info("Verification SUCCESS: {s}", .{name});
    }
}

pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .auto_free_wgt_grads = false,
        .auto_free_inp_grads = false,
        .auto_free_hid_nodes = false,
        .stream = stream,
    });

    defer G.deinit();

    const row_x: usize = 32;
    const col_x: usize = 32;

    // cpu side memory for verifying results
    const x1 = try std.heap.c_allocator.alloc(mp.types.r32, row_x);
        defer std.heap.c_allocator.free(x1);

    const x2 = try std.heap.c_allocator.alloc(mp.types.r32, col_x);
        defer std.heap.c_allocator.free(x2);

    const z1 = try std.heap.c_allocator.alloc(mp.types.r32, row_x * col_x);
        defer std.heap.c_allocator.free(z1);

    const z1_verify = try std.heap.c_allocator.alloc(mp.types.r32, row_x * col_x);
        defer std.heap.c_allocator.free(z1_verify);

    /////////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Dims(1){ row_x });  
        defer X1.free();

    const X2 = G.tensor("X2", .wgt, .r32, mp.Dims(1){ col_x });  
        defer X2.free();

    const Z1 = G.tensor("Z1", .wgt, .r32, mp.Dims(2){ row_x, col_x });  
        defer Z1.free();
    
    //randomize(X1, stream);
    //randomize(X2, stream);
    //mp.mem.sequence(X1, 0.0, 1.0);
    mp.mem.fill(X1, 1.0);
    mp.mem.sequence(X2, 0.0, 1.0);
    //mp.mem.fill(X2, 1.0);

    /////////////////////////////////////////////////////

    ops.outerProduct_i_j(stream, X1, X2, Z1);

    //mp.mem.copyFromDevice(X1.values(), x1, stream);
    //mp.mem.copyFromDevice(X2.values(), x2, stream);
    //mp.mem.copyFromDevice(Z1.values(), z1, stream);

    mp.stream.synchronize(stream);

    //cpu_matmul_2D(x1, x2, z1_verify, 1, row_x, col_x);
    //cpu_print_matrix("z1_verify", z1_verify, 1, col_x);

    copyAndPrintMatrix("Z1", Z1.values(), z1, row_x, col_x, stream);

    //verify_restuls("i,ij->j", z1, z1_verify);
    //_ = &Z1;

    //copyAndPrintMatrix("X1", X1.values(), x1, row_x, col_x, stream);
    //copyAndPrintMatrix("X2", X2.values(), x2, row_y, col_y, stream);

    //copyAndPrintMatrix("dX1", X1.grads().?, x1, row_x, col_x, stream);
    //copyAndPrintMatrix("dX2", X2.grads().?, x2, row_y, col_y, stream);
            
    ////////////////////////////////////////////
}
