const mp = @import("metaphor.zig");
const std = @import("std");
const ops = @import("tensor_ops.zig");
const Parser = @import("expression_parsing.zig");

pub fn copyAndPrintFlat(
    name: []const u8,
    src: anytype,
    dst: anytype,
    stream: anytype
) void {
    
    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\n{s}: {any}\n" , .{ name, dst });
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

    var start: usize = 0;
    for (0..row) |_| {
        std.debug.print("{any}\n", .{ dst[start..start + col]});
        start += col;
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

    ///////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Dims(2){ 10, 10 });  
        defer X1.free();

    const X2 = G.tensor("X1", .wgt, .r32, mp.Dims(2){ 10, 10 });  
        defer X2.free();
    
    mp.mem.sequence(X1, 0.0, 1.0);
    mp.mem.sequence(X2, 0.0, 1.0);

    ///////////////////////////////////////////////////

    const Z1 = mp.ops.hadamard(X1, X1);

    const XT = mp.ops.permutate(X1, "ij->ji");

    const Z2 = mp.ops.hadamard(X1, XT);

    const Z3 = mp.ops.add(Z1, Z2);

    Z3.reverse();
            
    ////////////////////////////////////////////
}
