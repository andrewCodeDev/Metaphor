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

    const row_x: usize = 3;
    const col_x: usize = 4;
    const row_y: usize = 4;
    const col_y: usize = 2;

    const mem_1 = try std.heap.c_allocator.alloc(mp.types.r32, row_x * col_x);
        defer std.heap.c_allocator.free(mem_1);

    const mem_2 = try std.heap.c_allocator.alloc(mp.types.r32, row_y * col_y);
        defer std.heap.c_allocator.free(mem_2);

    const mem_3 = try std.heap.c_allocator.alloc(mp.types.r32, row_x * col_y);
        defer std.heap.c_allocator.free(mem_3);

    /////////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Dims(2){ row_x, col_x });  
        defer X1.free();

    const X2 = G.tensor("X2", .wgt, .r32, mp.Dims(2){ row_y, col_y });  
        defer X2.free();
    
    mp.mem.sequence(X1, 0.0,  1.0);
    mp.mem.sequence(X2, 0.0,  1.0);

    /////////////////////////////////////////////////////

    //const Z1 = mp.ops.hadamard(X1, X1);
    //const XT = mp.ops.permutate(X1, "ij->ji");
    //const Z2 = mp.ops.hadamard(X1, XT);
    //const Z3 = mp.ops.add(Z1, Z2);

    const Z1 = mp.ops.innerProduct(X1, X2, "ij,jk->ik");

    //_ = &Z1;

    copyAndPrintMatrix("X1", X1.values(), mem_1, row_x, col_x, stream);
    copyAndPrintMatrix("X2", X2.values(), mem_2, row_y, col_y, stream);
    copyAndPrintMatrix("Z1", Z1.values(), mem_3, row_x, col_y, stream);

    Z1.reverse();

    copyAndPrintMatrix("dX1", X1.grads().?, mem_1, row_x, col_x, stream);
    copyAndPrintMatrix("dX2", X2.grads().?, mem_2, row_y, col_y, stream);
            
    ////////////////////////////////////////////
}
