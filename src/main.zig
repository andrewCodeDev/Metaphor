const mp = @import("metaphor.zig");
const std = @import("std");

pub fn copyAndPrint(name: []const u8, src: anytype, dst: anytype, stream: anytype) void {
    
    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\n{s}: {any}\n" , .{ name, dst });
}

pub fn main() !void {

    // To start Metaphor, you must initialize the
    // device for the GPU device you want to use

    // Initialize device and cuda context on device zero
    mp.device.init(0);

    // Open the stream you want to compute on.
    // Streams can be run in parallel to launch
    // multiple kernels simultaneous.
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

    const out = try std.heap.c_allocator.alloc(mp.types.r32, 4);
        defer std.heap.c_allocator.free(out);

    /////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Dims(2){ 2, 2 });  
        defer X1.free();
    
    const X2 = G.tensor("X2", .wgt, .r32, mp.Dims(2){ 2, 2 });
        defer X2.free();

    mp.ops.fill(X1, 2);
    mp.ops.fill(X2, 1);

    for (0..1) |_| {

        const Z1 = mp.ops.hadamard(X1, X2);
        const Z2 = mp.ops.hadamard(X1, X1);
        const Z3 = mp.ops.add(Z1, Z2);

        Z3.reverse();
        
        copyAndPrint("X1", X1.grads().?, out, stream);
        copyAndPrint("Z2", X2.grads().?, out, stream);
    }

    ////////////////////////////////////////////
}
