const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// TODO: actually explain the basics here


pub fn main() !void {
    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .eval });
    defer G.deinit();

    const M: usize = 32;
    const N: usize = 16;

    /////////////////////////////////////////////////////
    // feed forward network...

    const u = G.tensor(.wgt, .r32, mp.Rank(2){ M, N });
    const v = G.tensor(.wgt, .r32, mp.Rank(2){ M, N });

    //mp.mem.sequence(x, 0.0, 1.0);
    //mp.mem.sequence(u, 0.0, 1.0);
    //mp.mem.sequence(v, 0.0, 1.0);

    //const z1 = mp.ops.add(x,  u);
    //const z2 = mp.ops.add(z1, v);

    //_ = &z2;

    //G.save("data", "G");

    G.load("data", "G");

    try EU.copyAndPrintMatrix("u", u.values(), M, N, stream);
    try EU.copyAndPrintMatrix("v", v.values(), M, N, stream);
    
    //try mp.readTensor(x, std.heap.c_allocator, "data", stream);

//    const trgs_cpu = try EU.allocCPU(mp.types.SizeType, M);
//    defer EU.freeCPU(trgs_cpu);
//
//    var t: usize = 0;
//
//    for (0..M) |i| {
//        trgs_cpu[i] = t;
//        // wrap targets around
//        t = if (i == (N - 1)) 0 else t + 1;
//    }
//
//    const trgs = G.tensor_allocator.allocTensor(mp.types.SizeType, M, stream);
//    defer G.tensor_allocator.freeTensor(trgs, stream);
//
//    mp.mem.copyToDevice(trgs_cpu, trgs, stream);
//
//    const x = G.tensor(.inp, .r32, mp.Rank(2){M, N});
//
//    mp.mem.sequence(x, 0.0, 0.1);
//
//    mp.loss.cce(x, trgs, .{
//        .grads = true,
//        .score = null,
//    });
//
//    try EU.copyAndPrintMatrix("x value:", x.values(), M, N, stream);
//    try EU.copyAndPrintMatrix("x grads:", x.grads().?, M, N, stream);

    ////////////////////////////////////////////
}
