const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// assumes n % 2 == 0
//pub fn encode(x: anytype) void {
//    const T = mp.scalar.Child(@TypeOf(x));
//
//    const tmp = std.heap.c_allocator.alloc(f32, x.len()) catch unreachable;
//        defer std.heap.c_allocator.free(tmp);
//    
//    const out = std.heap.c_allocator.alloc(T, x.len()) catch unreachable;
//        defer std.heap.c_allocator.free(out);
//
//    const m = x.sizes()[0];
//    const n = x.sizes()[1];
//    const _n: f32 = @floatFromInt(n);
//
//    for (0..m) |i| {
//        var j: usize = 0;
//        while (j < n) : (j += 2) {
//            const _i: f32 = mp.scalar.as(f32, i);            
//            const _j: f32 = mp.scalar.as(f32, j);            
//            const theta = _i / std.math.pow(f32, 100.0, _j / _n);
//            tmp[i * n + j + 0] = std.math.sin(theta);
//            tmp[i * n + j + 1] = std.math.cos(theta);
//        }
//    }
//
//    for (0..x.len()) |i| {
//        out[i] = mp.scalar.as(T, tmp[i]);
//    }
//
//    mp.mem.copyToDevice(out, x.values(), x.stream());
//}


pub fn main() !void {
    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    var sgd = mp.optm.SGD.init(.{ 
        .rate = 0.1, 
        .clip = .{
            .lower = -2.0,
            .upper =  2.0,
        } 
    });

    _ = &sgd;

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
    defer G.deinit();

    const m: usize = 16;
    const n: usize = 32;

    const X1 = G.tensor(.wgt, .r32, mp.Rank(2){ m, n });
    //const X2 = G.tensor(.inp, .r32, mp.Rank(2){ m, n });

    //const Q = G.tensor(.wgt, .r32, mp.Rank(2){ m, n });
    //const K = G.tensor(.wgt, .r32, mp.Rank(2){ m, n });
    //const V = G.tensor(.wgt, .r32, mp.Rank(2){ m, n });
    //const alpha: f32 = 1.0 / @as(f32, @floatFromInt(n));

    ///////////////////////////////////////////////////
    // feed forward network...

    // project to squared dimensions...

    mp.mem.sequence(X1, 0.0, 0.1);

    Y.reverse(.keep);

    try EU.copyAndPrintMatrix("Y Values", Y.values(),   m, n, stream);
    try EU.copyAndPrintMatrix("X1 grads", X1.grads().?, m, n, stream);
    
    //mp.mem.fill(X2, 1.0);
    //mp.mem.fill(Q, 1.0);
    //mp.mem.fill(K, 1.0);

    //const QX = mp.ops.innerProduct(Q,  X1, "ij,jk->ik");
    //const KX = mp.ops.innerProduct(K,  X2, "ij,jk->ik");
    //const VX = mp.ops.innerProduct(V,  X1, "ij,jk->ik");
    //const QK = mp.ops.innerProduct(QX, KX, "ij,kj->ik");
    //const SM = mp.ops.softmax(QK, "ij|j");
    //const SV = mp.ops.innerProductScaled(SM, VX, alpha, "ij,jk->ik");

    //std.log.info("QX sizes: {any}", .{ QX.sizes() });
    //std.log.info("KX sizes: {any}", .{ KX.sizes() });
    //std.log.info("QK sizes: {any}", .{ QK.sizes() });

    //QK.reverse(.keep);

    //try EU.copyAndPrintMatrix("QX Values", QX.values(), m, m, stream);
    //try EU.copyAndPrintMatrix("KX Values", KX.values(), m, m, stream);

    //try EU.copyAndPrintMatrix("QX Grads", QX.grads().?, m, m, stream);
    //try EU.copyAndPrintMatrix("KX Grads", KX.grads().?, m, m, stream);
    

    //var net = NeuralNet(.r32, 3).init(G, m, n, true);
    //const x = G.tensor(.inp, .r32, mp.Rank(1){n});
    //const t = 16;

    //mp.mem.randomize(x, .gauss);
    //net.randomize();

    //var score: f64 = 0.0;

    //for (0..100) |_| {
    //    const y = net.forward(x);

    //    mp.loss.cce(y, t, .{
    //        .grads = true,
    //        .score = &score,
    //    });

    //    net.reverse();

    //    G.reset(.node, .all);
    //    G.reset(.leaf, .grd);

    //    std.log.info("score: {d:.4}", .{score});
    //}

    ////////////////////////////////////////////
    mp.device.check();

    std.log.info("Experimental: SUCCESS", .{});
}
