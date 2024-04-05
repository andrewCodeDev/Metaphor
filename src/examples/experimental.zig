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

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
    defer G.deinit();

    var sgd = mp.optm.SGD.init(.{ 
        .rate = 0.1, 
        .clip = .{
            .lower = -2.0,
            .upper =  2.0,
        } 
    });

    _ = &sgd;

    const m: usize = 32;
    const n: usize = 32;
    //const k: usize = 64;

    const X1 = G.tensor(.inp, .r32, mp.Rank(2){ m, n });
    const X2 = G.tensor(.inp, .r32, mp.Rank(2){ m, n });

    const t = G.tensor(.inp, .r32, mp.Rank(2){ m, 1 });
    
    const Q = G.tensor(.wgt, .r32, mp.Rank(2){ n, m });
    const K = G.tensor(.wgt, .r32, mp.Rank(2){ n, m });
    const V = G.tensor(.wgt, .r32, mp.Rank(2){ n, m });
    const alpha: f32 = 1.0 / @as(f32, @floatFromInt(n));

    //const W1 = G.tensor(.wgt, .r32, mp.Rank(2){ k, n });
    //const b1 = G.tensor(.wgt, .r32, mp.Rank(2){ k, n });

    //const W2 = G.tensor(.wgt, .r32, mp.Rank(2){ m, k });
    //const b2 = G.tensor(.wgt, .r32, mp.Rank(2){ m, k });

    var score: f32 = 0.0;

    var trg: mp.types.Key = 0;

    _ = &trg;

    _ = &score;

    ///////////////////////////////////////////////////
    // feed forward network...

    // project to squared dimensions...
    
    mp.mem.randomize(X1, .gauss);
    mp.mem.randomize(X2, .gauss);

    mp.mem.fill(t, 1.0);

    mp.mem.randomize(Q, .gauss);
    mp.mem.randomize(K, .gauss);
    mp.mem.randomize(V, .gauss);

    for (0..10) |_| {

        // query, key, value
        // mn,nm->mm
        const QX = mp.ops.innerProductScaled(Q, X1, alpha, "ij,jk->ik");
        const KX = mp.ops.innerProductScaled(K, X2, alpha, "ij,jk->ik");
        const VX = mp.ops.innerProductScaled(V, X1, alpha, "ij,jk->ik");

        // calculate overlap
        const QK = mp.ops.innerProduct(QX, KX, "ij,kj->ik");
        const SM = mp.ops.softmax(QK, "ij|j");
        const SV = mp.ops.innerProduct(SM, VX, "ij,jk->ik");
        const L2 = mp.ops.norm.l2(SV, "ij|j");
        const r = mp.ops.reduce(L2, "ij->i");

        mp.loss.bce(r, t, .{
            .grads = true,
            .score = &score
        });

        r.reverse(.keep);

        sgd.update(G);

        G.reset(.node, .all);
        G.reset(.leaf, .grd);

        std.log.info("score - {}", .{ score });
    }

    ////////////////////////////////////////////
    mp.device.check();

    std.log.info("Experimental: SUCCESS", .{});
}
