const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// TODO: actually explain the basics here


pub fn main() !void {
    mp.device.init(0);

    //mp.device.reset();

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .eval });
        defer G.deinit();

    const m: usize = 9;

    const x = G.tensor(.inp, .r32, mp.Rank(1){m});
        defer x.free();

    mp.mem.randomize(x);

    const keys = mp.mem.alloc(mp.types.Key, x.len(), stream);
        defer mp.mem.free(keys, stream);

    mp.algo.key.sort(x, keys);

    try EU.copyAndPrintKeys(x.values(), keys, x.stream());

    //const W = G.tensor(.inp, .r32, mp.Rank(2){m, n});
    //const c = G.tensor(.inp, .r32, mp.Rank(1){n});

    //mp.mem.randomize(x);
    //mp.mem.randomize(W);
    //
    //const keys = mp.mem.allocKeys(n, stream);
    //    defer mp.mem.free(keys, stream);

    //const top_k: usize = 10;
    //const alpha: f32 = 0.10;

    //for (0..10) |_| {

    //    // get highest overlap:
    //    const y = mp.ops.innerProduct(W, x, "ij,j->i");
    //        defer y.free();

    //    mp.algo.key.sort(y, keys);
    //    
    //    mp.algo.key.reduceScaled(W, c, keys[0..top_k], alpha, "ij->j");
    //    
    //    mp.mem.resetKeys(keys, c.stream());
    //}


    //try EU.copyAndPrintMatrix("Y", Y.values(), n, 1, stream);

    ////////////////////////////////////////////
}
