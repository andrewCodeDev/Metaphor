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

    const m: usize = 32;
    const n: usize = 16;

    const X = G.tensor(.inp, .r32, mp.Rank(2){m, n});
    const Y = G.tensor(.inp, .r32, mp.Rank(1){n});

    mp.mem.fill(X, 1.0);

    const key_len: usize = 6;
    
    const keys = mp.mem.allocKeys(key_len, stream);
        defer mp.mem.free(keys, stream);

    mp.algo.key.reduceScaled(X, Y, keys, 0.1, "ij->j");

    try EU.copyAndPrintMatrix("Y", Y.values(), n, 1, stream);

    ////////////////////////////////////////////
}
