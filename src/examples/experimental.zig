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

    const key_len: usize = 10;
    
    const keys_cpu = try EU.allocCPU(u32, key_len);
        defer EU.freeCPU(keys_cpu);

    for (0..key_len) |i| {
        keys_cpu[i] = 0;
    }

    const keys = mp.mem.alloc(mp.types.Key, key_len, stream);
        defer mp.mem.free(keys, stream);

    mp.mem.copyToDevice(keys_cpu, keys, stream);

    mp.stream.synchronize(stream);

    mp.algo.key.reduceScaled(X, Y, keys, 0.10, "ij->j");

    try EU.copyAndPrintMatrix("Y", Y.values(), n, 1, stream);

    ////////////////////////////////////////////
}
