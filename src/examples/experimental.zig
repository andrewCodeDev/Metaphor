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


    const M: usize = 33;
    const N: usize = 16;

    /////////////////////////////////////////////////////
    // feed forward network...

    const x = G.tensor(.inp, .r32, mp.Rank(2){M, N});

    //mp.mem.fill(x, 1.0);
    mp.mem.sequence(x, 0.0, 0.1);

    const y = mp.ops.softmax(x, "ij|j");

    try EU.copyAndPrintMatrix("softmax", y.values(), M, N, stream);

    ////////////////////////////////////////////
}
