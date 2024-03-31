const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// TODO: actually explain the basics here


pub fn main() !void {
    mp.device.init(0);
    
    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
        defer G.deinit();

    const m: usize = 32;

    const x = G.tensor(.inp, .r32, mp.Rank(1){m});
        defer x.free();

    mp.mem.sequence(x, 0.0, 1.0);

    const y = mp.ops.norm.l2(x, "i|i");

    y.reverse(.keep);

    mp.device.check();

    try EU.copyAndPrintMatrix("norm", y.values(), m, 1, x.stream());
    try EU.copyAndPrintMatrix("norm", x.grads().?, m, 1, x.stream());
}
