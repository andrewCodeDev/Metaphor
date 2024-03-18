const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

//TODO: actually explain the basics here

pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    var sgd = mp.optm.SGD.init(.{ .rate = 1.0, });

    const G = mp.Graph.init(.{
        .optimizer = sgd.optimizer(),
        .stream = stream,
        .mode = .train
    });

    defer G.deinit();

    const M: usize = 32;

    /////////////////////////////////////////////////////
    // feed forward network...

    const x = G.tensor(.inp, .r32, mp.Rank(1){ M });  
    const b = G.tensor(.wgt, .r32, mp.Rank(1){ M });  
    const t = G.tensor(.wgt, .r32, mp.Rank(1){ M });  

    mp.mem.randomize(x);
    mp.mem.randomize(b);
    mp.mem.randomize(t);

    var score: f64 = 0.0;

    for (0..10) |_| {

        const y = mp.ops.add(x, b);

        mp.loss.mse(y, t, .{
            .grads = true,
            .score = &score
        });

        y.reverse();

        mp.stream.synchronize(stream);

        G.reset(.node, .all);

        std.log.info("score: {d:.4}", .{ score });
    }
    ////////////////////////////////////////////
}
