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

    const row: usize = 32;

    /////////////////////////////////////////////////////

    const x = G.tensor(.inp, .r32, mp.Rank(1){ row });  
    const w = G.tensor(.wgt, .r32, mp.Rank(1){ row });  

    mp.mem.randomize(x);
    mp.mem.randomize(w);

    const trg: mp.types.IndexType = 31;
    var score: f64 = 0.0;

    /////////////////////////////////////////////////////

    for (0..10) |_| {

        const y = mp.ops.add(x, w);

        mp.loss.cce(y, trg, .{
            .grads = true,
            .score = &score,
        });

        y.reverse();

        mp.stream.synchronize(stream);

        G.reset(.node, .all);
        G.reset(.leaf, .grd);

        std.log.info("loss: {}", .{ score });
    }

    //try EU.copyAndPrintMatrix("grads", x.grads().?, row, 1, stream);



    /////////////////////////////////////////////////////
    //for (0..10) |_| {
    //    const start = try std.time.Instant.now();

    //    const y = mp.ops.softmax(x, "i|i");

    //    const stop = try std.time.Instant.now();

    //    const delta = stop.since(start);

    //    y.free();
    //    
    //    std.log.info("GPU 1 stream elapsed (ns): {} - Run 1", .{ delta });
    //}

    //try EU.copyAndPrintMatrix("Z1: value", Z1.values(), 1, row_x, stream);
    //try EU.copyAndPrintMatrix("X1: grads", X1.grads().?,     1, row_x, stream);

    //try EU.copyAndPrintMatrix("X1: value", X1.values(),      1, row_x, stream);
    //try EU.copyAndPrintMatrix("X2: grads", X2.grads().?, row_x, row_x, stream);

    ////////////////////////////////////////////
}
