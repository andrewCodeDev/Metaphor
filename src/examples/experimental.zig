const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

//TODO: actually explain the basics here

pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .stream = stream,
        .mode = .train
    });

    defer G.deinit();

    const row_x: usize = 50_000;

    /////////////////////////////////////////////////////

    const x = G.tensor(.wgt, .r32, mp.Rank(1){ row_x });  

    mp.mem.fill(x, 1.0);

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
    //try EU.copyAndPrintMatrix("X2: grads", X2.grads().?, row_x, col_x, stream);

    ////////////////////////////////////////////
}
