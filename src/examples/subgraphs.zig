const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

pub fn main() !void {
    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
    defer G.deinit();

    const M: usize = 32;

    /////////////////////////////////////////////////////

    const x  = G.tensor(.inp, .r32, &.{ M });
    const w1 = G.tensor(.wgt, .r32, &.{ M });
    const w2 = G.tensor(.wgt, .r32, &.{ M });

    mp.algo.randomize(x,  .gauss);
    mp.algo.randomize(w1, .gauss);
    mp.algo.randomize(w2, .gauss);

    /////////////////////////////////////////////////////

    // first, let's create a graph
    const z1 = mp.ops.add(x, w1);
    const z2 = mp.ops.selu(z1);

    const z3 = mp.ops.add(z2, w2);
    const z4 = mp.ops.selu(z3);

    // our graph now looks like the following:

    //                x   w1
    //                 \ /
    //                  +
    //                  |
    //                  z1
    //                  |
    //                 selu
    //                  |
    //                  z2  w2
    //                   \ /
    //                    +
    //                    |
    //                    z3
    //                    |
    //                   selu
    //                    |
    //                    z4

    // This could be thought of as two separate blocks:
    //
    //   block: selu(a + b)

    // We can split this into two subgraphs with "detach"

    z2.detach();

    // this has now broken the reversal chain into two
    // sub graphs that now look like the following:

    //                x   w1
    //                 \ /
    //                  +
    //                  |
    //                  z1
    //                  |
    //                 selu
    //                  |
    //                  z2
    //
    //                z2  w2
    //                 \ /
    //                  +
    //                  |
    //                  z3
    //                  |
    //                 selu
    //                  |
    //                  z4

    // reversals will calculate down to z2's gradients,
    // but no further than that. We will also free
    // all of the z4 subgraph up to but not including
    // z2 (the detachment)...

    z4.reverse(.free);

    std.debug.assert(z4.len() == 0);
    std.debug.assert(z3.len() == 0);
    std.debug.assert(z4.grads() == null);
    std.debug.assert(z3.grads() == null);

    // weights and inputs are not freed
    std.debug.assert(z2.len() != 0);
    std.debug.assert(w2.len() != 0);
    std.debug.assert(z2.grads() != null);
    std.debug.assert(w2.grads() != null);

    // we have not reversed these yet
    //std.debug.assert(x.len() != 0);
    std.debug.assert(z1.len() != 0);
    std.debug.assert(w1.len() != 0);
    //std.debug.assert(x.grads() == null);
    std.debug.assert(z1.grads() == null);
    std.debug.assert(w1.grads() == null);

    // we'll proceed to reverse z2's subgraph, but keep its nodes

    z2.reverse(.keep);

    std.debug.assert(x.len() != 0);
    std.debug.assert(z1.len() != 0);
    std.debug.assert(w1.len() != 0);
    std.debug.assert(z1.grads() != null);
    std.debug.assert(w1.grads() != null);
    //std.debug.assert(x.grads() != null);

    // This now gives us interesting opportunities for optimizations.
    // Notably, we can partially free the subgraph beneath z4. This
    // means that we can reuse the memory collected from subgraph 2
    // in the backwards calculation of subgraph 1.

    // Now, we can free nodes in the graph.
    G.reset(.node, .all);

    // To free the weight gradients, we can use reset:
    G.reset(.leaf, .grd);

    // You cannot reuse the nodes again after you do this.

    ////////////////////////////////////////////
    mp.device.check();

    std.log.info("Subgraphs: SUCCESS", .{});
}
