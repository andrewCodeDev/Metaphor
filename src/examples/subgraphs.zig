const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");


pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .stream = stream,
        .mode = .train
    });

    defer G.deinit();

    const M: usize = 32;

    /////////////////////////////////////////////////////

    const  x = G.tensor(.inp, .r32, mp.Rank(1){ M });  
    const w1 = G.tensor(.wgt, .r32, mp.Rank(1){ M });  
    const w2 = G.tensor(.wgt, .r32, mp.Rank(1){ M });  

    mp.mem.randomize(x);
    mp.mem.randomize(w1);
    mp.mem.randomize(w2);

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
    // but no further than that.

        z4.reverse();

        std.debug.assert(z4.grads() != null);
        std.debug.assert(z3.grads() != null);
        std.debug.assert(w2.grads() != null);

        std.debug.assert(z1.grads() == null);
        std.debug.assert(w1.grads() == null);
        std.debug.assert( x.grads() == null);

    // we'll proceed to reverse z2's subgraph...

        z2.reverse();

        std.debug.assert(z1.grads() != null);
        std.debug.assert(w1.grads() != null);
        std.debug.assert( x.grads() != null);

    // This now gives us interesting opportunities for optimzations.
    // Notably, we can partially free the subgraph beneath z4. This
    // means that we can reuse the memory collected from subgraph 2
    // in the backwards calculation of subgraph 1. In this case, we
    // haven't done that for the sake of demonstration.

        G.freeSubgraph(z4, .all);

    // we've freed the values and gradients for z3 and z4
    // we have not freed the values and gradients for z2 and z1

        std.debug.assert(z4.grads() == null);
        std.debug.assert(z3.grads() == null);
        std.debug.assert(w2.grads() != null);
        std.debug.assert(z2.grads() != null);

    // it's important to note that freeing the subgraph
    // *does not* free the weights or inputs associated
        G.freeSubgraph(z2, .all);

        std.debug.assert(z2.grads() == null);
        std.debug.assert(z1.grads() == null);

        std.debug.assert(w1.grads() != null);
        std.debug.assert( x.grads() != null);

    // check our values - only the leaves
    // should be still non-zero sized

        std.debug.assert(0 !=  x.len());
        std.debug.assert(0 != w1.len());
        std.debug.assert(0 != w2.len());

        std.debug.assert(0 == z1.len());
        std.debug.assert(0 == z2.len());
        std.debug.assert(0 == z3.len());
        std.debug.assert(0 == z4.len());

    ////////////////////////////////////////////

    std.log.info("Subgraphs: SUCCESS", .{});
}
