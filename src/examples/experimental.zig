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
    const N: usize = M * 4;

    /////////////////////////////////////////////////////
    // feed forward network...

    const x = G.tensor(.inp, .r32, mp.Rank(1){ M });  

    const W1 = G.tensor(.wgt, .r32, mp.Rank(2){ N, M });  
    const b1 = G.tensor(.wgt, .r32, mp.Rank(1){ N });  

    const W2 = G.tensor(.wgt, .r32, mp.Rank(2){ M, N });  
    const b2 = G.tensor(.wgt, .r32, mp.Rank(1){ M });  

    // pub fn forward(self: *Self, x: anytype) NodeTensor(T) {

        // prevent freeing beyond block
        if (comptime @TypeOf(x).Class == .hid) {
            // x.detach();
        }

        const v1 = mp.ops.linear(W1, x, b1, "ij,j->i");
        const z1 = mp.ops.selu(v1);

        const v2 = mp.ops.linear(W2, z1, b2, "ij,j->i");
        const z2 = mp.ops.selu(v2);

        // if (self.cleanup) {
            //mp.stream.synchronize(stream);
            //defer v1.free();
            //defer v2.free();
        // }

        // z2.detach();

        // self.z2 = z2;

        // return z2;
    //}

    // pub fn reverse(self: *Self) void {

        z2.reverse();

        // if (self.cleanup) {
            //z2.ptr.freeSubgraph(z2, .all);
        //}
    //}

    //////////////////////////////////////////////////////
    // pub fn toCPU(self: *Self, stream: Stream) {
    //
    //     DU.copyFromDevice(self.z1, self.z1_cpu, stream);
    //     DU.copyFromDevice(self.z1, self.z1_cpu, stream);
    //
        

    // }

    // later...

    z2.reverse();

    //mp.stream.synchronize(stream);

    //G.freeSubgraph(z2, .all);

    ////////////////////////////////////////////
}
