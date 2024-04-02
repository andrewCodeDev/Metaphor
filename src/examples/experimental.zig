const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// TODO: actually explain the basics here

pub fn populateCentroids(
    stream: mp.types.Stream,
    W: anytype,
    C: anytype,
    CT: anytype,
    R: anytype,
    max_gpu: []mp.types.Key,
    max_cpu: []mp.types.Key,
    rdx_gpu: []mp.types.Key,
    rdx_cpu: []mp.types.Key,
) void {
    // these could be moved outside of this function
    mp.mem.copyFromDevice(max_gpu, max_cpu, stream);

    mp.stream.synchronize(stream);

    const ct_sizes = CT.sizes();

    const m = ct_sizes[0];
    const n = ct_sizes[1];

    for (0..m) |c| {

        var rdx_len: usize = 0;

        for (0..max_cpu.len) |i| {

            if (c == max_cpu[i]) {
                rdx_cpu[rdx_len] = mp.scalar.as(mp.types.Key, i);
                rdx_len += 1;
            }
        }

        mp.mem.copyToDevice(rdx_cpu, rdx_gpu, stream);

        const coef: f32 = mp.scalar.as(f32, rdx_len);

        mp.algo.key.reduceScaled(W, R, rdx_gpu[0..rdx_len], 1.0 / coef, "ij->j");

        // locate the row
        const a: usize = n * c;

        // locate end col
        const b: usize = a + n;

        mp.mem.copySlice(@TypeOf(R).DataType, R.values(), CT.values()[a..b], stream);
    }

    mp.raw_ops.norm_l2_ij_j(stream, CT, CT);

    // write our values back to C for next computation
    mp.raw_ops.permutate_ij_ji(stream, CT, C);
}


//pub fn populateCentroids(
//    stream: mp.types.Stream,
//    W: anytype,
//    WB: anytype,
//    CT: anytype,
//    max_gpu: []mp.types.Key,
//    max_cpu: []mp.types.Key,
//    rdx_gpu: []mp.types.Key,
//    rdx_cpu: []mp.types.Key,
//) void {
//    // these could be moved outside of this function
//    mp.mem.copyFromDevice(max_gpu, max_cpu, stream);
//
//    mp.stream.synchronize(stream);
//
//    const ct_sizes = CT.sizes();
//
//    const m = ct_sizes[0];
//    const n = ct_sizes[1];
//
//    for (0..m) |c| {
//
//        var rdx_len: usize = 0;
//
//        for (0..max_cpu.len) |i| {
//
//            if (c == max_cpu[i]) {
//                rdx_cpu[rdx_len] = mp.scalar.as(mp.types.Key, i);
//                rdx_len += 1;
//            }
//        }
//
//        for (0..max_cpu.len)
//
//        mp.mem.copyToDevice(rdx_cpu, rdx_gpu, stream);
//
//        const coef: f32 = mp.scalar.as(f32, rdx_len);
//
//        mp.algo.key.reduceScaled(W, R, rdx_gpu[0..rdx_len], 1.0 / coef, "ij->j");
//
//        // locate the row
//        const a: usize = n * c;
//
//        // locate end col
//        const b: usize = a + n;
//    }
//
//    mp.raw_ops.norm_l2_ij_j(stream, CT, CT);
//
//    // write our values back to C for next computation
//    mp.raw_ops.permutate_ij_ji(stream, CT, C);
//}

pub fn main() !void {
    mp.device.init(0);
    
    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
        defer G.deinit();

    const m: usize = 32;
    const n: usize = 16;
    //const k: usize = 3;

    //////////////////////////////////////////////////

    // W: tensors that we want to cluster
    //const W   = G.tensor(.inp, .r32, mp.Rank(2){ m, n });
  //  const WB  = G.tensor(.inp, .r32, mp.Rank(2){ m, n });

    //// C: centroid matrix (centroids are columns)
    //const C  = G.tensor(.inp, .r32, mp.Rank(2){ n, k });    

    //// CT: centroid matrix transposed (used to write back to C)
    //const CT = G.tensor(.inp, .r32, mp.Rank(2){ k, n });    

    //// S: similarity score between W and C
    //const S  = G.tensor(.inp, .r32, mp.Rank(2){ m, k });    

    // R: storage for row reduce to calculate average tensor
    const R  = G.tensor(.inp, .r32, mp.Rank(1){ n });    

    mp.mem.sequence(R, 0.0, 1.0);

    const Y = mp.ops.broadcast(R, &.{ m, n }, "j->ij");

    Y.reverse(.keep);

    try EU.copyAndPrintMatrix("Broadcast", Y.values(), m, n, stream);
    try EU.copyAndPrintMatrix("Broadcast Reverse", R.grads().?, 1, n, stream);

    // We can make all of our key allocations upfront, too.

    //const max_gpu = mp.mem.alloc(mp.types.Key, m, stream);
    //    defer mp.mem.free(max_gpu, stream);

    //const rdx_gpu = mp.mem.alloc(mp.types.Key, max_gpu.len, stream);
    //    defer mp.mem.free(rdx_gpu, stream);

    //const max_cpu = std.heap.c_allocator.alloc(mp.types.Key, max_gpu.len) catch unreachable;
    //    defer std.heap.c_allocator.free(max_cpu);
    //
    //const rdx_cpu = std.heap.c_allocator.alloc(mp.types.Key, max_gpu.len) catch unreachable;
    //    defer std.heap.c_allocator.free(rdx_cpu);

    //mp.mem.randomize(CT, .gauss);
    //mp.mem.randomize(W,  .gauss);

    /////////////////////////////////////////////////////////
    // In this example, we'll use the raw_ops namespace to
    // do inplace operations that don't interact with the
    // graph to setup our input tensors. Note that these
    // cannot create a reversal trail.

    // Sometimes, raw_ops can save memory and compute faster
    // if you already know what dimensions your tensors have.

    // normalize matrix rows to itself
    //mp.raw_ops.norm_l2_ij_j(stream, W, W);

    ////// normalize matrix rows to itself
    //mp.raw_ops.norm_l2_ij_j(stream, CT, CT);

    ////// transpose values from CT to C
    //mp.raw_ops.permutate_ij_ji(stream, CT, C);

    //try EU.copyAndPrintMatrix("\nCentroids", C.values(), n, k, stream);

    //for (0..10) |_| {
    //        
    //    // calculate similarity:
    //    mp.raw_ops.linear_ij_jk(stream, W, C, 1.0, S, 0.0, S);

    //    // find top matches:
    //    mp.algo.key.max(S, max_gpu, "ij->j");

    //    populateCentroids(
    //        stream, W, C, CT, R, max_gpu, max_cpu, rdx_gpu, rdx_cpu
    //    );

    //    try EU.copyAndPrintMatrix("\nCentroids", C.values(), n, k, stream);

    //    mp.stream.synchronize(stream);
    //}
}
