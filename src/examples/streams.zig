const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

// Streams on CUDA act like queues.

// Each kernel call gets queued up to the GPU and can be
// launched without needing to wait for the last one to
// synchronize. Streams are synchronized with respect
// to their own queue. Streams are not synchronized
// among other streams.

pub fn main() !void {
    mp.device.init(0);

    // streams can be used as groups via the StreamGroup type

    const streams = mp.stream.Group(2).init();
    defer streams.deinit();

    const G = mp.Graph.init(.{
        /////////////////////////////
        // this is the initial stream
        .stream = streams.items[0],
        ////////////////////////////
        .mode = .eval,
    });

    defer G.deinit();

    ////////////////////////////////
    // streams can be swapped out...
    //     ex: G.stream = stream2;

    const M: usize = 2096;
    const N: usize = 2096;

    // A gets created using the stream in the graph. The
    // graph remembers what stream that A was created with.
    const A = G.tensor(.inp, .r32, mp.Rank(2){M, N});

    // x gets created using the stream in the graph. The
    // graph remembers what stream that x was created with.
    const x = G.tensor(.inp, .r32, mp.Rank(2){N, M});

    // both of these fill calls use the currently assigned
    // stream as well. If we had changed this stream,
    // then both fill calls would have been on that stream
    mp.mem.fill(A, 1.0);
    mp.mem.fill(x, 1.0);

    // pre-caching allows us to allocate and then cache
    // tensors to remove the overhead of the initial
    // allocation. This again uses the graph's recent
    // stream that has been assigned.
    G.precache(mp.scalar.r32, M * N, 10);

    // this prevents us from moving forward until the
    // stream has finished all the work in its queue.
    // Since we only have one stream, this is the same
    // as mp.stream.synchronize(G.stream);
    mp.stream.synchronize(streams.items[0]);

    std.log.info("Tensors in Cache after graph.precache: {}", .{G.tensor_allocator.used()});
    // What follows is 10 matrix multiplications. These
    // are queued up by the CPU one after another.

    // Here's what actually happens... the cpu goes ahead
    // and creates the data elements for each of these
    // products (sizes, computes strides, evicts caches...)

    // The kernels assume that the data is ready to go,
    // so they begin operating on the information as soon
    // as they are ready to go. It's important that we don't
    // free anything until we've resynchronized.

    ////////////////////////////////////////////////////
    {
        const start = try std.time.Instant.now();

        for (0..10) |_| {
            const y = mp.ops.innerProduct(A, x, "ij,jk->ik");
            _ = &y;
        }

        mp.stream.synchronize(streams.items[0]);

        const stop = try std.time.Instant.now();

        const delta = stop.since(start);

        std.log.info("GPU 1 stream elapsed (ns): {} - Run 1", .{delta});
    }
    /////////////////////////////////////////////////////

    // this clears our node cache, but keeps all of the
    // memory for each allocator, meaning we now have a
    // contiguous block of memory that will be used for
    // our next round of operations.

    std.log.info("Tensors in Cache Before graph.reset: {}", .{G.tensor_allocator.used()});

    G.reset(.node, .all);

    std.log.info("Tensors in Cache After graph.reset: {}", .{G.tensor_allocator.used()});
    ////////////////////////////////////////////////////
    {
        const start = try std.time.Instant.now();

        for (0..10) |_| {
            const y = mp.ops.innerProduct(A, x, "ij,jk->ik");
            _ = &y;
        }

        mp.stream.synchronize(streams.items[0]);

        const stop = try std.time.Instant.now();

        const delta = stop.since(start);

        std.log.info("GPU 1 stream elapsed (ns): {} - Run 2", .{delta});
    }
    /////////////////////////////////////////////////////

    // this time, we'll make two streams and do
    // 5 products on one, and five on the other

    G.reset(.node, .all);
    /////////////////////////////////////////////////////
    {
        const start = try std.time.Instant.now();

        G.stream = streams.items[0];

        for (0..5) |_| {
            const y = mp.ops.innerProduct(A, x, "ij,jk->ik");
            _ = &y;
        }

        G.stream = streams.items[1];

        for (0..5) |_| {
            const y = mp.ops.innerProduct(A, x, "ij,jk->ik");
            _ = &y;
        }

        // streams can be synchronized as a group
        streams.synchronize();

        const stop = try std.time.Instant.now();

        const delta = stop.since(start);

        std.log.info("GPU 2 stream elapsed (ns): {} - Run 3", .{delta});
    }
    /////////////////////////////////////////////////////

    // for fun, we'll compare to the CPU

    const A_cpu = try EU.allocCPU(f32, M * N);
    defer EU.freeCPU(A_cpu);

    const x_cpu = try EU.allocCPU(f32, M);
    defer EU.freeCPU(x_cpu);

    const y_cpu = try EU.allocCPU(f32, N);
    defer EU.freeCPU(y_cpu);

    {
        const start = try std.time.Instant.now();

        for (0..1) |_| {
            EU.cpuMatmul(A_cpu, x_cpu, y_cpu, M, N, M);
            _ = &y_cpu;
        }

        const stop = try std.time.Instant.now();

        const delta = stop.since(start);

        std.log.info("CPU 1 thread elapsed (ns): {}", .{delta});
    }

    ////////////////////////////////////////////
}
