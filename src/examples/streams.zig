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

    const stream = mp.stream.init();
        
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .auto_free_wgt_grads = false,
        .auto_free_inp_grads = false,
        .auto_free_hid_nodes = false,
        /////////////////////////////
        // this is the initial stream
        .stream = stream,
        ////////////////////////////
        .mode = .eval
    });

    defer G.deinit();

    ////////////////////////////////
    // streams can be swapped out...
    //     ex: G.stream = stream2;

    const M: usize = 2096;
    const N: usize = 2096;

    // A get created using the stream in the graph. The
    // graph remembers what stream that A was created with.
    const A = G.tensor("A", .wgt, .r32, mp.Rank(2){ M, N });  

    // X get created using the stream in the graph. The
    // graph remembers what stream that A was created with.
    const x = G.tensor("x", .wgt, .r32, mp.Rank(1){ M });  

    // both of these fill calls use the currently assigned
    // stream as well. If we had changed this stream,
    // then both fill calls would have been on that stream
    mp.mem.fill(A, 1.0);
    mp.mem.fill(x, 1.0);

    // precaching allows us to allocate and then cache
    // tensors to remove the overhead of the initial
    // allocation. This again uses the graph's recent
    // stream that has been assigned.
    G.precache(mp.scalar.r32, M, 10);

    // this prevents us from moving forward until the
    // stream has finished all the work in its queue.
    // Since we only have one stream, this is the same
    // as mp.stream.synchronize(G.stream);
    mp.stream.synchronize(stream);

    std.log.info("Tensors in Cache after graph.precache: {}", .{ G.tensor_allocator.used() });
    // What follows is 10 matrix multiplications. These
    // are queued up by the CPU one after another.

    // Here's what actually happens... the cpu goes ahead
    // and creates the data elements for each of these
    // products (sizes, computes strides, uncaches memory...)

    // The kernels asssume that the data is ready to go,
    // so they begin operating on the information as soon
    // as they are ready to go. It's important that we don't
    // free anything until we've resynchronized.

    ////////////////////////////////////////////////////
    {
    const start = try std.time.Instant.now();

    for (0..10) |_| {
        const y = mp.ops.innerProduct(A, x, "ij,j->i");
        _ = &y;
    }

    mp.stream.synchronize(stream);

    const stop = try std.time.Instant.now();

    const delta = stop.since(start);
    
    std.log.info("GPU 1 stream elapsed (ns): {} - Run 1", .{ delta });
    }
    /////////////////////////////////////////////////////

    // this clears our node cache, but keeps all of the
    // memory for each allocator, meaning we now have a
    // contiugous block of memory that will be used for
    // our next round of operations.

    std.log.info("Tensors in Cache Before graph.reset: {}", .{ G.tensor_allocator.used() });

    G.reset(.node);    

    std.log.info("Tensors in Cache After graph.reset: {}", .{ G.tensor_allocator.used() });
    ////////////////////////////////////////////////////
    {
    const start = try std.time.Instant.now();

    for (0..10) |_| {
        const y = mp.ops.innerProduct(A, x, "ij,j->i");
        _ = &y;
    }

    mp.stream.synchronize(stream);

    const stop = try std.time.Instant.now();

    const delta = stop.since(start);
    
    std.log.info("GPU 1 stream elapsed (ns): {} - Run 2", .{ delta });
    }
    /////////////////////////////////////////////////////

    // this time, we'll make two streams and do
    // 5 products on one, and five on the other
    const stream2 = mp.stream.init();

    G.reset(.node);
    /////////////////////////////////////////////////////
    {
    const start = try std.time.Instant.now();

    for (0..5) |_| {
        const y = mp.ops.innerProduct(A, x, "ij,j->i");
        _ = &y;
    }

    G.stream = stream2;

    for (0..5) |_| {
        const y = mp.ops.innerProduct(A, x, "ij,j->i");
        _ = &y;
    }   

    mp.stream.synchronize(stream);
    mp.stream.synchronize(stream2);

    const stop = try std.time.Instant.now();

    const delta = stop.since(start);
    
    std.log.info("GPU 2 stream elapsed (ns): {} - Run 3", .{ delta });
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

    for (0..10) |_| {
        EU.cpuMatmul(A_cpu, x_cpu, y_cpu, M, N, 1);
        _ = &y_cpu;
    }

    const stop = try std.time.Instant.now();

    const delta = stop.since(start);
    
    std.log.info("CPU 1 thread elapsed (ns): {}", .{ delta });
    }

    ////////////////////////////////////////////
}
