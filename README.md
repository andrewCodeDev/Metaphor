# Metaphor
Metaphor is a Zig library with Cuda backings for building machine learning models. This library is in a highly experimental state and will likely change a lot.


# Installation Steps

If you do not have a version of the CUDA Developer Toolkit on your machine, download it and locate the path it was saved to.

Clone this repository onto your local machine.

Either copy, move, or symlink the CUDA Developer Toolkit to the "Metaphor/dependencies" folder. Metaphor uses this path to ensure that CUDA includes are correct.

Open the "Metaphor/config.json" file and specify your GPU's compute architecture (such as sm_89) in the config.json file.

You can now use Metaphor!

# Examples:

```zig
const mp = @import("metaphor.zig");

pub fn main() !void {

    // To start Metaphor, you must initialize the
    // device for the GPU device you want to use

    // Initialize device and cuda context on device zero
    mp.device.init(0);

    // Open the stream you want to compute on.
    // Streams can be run in parallel to launch
    // multiple kernels simultaneous.
    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .auto_free_wgt_grads = false,
        .auto_free_inp_grads = false,
        .auto_free_hid_nodes = true,
        .stream = stream,
    });

    defer G.deinit();

    /////////////////////////////////////////////////

    // tensors are freed on graph.deinit()
    const X1 = G.tensor("X1", .wgt, .r32, mp.Rank(2){ 2, 2 });  
    const X2 = G.tensor("X2", .wgt, .r32, mp.Rank(2){ 2, 2 });

    mp.mem.fill(X1, 2);
    mp.mem.fill(X2, 1);

    for (0..10) |i| {

        var clock = try std.time.Timer.start();

        const Z1 = ops.hadamard(X1, X2);
        const Z2 = ops.hadamard(X1, X2);
        const Z3 = ops.add(Z1, Z2);

        Z3.reverse();

        const delta = clock.lap();

        std.debug.print(
           "\n\n==== Lap {}: {} ====\n", 
           .{ i, delta }
        );
    }
    ////////////////////////////////////////////
}

```


