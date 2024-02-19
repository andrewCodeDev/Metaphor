# Metaphor
Metaphor is a Zig library with Cuda backings for building machine learning models. This library is in a highly experimental state and will likely change a lot.

Examples:

```zig
pub fn main() !void {

    var G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .auto_free_wgt_grads = false,
        .auto_free_inp_grads = false,
        .auto_free_hid_nodes = true,
    });

    defer G.deinit();

    /////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Dims(2){ 2, 2 });    
        defer X1.free();
    
    const X2 = G.tensor("X2", .wgt, .r32, mp.Dims(2){ 2, 2 });
        defer X2.free();

    ops.fill(X1, 2);
    ops.fill(X2, 1);

    for (0..10) |i| {

        var clock = try std.time.Timer.start();

        const Z1 = ops.hadamard(X1, X2);
        const Z2 = ops.hadamard(X1, X2);
        const Z3 = ops.add(Z1, Z2);

        Z3.reverse();

        const delta = clock.lap();

        std.debug.print(
           "\n\n==== Lap {}: {} ====\n" ++
           "\nX2: {d:.2}\nX2: {d:.2}\n" ++
           "\nX1: {d:.2}\nX1: {d:.2}\n",.{
        
           i, delta,
           X2.values(), X2.grads().?,    
           X1.values(), X1.grads().?,
        });
    }

    ////////////////////////////////////////////
}

```


