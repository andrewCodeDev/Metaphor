# Metaphor

Metaphor is a Zig based machine-learning library. The goal is to be fully featured and feel pythonic without sacrificing the low-level control of Zig.

This library is in a highly experimental state and will likely change a lot. Contributions and feedback are welcome!

# How does Metaphor work?

Metaphor is a stream based library. Streams are work queues that can be created on your GPU (much like threads):

```zig
// To start Metaphor, you must initialize the
// device for the GPU device you want to use
// Initialize device and on device zero:

mp.device.init(0);

// Open the stream you want to compute on.
// Streams can be run in parallel to launch
// multiple kernels simultaneously:

const stream = mp.stream.init();
defer mp.stream.deinit(stream);

// wait until all work is done on a stream
mp.streams.synchronize(stream);

//////////////////////////////////////////////

// alternatively, you can create stream-groups:
const streams = mp.stream.Group(3).init();
defer streams.deinit();

// get your streams
const s0 = streams.items[0];

// wait until all streams are synchronized
streams.synchronize();
```

Once you've setup your streams, you can create your graph. The graph manages your tensor data and can track operations that occur:

```zig
const G = mp.Graph.init(.{
    .stream = stream,
    .mode = train,
});
defer G.deinit();
```

Now, we can build tensors. Tensors are the unit of computation in Metaphor:

```zig
// tensors are freed on graph.deinit()...
const X1 = G.tensor(.inp, .r32, mp.Rank(2){ 2, 2 });
const X2 = G.tensor(.wgt, .r32, mp.Rank(2){ 2, 2 });

// but they can be safely freed before deinit like this:
// X1.free();
```

There are several namespaces within Metaphor, including the "ops", "mem", "scalar", "loss", and more.

The Metaphor device utilities (which including copying to the device, creating streams, synchronization, etc...) can be used independently of machine learning!

```zig
// fill with a single value (casted automatically)
mp.mem.fill(X1, 2);

// sequence all elements from 0.0 in increments of 1.0
mp.mem.sequence(X2, 0.0, 1.0);
```

Metaphor expresses dimensional operations inspired by the Einsteinian convention. All strings are parsed at compile time - no runtime overhead!

```zig
// y = A.x
const y = mp.ops.innerProduct(A, x, "ij,j->i");

// y = A.x + b
const y = mp.ops.linear(A, x, b, "ij,j->i");

// B = A transpose
const B = mp.ops.permutate(A, "ij->ji");

// w = u + v
const w = mp.ops.add(u, v);

// operations can be composed, e = (a + b) * (c + d)
const e = mp.ops.hadamard(mp.ops.add(a, b), mp.ops.add(c, d));
```

Reversal is straight-forward, but has many additional features (see src/examples for more):

```zig
// feed-forward block
const y = mp.ops.selu(mp.ops.linear(x, A, b, "i,ij->j"));

// if we want to free the hidden nodes, we can use
// ".free" - this does not free weights or inputs.
y.reverse(.free);

// otherwise, if we don't want to keep hidden nodes, we can use
// ".keep" - this does not free weights or inputs.

// inspect gradients
if (A.grads()) |grd| {
    // use gradient...
}
```

# Installation Steps

If you do not have a version of the GCC compiler, download/install it and locate the path it was saved to.

If you do not have a version of the CUDA Developer Toolkit, download it and locate the path it was saved to.

Clone this repository onto your local machine.

Either copy, move, or symlink the CUDA Developer Toolkit to the "Metaphor/deps" folder. Metaphor uses this path to ensure that CUDA includes are correct.

Open the "Metaphor/config.json" file and provide your own values for the following:

```json
{
    "gcc-bin-path": "/usr/bin/gcc",
    "gpu-architecture": "sm_89"
}
```

You can now use Metaphor!
