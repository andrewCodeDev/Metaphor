const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

/////////////////////////////////////////////////
// Using Keys with Metaphor /////////////////////

// Keys are indices that represent an element or a
// a subsection of a tensor. Keys can be used to 
// parameterize operations on tensors and are useful
// for gaining insight about the elements in a tensor.

// It is important to understand that what a key represents
// depends on the algorithm using the keys. Some algorithms
// may interpret a key to signify a row, while others may
// recognize keys as referring to single elements.

// This may seem overly flexible, but it has several
// important advantages.

// Internally, keys are represented as u32 for spatial
// reasons. This means that in this current version of
// metaphor, key algorithms work up to MAX(u32) values.

// Future versions of metaphor will support multi-dimensional
// keys and algorithms. Here, we'll just focus on the case of
// key vectors and rank-1 tensors.

pub fn main() !void {
    // the usual setup - nothing special here

    mp.device.init(0);

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .eval });
    defer G.deinit();

    const n: usize = 512;

    // call normalize just for this example...
    const x = mp.ops.norm.l2(G.tensor(.inp, .r32, mp.Rank(1){n}), "i|i");
    defer x.free();

    const W = mp.ops.norm.l2(G.tensor(.inp, .r32, mp.Rank(2){n, n}), "ij|j");
    defer W.free();

    mp.mem.randomize(x);
    mp.mem.randomize(W);

    ///////////////////////////////////////////////
    // Creating and initializing keys /////////////

    // Here, we create a slice of keys. This is nothing special,
    // it's just an uninitialized array of u32's. I still recommend
    // using mp.types.Key incase implementation details change.
    const keys = mp.mem.alloc(mp.types.Key, n, stream);

    // You are responsible for freeing the memory for keys.
    defer mp.mem.free(keys, stream);

    // It is not recommended to create keys using the tensor_allocator
    // in the graph. The graph is free to make assumptions about its
    // own allocator and those will change with future versions.

    // AVOID: const keys = G.tensor_allocator.alloc(mp.types.Keys, n);

    // To start, let's begin by finding the similarity between a
    // single vector and a group of row vectors in a matrix
    // using the inner product. This works specifically because
    // we have normalized the vector and the rows of our matrix:
    const y = mp.ops.innerProduct(W, x, "ij,j->i");
    
    // y now contains the result of each dot-product of
    // x and every row in W. If x more closely resembles
    // on of the columns in W, the value of that dot product
    // result will be close to 1.0 - we can now sort those
    // values:

    // NOTE: Sort will initialize the keys.
    mp.algo.key.sort(y, keys);

    // Now the key vector contains the indices of each element
    // in order from lowest to highest. Let's say we want to
    // get the top 10 results:
    const top_k = keys[keys.len - 10..];
    
    // Recall that y contained the measure of similarity between
    // x and the rows of W - top_k now tells us which vectors we
    // most closely resemble.

    // We can use these to populate a vector with the sum of
    // these top 10 rows using a key-reduce algorithm.

    // NOTE: 
    //  Functions in the "algo" namespace do *not* allocate new
    //  tensors. They may allocate scratch memory, but a result
    //  parameter needs to be provided. In the case of key.sort,
    //  the out parameter was the "keys" slice.
    
    // create our output tensor "c"
    const c = G.tensor(.inp, .r32, mp.Rank(1){ n });

    // We're using the reduceScaled algorithm instead of just
    // reduce. This allows us to multiply by an arbitrarty alpha
    // scalar. In this case, I choose 0.1 because I want to
    // calculate an average vector that represents all 10 keys.

    // Alternative: mp.algo.key.reduce(W, c, top_k, "ij-j");

    // reduce key-valued i'th rows into a row of j columns
    mp.algo.key.reduceScaled(W, c, top_k, 0.1, "ij->j");

    // c now contains the average of the top-10 most similar
    // rows in W against x. This demonstrates the flexibility
    // of keys and why algorithms have the choice of their
    // intepretation of what keys represents.
    
    ////////////////////////////////////////////
    mp.device.check();
}
