const std = @import("std");
const assert = std.debug.assert;

const core = @import("core");
pub const Graph = core.Graph;
pub const Tensor = core.Tensor;
pub const OpDatum = core.OpDatum;
pub const scalar = core.scalar;

//const appendNode = CG.appendNode;
//pub const enableGradient = CG.enableGradient;
//const nodeTensor = CG.nodeTensor;

//const Child = UT.Child;
//const Tensor = CG.Tensor;
//const Optimizer = @import("optimizer.zig");

//const Algo = @import("algorithm.zig");
//const TenOps = @import("tensor_ops.zig");
//const Loss = @import("loss.zig");

///////////////////////////////////////////////
///////////////////////////////////////////////

pub const device = struct {
    pub const init = core.utils.init_device;
    pub const sync = core.utils.synchronize_device;
    pub const check = core.utils.check_last_error;
};

pub const stream = struct {
    pub const init = core.utils.init_stream;
    pub const deinit = core.utils.deinit_stream;
    pub const sync = core.utils.synchronize_stream;
    pub const Group = core.utils.StreamGroup;
};

pub const util = struct {
    pub const to_device = core.utils.copy_to_device;
    pub const from_device = core.utils.copy_from_device;
    pub const alloc = core.utils.alloc;
    pub const create = core.utils.create;
    pub const free = core.utils.free;

    //pub const load = @import("tensor_file.zig").loadTensor;
    //pub const save = @import("tensor_file.zig").saveTensor;
};

//pub const types = struct {
//    pub const Tensor = G.Tensor;
//    pub const LeafTensor = CG.LeafTensor;
//    pub const SliceUnion = TC.SliceUnion;
//    pub const IndexType = TC.IndexType;
//    pub const SizeType = TC.SizeType;
//    pub const Strides = TC.Strides;
//    pub const Sizes = TC.Sizes;
//    pub const Stream = core.Stream;
//    pub const Key = Algo.Key;
//};

//pub const optm = struct {
//    pub const SGD = Optimizer.SGD;
//    pub const Momentum = Optimizer.Momentum;
//};

//pub const loss = struct {
//    pub fn cce(x: anytype, y: anytype, config: struct {
//        grads: bool,
//        score: ?*f32,
//    }) void {
//        const T = @TypeOf(x);
//
//        const graph = x.ptr;
//
//        if (config.grads) {
//            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
//        }
//        const redux: ?[*]f32 = if (config.score != null)
//            graph.tensor_allocator.create(f32, graph.stream)
//        else
//            null;
//        defer {
//            if (redux) |ptr| graph.tensor_allocator.destroy(ptr);
//        }
//        const score: ?[*]f32 = if (config.score) |ptr|
//            @ptrCast(@alignCast(ptr))
//        else
//            null;
//
//        Loss.cce(graph.stream, x, y, redux, score);
//    }
//
//    pub fn mse(x: anytype, y: anytype, config: struct {
//        grads: bool,
//        score: ?*f32,
//    }) void {
//        const T = @TypeOf(x);
//
//        const graph = x.ptr;
//
//        if (config.grads) {
//            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
//        }
//
//        const redux: ?[*]f32 = if (config.score != null)
//            graph.tensor_allocator.create(f32, graph.stream) else null;
//
//        defer {
//            if (redux) |ptr| graph.tensor_allocator.destroy(ptr);
//        }
//
//        const score: ?[*]f32 = if (config.score)
//            |ptr| @ptrCast(@alignCast(ptr)) else null;
//
//        Loss.mse(graph.stream, x, y, redux, score);
//    }
//
//    pub fn bce(x: anytype, y: anytype, config: struct {
//        grads: bool,
//        score: ?*f32,
//    }) void {
//        const T = @TypeOf(x);
//
//        const graph = x.ptr;
//
//        if (config.grads) {
//            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
//        }
//
//        const redux: ?[*]f32 = if (config.score != null) 
//            graph.tensor_allocator.create(f32, graph.stream) else null;
//
//        defer {
//            if (redux) |ptr| graph.tensor_allocator.destroy(ptr);
//        }
//
//        const score: ?[*]f32 = if (config.score)
//            |ptr| @ptrCast(@alignCast(ptr)) else null;
//
//        Loss.bce(graph.stream, x, y, redux, score);
//    }
//
//    pub fn contrast(
//        x: anytype, 
//        y: anytype, 
//        trg: types.Key,
//        config: struct {
//        p_margin: f32,
//        n_margin: f32,
//        x_grads: bool,
//        y_grads: bool,
//        score: ?*f32,
//    }) void {
//        const T = @TypeOf(x);
//        const U = @TypeOf(y);
//
//        const graph = x.ptr;
//
//        std.debug.assert(x.len() == y.len());
//
//        if (config.x_grads) 
//            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
//
//        if (config.y_grads) 
//            _ = enableGradient(graph, U.DataType, U.Class, y.idx);
//
//        const cpu_score: ?[*]f32 = if (config.score) 
//            |ptr| @ptrCast(@alignCast(ptr)) else null;
//
//        const gpu_score: ?[*]f32 = if (config.score != null) 
//            graph.tensor_allocator.create(f32, graph.stream) else null;
//
//        defer {
//            if (gpu_score) |ptr| graph.tensor_allocator.destroy(ptr);
//        }
//
//        Loss.contrast(
//            graph.stream, x, y, trg, config.p_margin, config.n_margin, cpu_score, gpu_score
//        );
//    }
//};

/////////////////////////////////////////////
/////////////////////////////////////////////

pub const algo = struct {

    const mem = @import("algo/mem.zig");
    pub const fill = mem.fill;
    pub const copy = mem.copy;
    pub const sequence = mem.sequence;

};

pub const ops = struct {
    pub const add = @import("ops/addition.zig").forward;
    pub const sub = @import("ops/subtraction.zig").forward;
    pub const hadamard = @import("ops/hadamard.zig").forward;
    pub const translate = @import("ops/translate.zig").forward;
    pub const dialate = @import("ops/dialate.zig").forward;

    // <>--------------------------------------------------------<>

    //pub fn leakyRelu(X: anytype, coef: f32) Tensor(Child(@TypeOf(X))) {
    //    if (comptime !isGraphTensor(@TypeOf(X)))
    //        @compileError("Leaky Relu requires graph tensor.");

    //    const graph = X.ptr;

    //    assert((0.0 <= coef) and (coef < 1.0));

    //    const Y = CG.nodeTensor(graph, X.sizes(), @TypeOf(X).DataType);

    //    TenOps.leakyReluForward(graph.stream, X, coef, Y);

    //    if (graph.mode == .train){
    //        _ = CG.adjustDependencies(X, 1);
    //        CG.appendNode(graph, TenOps.LeakyReluCallback, .{ X, coef, Y });
    //    }
    //    return Y;
    //}

    // <>--------------------------------------------------------<>

//    pub fn relu(X: anytype) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)))
//            @compileError("Relu requires graph tensor.");
//
//        return @call(.always_inline, leakyRelu, .{ X, 0.0 });
//    }
//
//    // <>--------------------------------------------------------<>
//
//    pub fn tanh(X: anytype) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)))
//            @compileError("Tanh requires graph tensor.");
//
//        return @call(.always_inline, activationDispatch, .{ TenOps.TanhCallback, X });
//    }
//
//    // <>--------------------------------------------------------<>
//
//    pub fn selu(X: anytype) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)))
//            @compileError("Selu requires graph tensor.");
//
//        return @call(.always_inline, activationDispatch, .{ TenOps.SeluCallback, X });
//    }
//
//    // <>--------------------------------------------------------<>
//
//    pub fn permutate(X: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)))
//            @compileError("Permutate requires graph tensor.");
//
//        const graph = X.ptr;
//
//        // tells us which size index to map from x to y
//        const map = comptime Parser.permutateSizes(expression);
//
//        std.debug.assert(X.sizes().len == map.len);
//
//        var y_sizes: [map.len]types.SizeType = undefined;
//
//        for (X.sizes(), 0..) |elem, i| {
//            y_sizes[map.perm[i]] = elem;
//        }
//
//        const Y = CG.nodeTensor(graph, y_sizes[0..], UT.Child(@TypeOf(X)));
//
//        const perm = TenOps.findPermutation(expression){};
//
//        perm.forward(graph.stream, X, Y);
//
//        if (graph.mode == .train){
//            _ = CG.adjustDependencies(X, 1);
//            CG.appendNode(graph, @TypeOf(perm), .{ X, Y });
//        }
//        return Y;
//    }
//
//    pub fn softmax(X: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)))
//            @compileError("Permutate requires graph tensor.");
//
//        const graph = X.ptr;
//
//        // tells us which size index to map from x to y
//        const Y = CG.nodeTensor(graph, X.sizes(), UT.Child(@TypeOf(X)));
//
//        const sm = TenOps.findSoftmax(expression){};
//
//        sm.forward(graph.stream, X, Y);
//
//        if (graph.mode == .train){
//            _ = CG.adjustDependencies(X, 1);
//            CG.appendNode(graph, @TypeOf(sm), .{ X, Y });
//        }
//        return Y;
//    }

    // <>--------------------------------------------------------<>

//    pub fn innerProduct(X: anytype, Y: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
//            @compileError("innerProduct requires graph tensors.");
//        return innerProductScaled(X, Y, 1.0, expression);
//    }
//
//    pub fn innerProductScaled(X: anytype, Y: anytype, alpha: f32, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
//            @compileError("innerProduct requires graph tensors.");
//
//        const graph = X.ptr;
//
//        // tells us which size index to map from x to y
//        const maps = comptime Parser.innerProductSizes(expression);
//
//        std.debug.assert(X.sizes().len == maps.x_map.len);
//        std.debug.assert(Y.sizes().len == maps.y_map.len);
//
//        var z_sizes: [maps.len]types.SizeType = undefined;
//
//        for (X.sizes(), 0..) |elem, i| {
//            if (maps.x_map[i]) |idx| {
//                z_sizes[idx] = elem;
//            }
//        }
//        for (Y.sizes(), 0..) |elem, i| {
//            if (maps.y_map[i]) |idx| {
//                z_sizes[idx] = elem;
//            }
//        }
//
//        const Z = CG.nodeTensor(graph, z_sizes[0..], UT.Child(@TypeOf(X)));
//
//        // locate inner product by expression
//        const ip = TenOps.findLinear(expression, false){};
//
//        // cancel out addition using 0.0 for beta
//        ip.forward(graph.stream, X, Y, alpha, Z, 0.0, Z);
//
//        if (graph.mode == .train) {
//            _ = CG.adjustDependencies(X, 1);
//            _ = CG.adjustDependencies(Y, 1);
//            CG.appendNode(graph, @TypeOf(ip), .{ X, Y, alpha, Z, 0.0, Z });   
//        }
//
//        return Z;
//    }

//    pub fn linear(X: anytype, Y: anytype, B: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        return linearScaled(X, Y, 1.0, B, expression);
//    }
//
//    pub fn linearScaled(X: anytype, Y: anytype, alpha: f32, B: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)) or !isGraphTensor(@TypeOf(B)))
//            @compileError("Linear requires graph tensors.");
//
//        const graph = X.ptr;
//
//        // tells us which size index to map from x to y
//        const maps = comptime Parser.innerProductSizes(expression);
//
//        std.debug.assert(X.sizes().len == maps.x_map.len);
//        std.debug.assert(Y.sizes().len == maps.y_map.len);
//
//        var z_sizes: [maps.len]types.SizeType = undefined;
//
//        for (X.sizes(), 0..) |elem, i| {
//            if (maps.x_map[i]) |idx| {
//                z_sizes[idx] = elem;
//            }
//        }
//        for (Y.sizes(), 0..) |elem, i| {
//            if (maps.y_map[i]) |idx| {
//                z_sizes[idx] = elem;
//            }
//        }
//
//        // bias needs to have the same dimensions as output
//        std.debug.assert(std.mem.eql(types.SizeType, z_sizes[0..], B.sizes()));
//
//        const Z = CG.nodeTensor(graph, z_sizes[0..], UT.Child(@TypeOf(X)));
//
//        // locate inner product by expression
//        const lin = TenOps.findLinear(expression, true){};
//
//        // cancel out addition using 0.0 for beta
//        lin.forward(graph.stream, X, Y, alpha, B, 1.0, Z);
//
//        if (graph.mode == .train) {
//            _ = CG.adjustDependencies(X, 1);
//            _ = CG.adjustDependencies(Y, 1);
//            _ = CG.adjustDependencies(B, 1);
//            CG.appendNode(graph, @TypeOf(lin), .{ X, Y, alpha, B, 1.0, Z });   
//        }
//        return Z;
//    }
//
//
//    pub fn reduce(X: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//
//        // TODO: extend this function to scalar output calls, need to address parser first
//
//        const graph = X.ptr;
//
//        const map = comptime Parser.reduceSizes(expression);
//
//        var y_sizes: [map.len]types.SizeType = undefined;
//
//        for (X.sizes(), 0..) |elem, i| {
//            if (map.x_map[i]) |idx| y_sizes[idx] = elem;
//        }
//
//        const Y = CG.nodeTensor(graph, y_sizes[0..], UT.Child(@TypeOf(X)));
//
//        const rdx = comptime TenOps.findReduce(expression){};
//
//        rdx.forward(graph.stream, X, Y);
//
//        if (graph.mode == .train) {
//            _ = CG.adjustDependencies(X, 1);
//            CG.appendNode(graph, @TypeOf(rdx), .{ X, Y }); 
//        }
//        return Y;
//    }
//
//    pub fn broadcast(X: anytype, ranks: []const types.SizeType, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//
//        // NOTE: see todo in expression parser above broadcasting function.
//
//        const graph = X.ptr;
//
//        const Y = CG.nodeTensor(graph, ranks, UT.Child(@TypeOf(X)));
//
//        const bcast = comptime TenOps.findBroadcast(expression){};
//
//        bcast.forward(graph.stream, X, Y);
//
//        if (graph.mode == .train) {
//            _ = CG.adjustDependencies(X, 1);
//            CG.appendNode(graph, @TypeOf(bcast), .{ X, Y }); 
//        }
//        return Y;
//    }
//
//    // TODO: Convolution isn't ready yet.
//
//    //pub fn convolution(
//    //    src: anytype,
//    //    krn: anytype,
//    //    comptime config: struct {
//    //        dims: usize,
//    //        channels: usize,
//    //        stride: usize,
//    //    },
//    //) Tensor(Child(@TypeOf(src))) {
//    //    
//    //    // These restrictions are temporary
//    //    if (config.dims != 2) {
//    //        @compileError("TODO: implement other dimensions besides 2");
//    //    }
//    //    if (config.channels != 1) {
//    //        @compileError("TODO: implement other channels besides 1");
//    //    }
//
//    //    const graph = src.ptr;
//
//    //    // this function will be a multi-level dispatcher
//    //    // right now it's just doing 2-D 1-channel
//    //    const src_sizes = src.sizes();
//    //    const krn_sizes = krn.sizes();
//
//    //    std.debug.assert(src_sizes.len == 2);
//    //    std.debug.assert(krn_sizes.len == 2);
//    //    std.debug.assert(krn_sizes[0] <= 32);
//
//    //    // kernel must be square and smaller than source
//    //    std.debug.assert(krn_sizes[0] == krn_sizes[1]);
//    //    std.debug.assert(krn_sizes[0] <= src_sizes[0]);
//    //    std.debug.assert(krn_sizes[0] <= src_sizes[1]);
//
//    //    const m = UT.windowCount(src_sizes[0], krn_sizes[0], config.stride);
//    //    const n = UT.windowCount(src_sizes[1], krn_sizes[0], config.stride);
//    //    const out = CG.nodeTensor(graph, &.{ m, n }, Child(@TypeOf(src)));
//    //    
//    //    const conv: TenOps.Convolution_2D_Callback = .{ };
//
//    //    conv.forward(out.stream(), src, krn, out, config.stride);
//    //    
//    //    if (graph.mode == .train) {
//    //        _ = CG.adjustDependencies(src, 1);
//    //        _ = CG.adjustDependencies(krn, 1);
//    //        CG.appendNode(graph, @TypeOf(conv), .{ src, krn, out, config.stride });   
//    //    }
//    //    return out;
//    //}
//
//    pub const norm = struct {
//
//        pub fn l2(X: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//            if (comptime !isGraphTensor(@TypeOf(X))){
//                @compileError("Linear requires graph tensors.");
//            }
//            const graph = X.ptr;
//    
//            const Y = CG.nodeTensor(graph, X.sizes(), UT.Child(@TypeOf(X)));
//
//            // locate inner product by expression
//            const l2_norm = TenOps.findNormL2(expression){};
//
//            // cancel out addition using 0.0 for beta
//            l2_norm.forward(X.stream(), X, Y);
//
//            if (graph.mode == .train) {
//                _ = CG.adjustDependencies(X, 1);
//                CG.appendNode(graph, @TypeOf(l2_norm), .{ X, Y }); 
//            }
//            return Y;
//        }
//
//        pub fn minmax(X: anytype, comptime expression: []const u8) Tensor(Child(@TypeOf(X))) {
//            if (comptime !isGraphTensor(@TypeOf(X))){
//                @compileError("Linear requires graph tensors.");
//            }
//            const graph = X.ptr;
//    
//            const Y = CG.nodeTensor(graph, X.sizes(), UT.Child(@TypeOf(X)));
//
//            // locate inner product by expression
//            const mm = TenOps.findMinmax(expression){};
//
//            // cancel out addition using 0.0 for beta
//            mm.forward(X.stream(), X, Y);
//
//            if (graph.mode == .train) {
//                _ = CG.adjustDependencies(X, 1);
//                CG.appendNode(graph, @TypeOf(mm), .{ X, Y }); 
//            }
//            return Y;
//        }
//    };
};


//pub const algo = struct {
//
//    // these functions require an output tensor because we do not
//    // want to encourage appending multiple copies to the graph. 
//    
//    pub const key = struct {
//        pub fn reduce(src: anytype, dst: anytype, keys: []const types.Key, comptime expression: []const u8) void {
//            reduceScaled(src, dst, keys, 1.0, expression);
//        }
//
//        pub fn reduceScaled(src: anytype, dst: anytype, keys: []const types.Key, alpha: f32, comptime expression: []const u8) void {
//
//            // TODO: extend this function to scalar output calls, need to address parser first
//
//            if (comptime !isGraphTensor(@TypeOf(src)) or !isGraphTensor(@TypeOf(dst)))
//                @compileError("reduce key requires graph tensors.");
//
//            Algo.callReduceKey(dst.stream(), src, dst, keys, alpha, expression);
//        }
//
//        pub fn sort(src: anytype, keys: []types.Key) void {
//            if (comptime !isGraphTensor(@TypeOf(src)))
//                @compileError("reduce key requires graph tensors.");
//            
//            Algo.callSortKey(src.stream(), src, keys);
//        }
//
//        pub fn max(src: anytype, keys: []types.Key, comptime expression: []const u8) void {
//            if (comptime !isGraphTensor(@TypeOf(src)))
//                @compileError("reduce key requires graph tensors.");
//            
//            Algo.callMaxKey(src.stream(), src, keys, expression);
//        }
//    };
//};

// This is here incase the user has a special need for direct
// operational access. This isn't the normal use case, but I don't
// believe in hiding this from the user as they may know something
// about their situation that I do not.

// Use at your own risk.

//pub const raw_ops = TenOps;

/////////////////////////////////////////////
/////////////////////////////////////////////
