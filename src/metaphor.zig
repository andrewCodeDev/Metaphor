const std = @import("std");
const assert = std.debug.assert;

const UT = @import("utility.zig");
const SC = @import("scalar.zig");
const TC = @import("tensor_components.zig");
const CG = @import("graph.zig");
const DU = @import("device_utils.zig");
const Parser = @import("expression_parsing.zig");

const appendNode = CG.appendNode;
const enableGradient = CG.enableGradient;
const nodeTensor = CG.nodeTensor;

const Contract = UT.Contract;
const Returns = UT.Returns;
const Child = UT.Child;

const NodeTensor = CG.NodeTensor;
const isGraphTensor = CG.isGraphTensor;

const Optimizer = @import("optimizer.zig");

const TenOps = @import("tensor_ops.zig");
const Loss = @import("loss.zig");

///////////////////////////////////////////////
///////////////////////////////////////////////

pub const device = struct {
    pub const init = DU.initDevice;
    pub const synchronize = DU.synchronizeDevice;
};

pub const stream = struct {
    pub const init = DU.initStream;
    pub const deinit = DU.deinitStream;
    pub const synchronize = DU.synchronizeStream;
    pub const Group = DU.StreamGroup;
};

pub const mem = struct {
    pub const copyToDevice = DU.copyToDevice;
    pub const copyFromDevice = DU.copyFromDevice;
    pub const alloc = DU.alloc;
    pub const create = DU.create;
    pub const free = DU.free;
    pub const fill = CG.fill;
    pub const sequence = TenOps.sequence;
    pub const randomize = TenOps.randomize;
};

pub const types = struct {
    pub const NodeTensor = CG.NodeTensor;
    pub const LeafTensor = CG.LeafTensor;
    pub const SliceUnion = TC.SliceUnion;
    pub const IndexType = TC.IndexType;
    pub const SizeType = TC.SizeType;
    pub const Strides = TC.Strides;
    pub const Sizes = TC.Sizes;
    pub const Stream = DU.Stream;
};

pub const scalar = struct {
    pub const r16 = SC.r16;
    pub const r32 = SC.r32;
    pub const r64 = SC.r64;
    pub const c16 = SC.c16;
    pub const c32 = SC.c32;
    pub const c64 = SC.c64;
    pub const Tag = SC.ScalarTag;
    pub const as = SC.asScalar;
};

pub const optm = struct {
    pub const SGD = Optimizer.SGD;
};

pub const loss = struct {
    pub fn cce(x: anytype, y: anytype, config: struct {
        grads: bool,
        score: ?*f64,
    }) void {
        const T = @TypeOf(x);

        const graph = x.ptr;

        if (config.grads) {
            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
        }
        const redux: ?[*]f64 = if (config.score != null)
            graph.tensor_allocator.allocScalar(f64, graph.stream)
        else
            null;
        defer {
            if (redux) |ptr| graph.tensor_allocator.freeScalar(ptr);
        }
        const score: ?[*]f64 = if (config.score) |ptr|
            @ptrCast(@alignCast(ptr))
        else
            null;

        Loss.cce(graph.stream, x, y, redux, score);
    }

    pub fn mse(x: anytype, y: anytype, config: struct {
        grads: bool,
        score: ?*f64,
    }) void {
        const T = @TypeOf(x);

        const graph = x.ptr;

        if (config.grads) {
            _ = enableGradient(graph, T.DataType, T.Class, x.idx);
        }

        const redux: ?[*]f64 = if (config.score != null) graph.tensor_allocator.allocScalar(f64, graph.stream) else null;
        defer {
            if (redux) |ptr| graph.tensor_allocator.freeScalar(ptr);
        }

        const score: ?[*]f64 = if (config.score) |ptr| @ptrCast(@alignCast(ptr)) else null;

        Loss.mse(graph.stream, x, y, redux, score);
    }
};

pub const Graph = CG.Graph;

pub fn Rank(comptime rank: usize) type {
    return [rank]types.SizeType;
}

/////////////////////////////////////////////
/////////////////////////////////////////////

pub const ops = struct {
    inline fn elementwiseDispatch(comptime Callback: type, X: anytype, Y: anytype) NodeTensor(SC.ScalarResult(@TypeOf(X), @TypeOf(Y))) {
        assert(std.mem.eql(types.SizeType, X.sizes(), Y.sizes()));
        const graph = X.ptr;
        const DataType = SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType);
        const Z = nodeTensor(graph, X.sizes(), DataType);
        const callback = Callback{}; // instance for comptime fields
        callback.forward(graph.stream, X, Y, Z);
        return if (graph.mode == .eval)
            Z
        else
            appendNode(graph, Callback, .{ X, Y }, Z);
    }

    inline fn activationDispatch(comptime Callback: type, X: anytype) NodeTensor(Child(@TypeOf(X))) {
        const graph = X.ptr;
        const Y = nodeTensor(graph, X.sizes(), @TypeOf(X).DataType);
        const callback = Callback{}; // instance for comptime fields
        callback.forward(graph.stream, X, Y);
        return if (graph.mode == .eval)
            Y
        else
            appendNode(graph, Callback, .{X}, Y);
    }

    // <>--------------------------------------------------------<>

    pub fn add(X: anytype, Y: anytype) NodeTensor(SC.ScalarResult(@TypeOf(X), @TypeOf(Y))) {
        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
            @compileError("Addition requires graph tensor.");

        return @call(.always_inline, elementwiseDispatch, .{ TenOps.AddCallback, X, Y });
    }

    // <>--------------------------------------------------------<>

    pub fn hadamard(X: anytype, Y: anytype) NodeTensor(SC.ScalarResult(@TypeOf(X), @TypeOf(Y))) {
        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
            @compileError("Hadamard requires graph tensor.");

        return @call(.always_inline, elementwiseDispatch, .{ TenOps.HadamardCallback, X, Y });
    }

    // <>--------------------------------------------------------<>

    pub fn subtract(X: anytype, Y: anytype) NodeTensor(SC.ScalarResult(@TypeOf(X), @TypeOf(Y))) {
        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
            @compileError("Subtract requires graph tensor.");

        return @call(.always_inline, elementwiseDispatch, .{ TenOps.SubCallback, X, Y });
    }

    // <>--------------------------------------------------------<>

    pub fn leakyRelu(X: anytype, coef: f16) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Leaky Relu requires graph tensor.");

        const graph = X.ptr;

        assert((0.0 <= coef) and (coef < 1.0));

        const Y = nodeTensor(graph, X.sizes(), @TypeOf(X).DataType);

        TenOps.leakyReluForward(graph.stream, X, coef, Y);

        return if (graph.mode == .eval) Y else appendNode(graph, TenOps.LeakyReluCallback, .{X}, Y);
    }

    // <>--------------------------------------------------------<>

    pub fn relu(X: anytype) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Relu requires graph tensor.");

        return @call(.always_inline, leakyRelu, .{ X, 0.0 });
    }

    // <>--------------------------------------------------------<>

    pub fn tanh(X: anytype) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Tanh requires graph tensor.");

        return @call(.always_inline, activationDispatch, .{ TenOps.TanhCallback, X });
    }

    // <>--------------------------------------------------------<>

    pub fn selu(X: anytype) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Selu requires graph tensor.");

        return @call(.always_inline, activationDispatch, .{ TenOps.SeluCallback, X });
    }

    // <>--------------------------------------------------------<>

    pub fn permutate(X: anytype, comptime expression: []const u8) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Permutate requires graph tensor.");

        const graph = X.ptr;

        // tells us which size index to map from x to y
        const map = comptime Parser.permutateSizes(expression);

        std.debug.assert(X.sizes().len == map.len);

        var y_sizes: [map.len]types.SizeType = undefined;

        for (X.sizes(), 0..) |elem, i| {
            y_sizes[map.perm[i]] = elem;
        }

        const Y = nodeTensor(graph, y_sizes[0..], UT.Child(@TypeOf(X)));

        const perm = TenOps.findPermutation(expression){};

        perm.forward(graph.stream, X, Y);

        return if (graph.mode == .eval) Y else appendNode(graph, @TypeOf(perm), .{X}, Y);
    }

    pub fn softmax(X: anytype, comptime expression: []const u8) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)))
            @compileError("Permutate requires graph tensor.");

        const graph = X.ptr;

        // tells us which size index to map from x to y
        const Y = nodeTensor(graph, X.sizes(), UT.Child(@TypeOf(X)));

        const sm = TenOps.findSoftmax(expression){};

        sm.forward(graph.stream, X, Y);

        return if (graph.mode == .eval) Y else appendNode(graph, @TypeOf(sm), .{X}, Y);
    }

    // <>--------------------------------------------------------<>

    pub fn innerProduct(X: anytype, Y: anytype, comptime expression: []const u8) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)))
            @compileError("innerProduct requires graph tensors.");

        const graph = X.ptr;

        // tells us which size index to map from x to y
        const maps = comptime Parser.innerProductSizes(expression);

        std.debug.assert(X.sizes().len == maps.x_map.len);
        std.debug.assert(Y.sizes().len == maps.y_map.len);

        var z_sizes: [maps.len]types.SizeType = undefined;

        for (X.sizes(), 0..) |elem, i| {
            if (maps.x_map[i]) |idx| {
                z_sizes[idx] = elem;
            }
        }
        for (Y.sizes(), 0..) |elem, i| {
            if (maps.y_map[i]) |idx| {
                z_sizes[idx] = elem;
            }
        }

        const Z = nodeTensor(graph, z_sizes[0..], UT.Child(@TypeOf(X)));

        // locate inner product by expression
        const ip = TenOps.findLinear(expression, false){};

        // cancel out addition using 0.0 for beta
        ip.forward(graph.stream, X, Y, 1.0, Z, 0.0, Z);

        return if (graph.mode == .eval) Z else appendNode(graph, @TypeOf(ip), .{ X, Y, 1.0, Z, 0.0 }, Z);
    }

    pub fn linear(X: anytype, Y: anytype, B: anytype, comptime expression: []const u8) NodeTensor(Child(@TypeOf(X))) {
        return linearScaled(X, Y, 1.0, B, expression);
    }

    pub fn linearScaled(X: anytype, Y: anytype, alpha: f16, B: anytype, comptime expression: []const u8) NodeTensor(Child(@TypeOf(X))) {
        if (comptime !isGraphTensor(@TypeOf(X)) or !isGraphTensor(@TypeOf(Y)) or !isGraphTensor(@TypeOf(B)))
            @compileError("Linear requires graph tensors.");

        const graph = X.ptr;

        // tells us which size index to map from x to y
        const maps = comptime Parser.innerProductSizes(expression);

        std.debug.assert(X.sizes().len == maps.x_map.len);
        std.debug.assert(Y.sizes().len == maps.y_map.len);

        var z_sizes: [maps.len]types.SizeType = undefined;

        for (X.sizes(), 0..) |elem, i| {
            if (maps.x_map[i]) |idx| {
                z_sizes[idx] = elem;
            }
        }
        for (Y.sizes(), 0..) |elem, i| {
            if (maps.y_map[i]) |idx| {
                z_sizes[idx] = elem;
            }
        }

        // bias needs to have the same dimensions as output
        std.debug.assert(std.mem.eql(types.SizeType, z_sizes[0..], B.sizes()));

        const Z = nodeTensor(graph, z_sizes[0..], UT.Child(@TypeOf(X)));

        // locate inner product by expression
        const lin = TenOps.findLinear(expression, true){};

        // cancel out addition using 0.0 for beta
        lin.forward(graph.stream, X, Y, alpha, B, 1.0, Z);

        return if (graph.mode == .eval) Z else appendNode(graph, @TypeOf(lin), .{ X, Y, alpha, B, 1.0 }, Z);
    }
};

/////////////////////////////////////////////
/////////////////////////////////////////////
