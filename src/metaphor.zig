
const std = @import("std");
const assert = std.debug.assert;

const UT = @import("utility.zig");
const SC = @import("scalar.zig");
const TC = @import("tensor_components.zig");
const CG = @import("graph.zig");
const DU = @import("device_utils.zig");

const Contract = UT.Contract;
const Returns = UT.Returns;
const Child = UT.Child;

const NodeTensor = CG.NodeTensor;
const isGraphTensor = CG.isGraphTensor;

const Optimizer = @import("optimizer.zig");

const TenOps = @import("tensor_ops.zig");

//const Scale = @import("scale.zig");

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
};

pub const mem = struct {
    pub const copyToDevice = DU.copyToDevice;
    pub const copyFromDevice = DU.copyFromDevice;
    pub const alloc = DU.alloc;
    pub const create = DU.create;
    pub const free = DU.free;
};

pub const types = struct {
    pub const r16 = SC.r16;
    pub const r32 = SC.r32;
    pub const r64 = SC.r64;
    pub const c16 = SC.c16;
    pub const c32 = SC.c32;
    pub const c64 = SC.c64;
    pub const SliceUnion = TC.SliceUnion;
    pub const IndexType = TC.IndexType;
    pub const SizeType = TC.SizeType;
    pub const Strides = TC.Strides;
    pub const Sizes = TC.Sizes;
};

pub const Graph = CG.Graph;

pub const null_optimizer = Optimizer.NullOptimizer.optimizer();

pub fn Dims(comptime Rank: usize) type {
    return [Rank]types.SizeType;
}

 /////////////////////////////////////////////
/////////////////////////////////////////////

pub const ops = struct {

    pub const fill = CG.fill;

    inline fn elementwiseDispatch(comptime Impl: type, X: anytype, Y: anytype)
        NodeTensor(SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType)) {
        assert(std.mem.eql(types.SizeType, X.sizes(), Y.sizes()));
        const graph = X.ptr;
        const DataType = SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType);
        const Z = graph.nodeTensor(X.sizes(), DataType);        
        const callback = Impl{ };  // instance for comptime fields
        callback.forward(graph.stream, X, Y, Z);
        return graph.appendNode(Impl, .{ graph.stream, X, Y }, Z);
    }

    inline fn activationDispatch(comptime Impl: type, X: anytype)
        NodeTensor(@TypeOf(X).DataType) {
        const graph = X.ptr;
        const Y = graph.nodeTensor(X.sizes(), @TypeOf(X).DataType);        
        const callback = Impl{ };  // instance for comptime fields
        callback.forward(graph.stream, X, Y);
        return graph.appendNode(Impl, .{ graph.stream, X }, Y);
    }

    pub fn add(X: anytype, Y: anytype) Contract(
            isGraphTensor(@TypeOf(X)) and
            isGraphTensor(@TypeOf(Y)),
        NodeTensor(SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType))) {
            return @call(.always_inline, elementwiseDispatch, .{ TenOps.AddImpl, X, Y});
        }
    
    pub fn hadamard(X: anytype, Y: anytype) Contract(
            isGraphTensor(@TypeOf(X)) and
            isGraphTensor(@TypeOf(Y)),
        NodeTensor(SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType))) {
            return @call(.always_inline, elementwiseDispatch, .{ TenOps.HadamardImpl, X, Y});
        }
    
    pub fn subtract(X: anytype, Y: anytype) Contract(
            isGraphTensor(@TypeOf(X)) and
            isGraphTensor(@TypeOf(Y)),
        NodeTensor(SC.ScalarResult(@TypeOf(X).DataType, @TypeOf(Y).DataType))) {
            return @call(.always_inline, elementwiseDispatch, .{ TenOps.SubImpl, X, Y});    
        }
    
    pub fn leakyRelu(X: anytype, coef: anytype) Contract(
            isGraphTensor(@TypeOf(X)) and SC.isFloat(@TypeOf(coef)), 
        Returns(@TypeOf(X))) {
        assert((0.0 <= coef) and (coef < 1.0));
        const graph = X.ptr;
        const Y = graph.nodeTensor(X.sizes(), @TypeOf(X).DataType);        
        TenOps.leakyReluForward(graph.stream, X, coef, Y);
        return graph.appendNode(TenOps.LeakyReluImpl, .{ graph.stream, X, coef }, Y);        
    }

    pub fn relu(X: anytype) Contract(
            isGraphTensor(@TypeOf(X)), 
        Returns(@TypeOf(X))) {
            return @call(.always_inline, leakyRelu, .{ X, 0.0 });
        }    

    pub fn tanh(X: anytype) Contract(
            isGraphTensor(@TypeOf(X)), 
        Returns(@TypeOf(X))) {
            return @call(.always_inline, activationDispatch, .{ TenOps.TanhImpl, X, 0.0 });
        }    
};


 /////////////////////////////////////////////
/////////////////////////////////////////////

