
const std = @import("std");
const math = std.math;
const SC = @import("scalar.zig");
const UT = @import("utility.zig");
const CB = @import("callback_builder.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");
const Child = UT.Child;

const CallbackBuilder = CB.CallbackBuilder;
const NoCleanup = CB.NoCleanup;
const overloads = @import("kernel_overloads.zig");
const Parser = @import("expression_parsing.zig");
const Stream = DU.Stream;

// <>--------------------------------------------------------<>

pub fn additionForward(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_addition.call(.{
        stream.context, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn additionReverseArg0(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream.context, x_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn additionReverseArg1(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream.context, y_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub const AddImpl = CallbackBuilder(
    additionForward, .{
        .{ additionReverseArg0, 0 },
        .{ additionReverseArg1, 1 }
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn subtractionForward(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_subtraction.call(.{
        stream.context, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn subtractionReverseArg0(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    // scalar is positive for left-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), 1.0);
    
    overloads.kernel_subtraction_reverse.call(.{
        stream.context, x_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub fn subtractionReverseArg1(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    // scalar is negative for right-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), -1.0);

    overloads.kernel_subtraction_reverse.call(.{
        stream.context, y_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub const SubImpl = CallbackBuilder(
    subtractionForward, .{
        .{ subtractionReverseArg0, 0 },
        .{ subtractionReverseArg1, 1 }
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn hadamardForward(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_hadamard.call(.{
        stream.context, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn hadamardReverseArg0(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const y_value = Y.values();
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream.context, x_grads.ptr, y_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn hadamardReverseArg1(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_value = X.values();
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream.context, y_grads.ptr, x_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub const HadamardImpl = CallbackBuilder(
    hadamardForward, .{
        .{ hadamardReverseArg0, 0 },
        .{ hadamardReverseArg1, 1 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn leakyReluForward(stream: Stream, x: anytype, coef: anytype, y: anytype) void {
    const T = Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_leaky_relu.call(.{
        stream.context,
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, coef), 
        y_values.len
    });
}

pub fn leakyReluReverse(stream: Stream, x: anytype, coef: anytype, y: anytype) void {
    const T = Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_leaky_relu_reverse.call(.{
        stream.context,
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, coef), 
        y_values.len
    });
}

pub const LeakyReluImpl = CallbackBuilder(
    leakyReluForward, .{
        .{ leakyReluReverse, 0 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn tanhForward(stream: Stream, x: anytype, y: anytype) void {
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_tanh.call(.{
        stream.context, x_values.ptr,  y_values.ptr,  y_values.len
    });
}

pub fn tanhReverse(stream: Stream, x: anytype, y: anytype) void {
    const x_grads = x.values();
    const y_values = y.values();
    const y_grads = UT.assertGrads(y);

    overloads.kernel_tanh_reverse.call(.{
        stream.context, x_grads.ptr, y_values.ptr, y_grads.ptr, y_values.len
    });
}

pub const TanhImpl = CallbackBuilder(
    tanhForward, .{
        .{ tanhReverse, 0 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn sequence(
    tensor: anytype, 
    init: anytype,
    step: anytype
) void {
    const T = UT.Child(@TypeOf(tensor));
    const _init = SC.asScalar(T, init);
    const _step = SC.asScalar(T, step);
    const values = tensor.values();

    overloads.kernel_sequence.call(.{
       tensor.ptr.stream.context, values.ptr, _init, _step, values.len 
    });
}

// <>--------------------------------------------------------<>

pub fn randomize(
    x: anytype,
    stream: DU.Stream
) void {

    //TODO: replace this with a kernel call
    
    var backing = std.rand.DefaultPrng.init(42);

    var random = backing.random();
    
    const mem = std.heap.c_allocator.alloc(@TypeOf(x).DataType, x.len())
        catch @panic("randomize out of memory");

        defer std.heap.c_allocator.free(mem);

    for (0..x.len()) |i|
        mem[i] = random.float(@TypeOf(x).DataType);

    DU.copyToDevice(mem, x.values(), stream);

    DU.synchronizeStream(stream);
}
// <>--------------------------------------------------------<>

pub inline fn permutate_ij_ji(
    stream: Stream, 
    x: anytype, 
    y: anytype,
) void {
    const T = UT.Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();
    const x_sizes = x.sizes();
    overloads.kernel_permutate_ij_ji.call(.{
       stream.context, x_values.ptr, y_values.ptr, SC.asScalar(T, 0.0), x_sizes[0], x_sizes[1] 
    });
}

pub inline fn permutate_ij_ji_Reverse(
    stream: Stream, 
    x: anytype, 
    y: anytype,
) void {
    const T = UT.Child(@TypeOf(x));
    const x_grads = x.grads().?;
    const y_grads = y.grads().?;
    const y_sizes = y.sizes();
    overloads.kernel_permutate_ij_ji.call(.{
       stream.context, y_grads.ptr, x_grads.ptr, SC.asScalar(T, 1.0), y_sizes[0], y_sizes[1] 
    });
}

const Perm_ij_ji_Impl = CallbackBuilder(
    permutate_ij_ji, .{ .{ permutate_ij_ji, 0 } }, NoCleanup
);

const permutation_expressions = std.ComptimeStringMap(
    type, .{
        .{ "ij->ji", Perm_ij_ji_Impl },
        .{ "ji->ij", Perm_ij_ji_Impl },
    }
);

pub fn findPermutation(comptime expression: []const u8) type {
    const parsed = comptime Parser.permutationExpression(expression);
    if (permutation_expressions.get(parsed)) |perm| {
        return perm;
    } else {
        @compileError("TODO: Declare General Permutation Kernel: " ++ expression);
    }
}

// <>--------------------------------------------------------<>

//////////////////////////////
// ---- matrix-to-vector -----


inline fn __innerProduct_ij_j(
    stream: Stream,
    x_values: anytype,
    y_values: anytype,
    alpha: f16,
    z_values: anytype,
    beta: f16,
    m: TC.SizeType,
    n: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(z_values));
    
    overloads.kernel_inner_product_ij_j.call(.{
        stream.context, 
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, alpha),
        z_values.ptr,
        SC.asScalar(T, beta), 
        m, n
    });
}

pub inline fn innerProduct_ij_j(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {

    std.debug.assert(x.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);
    std.debug.assert(z.sizes().len == 1);

    const x_sizes = x.sizes();

    std.debug.assert(x_sizes[0] == z.len());
    std.debug.assert(x_sizes[1] == y.len());

    __innerProduct_ij_j(
        stream,
        x.values(),
        y.values(),
        1.0, // alpha
        z.values(),
        0.0, //beta
        x_sizes[0],
        x_sizes[1],
    );
}

pub inline fn innerProduct_ij_j_ReverseArg0(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    // outer product i,j to get ij
    std.debug.assert(x.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);
    std.debug.assert(z.sizes().len == 1);

    const x_sizes = x.sizes();

    std.debug.assert(x_sizes[0] == z.len());
    std.debug.assert(x_sizes[1] == y.len());

    __outerProduct_i_j(
        stream,
        z.grads().?,
        y.values(),
        1.0, // alpha
        x.grads().?,
        1.0, //beta
        x_sizes[0],
        x_sizes[1],
    );
}

pub inline fn innerProduct_ij_j_ReverseArg1(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    // inner product i,ij to get j
    std.debug.assert(x.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);
    std.debug.assert(z.sizes().len == 1);

    const x_sizes = x.sizes();

    std.debug.assert(x_sizes[0] == z.len());
    std.debug.assert(x_sizes[1] == y.len());

    __innerProduct_i_ij(
        stream,
        z.grads().?,
        x.values(),
        1.0, // alpha
        y.grads().?,
        0.0, //beta
        x_sizes[0],
        x_sizes[1],
    );
}

inline fn __innerProduct_i_ij(
    stream: Stream,
    x_values: anytype,
    y_values: anytype,
    alpha: f16,
    z_values: anytype,
    beta: f16,
    m: TC.SizeType,
    n: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(z_values));
    
    overloads.kernel_inner_product_i_ij.call(.{
        stream.context, 
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, alpha),
        z_values.ptr,
        SC.asScalar(T, beta), 
        m, n
    });
}

const IP_i_ij_Impl = CallbackBuilder(
    innerProduct_i_ij, .{
        .{ innerProduct_i_ij_ReverseArg0, 0 },   
        .{ innerProduct_i_ij_ReverseArg1, 1 },   
    }, NoCleanup
);

pub inline fn innerProduct_i_ij(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {

    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 2);
    std.debug.assert(z.sizes().len == 1);

    const y_sizes = y.sizes();

    std.debug.assert(y_sizes[0] == x.len());
    std.debug.assert(y_sizes[1] == z.len());

    __innerProduct_ij_j(
        stream,
        x.values(),
        y.values(),
        1.0, // alpha
        z.values(),
        1.0, //beta
        y_sizes[0],
        y_sizes[1],
    );
}


pub inline fn innerProduct_i_ij_ReverseArg0(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    // inner product ij,j to get i
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 2);
    std.debug.assert(z.sizes().len == 1);

    const y_sizes = y.sizes();

    std.debug.assert(y_sizes[0] == x.len());
    std.debug.assert(y_sizes[1] == z.len());

    __innerProduct_ij_j(
        stream,
        y.values(),
        z.grads().?,
        1.0, // alpha
        x.grads().?,
        0.0, //beta
        y_sizes[0],
        y_sizes[1],
    );
}

pub inline fn innerProduct_i_ij_ReverseArg1(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    // outer product i,j to get ij
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 2);
    std.debug.assert(z.sizes().len == 1);

    const y_sizes = y.sizes();

    std.debug.assert(y_sizes[0] == x.len());
    std.debug.assert(y_sizes[1] == z.len());

    _ = &stream;

    __outerProduct_i_j(
        stream,
        x.values(),
        z.grads().?,
        1.0, // alpha
        y.grads().?,
        1.0, //beta
        y_sizes[0],
        y_sizes[1],
    );
}

const IP_ij_j_Impl = CallbackBuilder(
    innerProduct_ij_j, .{
        .{ innerProduct_ij_j_ReverseArg0, 0 },
        .{ innerProduct_ij_j_ReverseArg1, 1 },
    }, NoCleanup
);

pub fn innerProduct_i_ji(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_ij_j(stream, y, x, z);
}
pub fn innerProduct_i_ji_ReverseArg0(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_ij_j_ReverseArg0(stream, y, x, z);
}
pub fn innerProduct_i_ji_ReverseArg1(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_ij_j_ReverseArg1(stream, y, x, z);
}
const IP_i_ji_Impl = CallbackBuilder(
    innerProduct_i_ji, .{
        .{ innerProduct_i_ji_ReverseArg0, 0 },
        .{ innerProduct_i_ji_ReverseArg1, 1 },
    }, NoCleanup
);

pub fn innerProduct_ij_i(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_i_ij(stream, y, x, z);
}
pub fn innerProduct_ij_i_ReverseArg0(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_i_ij_ReverseArg0(stream, y, x, z);
}
pub fn innerProduct_ij_i_ReverseArg1(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    return innerProduct_i_ij_ReverseArg1(stream, y, x, z);
}
const IP_ij_i_Impl = CallbackBuilder(
    innerProduct_ij_i, .{
        .{ innerProduct_ij_i_ReverseArg0, 0 },
        .{ innerProduct_ij_i_ReverseArg1, 1 },
    }, NoCleanup
);
//////////////////////////////
// ---- matrix-to-matrix -----

inline fn __innerProduct_ij_jk(
    stream: Stream,
    x_values: anytype,
    y_values: anytype,
    alpha: f16,
    z_values: anytype,
    beta: f16,
    m: TC.SizeType,
    n: TC.SizeType,
    k: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(z_values));
    
    overloads.kernel_matmul_2D.call(.{
        stream.context, 
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, alpha),
        z_values.ptr,
        SC.asScalar(T, beta), 
        m, n, k
    });
}

pub inline fn innerProduct_ij_jk(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    const x_sizes = x.sizes();
    const y_sizes = y.sizes();

    std.debug.assert(x_sizes.len == 2);
    std.debug.assert(y_sizes.len == 2);
    std.debug.assert(x_sizes[1] == y_sizes[0]);

    __innerProduct_ij_jk(
        stream,
        x.values(),
        y.values(),
        1.0, // alpha
        z.values(),
        0.0, //beta
        x_sizes[0],
        x_sizes[1],
        y_sizes[1],
    );
}

inline fn innerProduct_ij_jk_ReverseArg0(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    const T = Child(@TypeOf(x));
    const x_grads = x.grads().?;
    const y_values = y.values();
    const z_grads = z.grads().?;
    const z_sizes = z.sizes();
    const y_sizes = y.sizes();

    const y_tran = stream.getScratch(T, y_values.len);

    overloads.kernel_permutate_ij_ji.call(.{
        stream.context, 
        y_values.ptr, 
        y_tran.ptr, 
        SC.asScalar(T, 0.0), 
        y_sizes[0], 
        y_sizes[1] 
    });

    __innerProduct_ij_jk(
        stream, 
        z_grads,
        y_tran,
        1.0, // alpha
        x_grads,
        1.0, // beta
        z_sizes[0],
        z_sizes[1],
        y_sizes[0],
    );
}

inline fn innerProduct_ij_jk_ReverseArg1(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    const T = Child(@TypeOf(x));

    const x_values = x.values();
    const y_grads = y.grads().?;
    const z_grads = z.grads().?;

    const z_sizes = z.sizes();
    const x_sizes = x.sizes();

    const x_tran = stream.getScratch(T, x_values.len);

    overloads.kernel_permutate_ij_ji.call(.{
        stream.context, 
        x_values.ptr, 
        x_tran.ptr, 
        SC.asScalar(T, 0.0), 
        x_sizes[0], 
        x_sizes[1] 
    });

    __innerProduct_ij_jk(
        stream, 
        x_tran,
        z_grads,
        1.0, // alpha
        y_grads,
        1.0, // beta
        x_sizes[1],
        x_sizes[0],
        z_sizes[1],
    );
}

const IP_ij_jk_Impl = CallbackBuilder(
    innerProduct_ij_jk, .{
        .{ innerProduct_ij_jk_ReverseArg0, 0 },
        .{ innerProduct_ij_jk_ReverseArg1, 1 },
    }, NoCleanup,
);

const inner_product_expressions = std.ComptimeStringMap(
    type, .{
        // Rank-1-to-Rank-2
        .{ "i,ij->j", IP_i_ij_Impl },
        .{ "i,ji->j", IP_i_ji_Impl },
        .{ "j,ij->i", IP_i_ji_Impl },
        .{ "j,ji->i", IP_i_ij_Impl },

        // Rank-2-to-Rank-1
        .{ "ij,j->i", IP_ij_j_Impl },
        .{ "ij,i->j", IP_ij_i_Impl },
        .{ "ji,i->j", IP_ij_j_Impl },
        .{ "ji,j->i", IP_ij_i_Impl },

        // Rank-2-to-Rank-2
        .{ "ij,jk->ik", IP_ij_jk_Impl }
    }
);

pub fn findInnerProduct(comptime expression: []const u8) type {
    const parsed = comptime Parser.innerProductExpression(expression);
    if (inner_product_expressions.get(parsed)) |ip| {
        return ip;
    } else {
        @compileError("TODO: Declare General Inner Product Kernel: " ++ expression);            
    }
}

//////////////////////////////
// outer product vector-vector

inline fn __outerProduct_i_j(
    stream: Stream,
    x_values: anytype,
    y_values: anytype,
    alpha: f16,
    z_values: anytype,
    beta: f16,
    m: TC.SizeType,
    n: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(z_values));
    
    overloads.kernel_outer_product_i_j.call(.{
        stream.context, 
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, alpha),
        z_values.ptr,
        SC.asScalar(T, beta), 
        m, n
    });
}

pub inline fn outerProduct_i_j(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,
) void {
    const z_sizes = z.sizes();

    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 1);
    std.debug.assert(z_sizes[0] == x.len());
    std.debug.assert(z_sizes[1] == y.len());

    __outerProduct_i_j(
        stream,
        x.values(),
        y.values(),
        1.0, // alpha
        z.values(),
        0.0, //beta
        z_sizes[0],
        z_sizes[1],
    );
}
