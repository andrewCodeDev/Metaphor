const std = @import("std");
const math = std.math;
const SC = @import("scalar.zig");
const UT = @import("utility.zig");
const CB = @import("callback_builder.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");
const Child = UT.Child;

const CallbackBuilder = CB.CallbackBuilder;
const CallbackDropReverse = CB.CallbackDropReverse;
const NoCleanup = CB.NoCleanup;
const overloads = @import("kernel_overloads.zig");
const Parser = @import("expression_parsing.zig");
const Stream = DU.Stream;
const NoArg = CB.NoArg;

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

pub const AddCallback = CallbackBuilder(
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

pub const SubCallback = CallbackBuilder(
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

pub const HadamardCallback = CallbackBuilder(
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

pub const LeakyReluCallback = CallbackBuilder(
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

pub const TanhCallback = CallbackBuilder(
    tanhForward, .{
        .{ tanhReverse, 0 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn seluForward(stream: Stream, x: anytype, y: anytype) void {
    overloads.kernel_selu.call(.{
        stream.context, x.values().ptr, y.values().ptr, y.len()
    });
}

pub fn seluReverse(stream: Stream, x: anytype, y: anytype) void {
    overloads.kernel_selu_reverse.call(.{
        stream.context, x.grads().?.ptr, y.values().ptr, y.grads().?.ptr, y.len()
    });
}

pub const SeluCallback = CallbackBuilder(
    seluForward, .{
        .{ seluReverse, 0 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn softmaxForward_i_i(stream: Stream, x: anytype, y: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const T = @TypeOf(x);

    std.debug.assert(x_values.len == y_values.len);
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 1);

    // make a proper utility for this
    const n = ((x_values.len / (32 * 32 * 4)) + 2);

    const scratch = stream.getScratch(Child(T), n);

    overloads.kernel_softmax_i_i.call(.{
        stream.context, x_values.ptr, y_values.ptr, scratch.ptr, y_values.len
    });
}


pub fn logisticReverse(stream: Stream, x: anytype, y: anytype) void {
    const x_grads = x.values();
    const y_values = y.values();
    const y_grads = UT.assertGrads(y);

    overloads.kernel_logistic_reverse.call(.{
        stream.context, x_grads.ptr, y_values.ptr, y_grads.ptr, y_values.len
    });
}

pub const SM_i_i_Callback = CallbackBuilder(
    softmaxForward_i_i, .{
        .{ logisticReverse, 0 },
    }, NoCleanup
);

const softmax_expressions = std.ComptimeStringMap(
    type, .{
        // Rank-1-to-Rank-2
        .{ "i|i", SM_i_i_Callback },
    }
);

pub fn findSoftmax(comptime expression: []const u8) type {
    const parsed = comptime Parser.softmaxExpression(expression);
    if (softmax_expressions.get(parsed)) |sm| {
        return sm;
    } else {
        @compileError("TODO: invalid softmax expression:" ++ expression);
    }
}
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
) void {

    //TODO: replace this with a kernel call...?
    //      really though, how often is this called?
    
    var backing = std.rand.DefaultPrng.init(42);

    var random = backing.random();
    
    const mem = std.heap.c_allocator.alloc(@TypeOf(x).DataType, x.len())
        catch @panic("randomize out of memory");

        defer std.heap.c_allocator.free(mem);

    for (0..x.len()) |i|
        mem[i] = random.float(@TypeOf(x).DataType);

    DU.copyToDevice(mem, x.values(), x.ptr.stream);

    DU.synchronizeStream(x.ptr.stream);
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

const Perm_ij_ji_Callback = CallbackBuilder(
    permutate_ij_ji, .{ .{ permutate_ij_ji, 0 } }, NoCleanup
);

const permutation_expressions = std.ComptimeStringMap(
    type, .{
        .{ "ij->ji", Perm_ij_ji_Callback },
        .{ "ji->ij", Perm_ij_ji_Callback },
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

// to drastically simplify our lives, we can treat all inner products
// as a special case of linear where the bias is skipped. To signal
// this in the reversal process, we drop the bias reversal from the
// linear callback type and pass it a zero beta value.

pub fn LinearType(comptime callback: type, comptime bias: bool) type {
    return if (comptime bias) callback else CallbackDropReverse(callback, 3);
}

//////////////////////////////
// ---- matrix-to-vector -----

inline fn __linear_i_ij(
        stream: Stream,
    x_values: anytype,
    A_values: anytype,
        alpha: f16,
    b_values: anytype,
        beta: f16,
    y_values: anytype,
    m: TC.SizeType,
    n: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(y_values));
    
    overloads.kernel_linear_i_ij.call(.{
            stream.context, 
        x_values.ptr, 
        A_values.ptr, 
            SC.asScalar(T, alpha),
        b_values.ptr,
            SC.asScalar(T, beta),
        y_values.ptr,
        m, n
    });
}

inline fn __linear_ij_j(
        stream: Stream,
    A_values: anytype,
    x_values: anytype,
        alpha: f16,
    b_values: anytype,
        beta: f16,
    y_values: anytype,
        m: TC.SizeType,
        n: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(y_values));
    
    overloads.kernel_linear_ij_j.call(.{
            stream.context, 
        A_values.ptr, 
        x_values.ptr, 
            SC.asScalar(T, alpha),
        b_values.ptr,
            SC.asScalar(T, beta), 
        y_values.ptr,
        m, n
    });
}

pub fn linear_ij_j(
    stream: Stream, 
    A: anytype, 
    x: anytype,
    alpha: f16,
    b: anytype,
    beta: f16,
    y: anytype,
) void {

    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(b.sizes().len == 1);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(n == x.len());
    std.debug.assert(m == b.len());
    std.debug.assert(m == y.len());

    __linear_ij_j(
        stream, A.values(), x.values(), alpha, b.values(), beta, y.values(), m, n
    );
}

pub fn linear_ij_j_ReverseArg0(
    stream: Stream, 
    A: anytype, 
    x: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    y: anytype,
) void {
    // outer product i,j to get ij
    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(m == x.len());
    std.debug.assert(n == y.len());

    __outerProduct_i_j(
        stream, y.grads().?, x.values(), alpha, A.grads().?, 1.0, m, n
    );
}

pub fn linear_ij_j_ReverseArg1(
    stream: Stream, 
    A: anytype, 
    x: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    y: anytype,
) void {
    // inner product i,ij to get j
    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(m == y.len());
    std.debug.assert(n == x.len());

    __linear_i_ij(
        stream, y.grads().?, A.values(), alpha, x.grads().?, 1.0, x.grads().?, m, n
    );
}

pub inline fn linear_bias_ReverseArg3(
    stream: Stream, 
    _: anytype, 
    _: anytype,
    _: f16,
    b: anytype,
    _: f16,
    y: anytype,
) void {
    additionReverseArg0(stream, b, NoArg, y);
}

const Linear_ij_j_Callback = CallbackBuilder(
    linear_ij_j, .{
        .{ linear_ij_j_ReverseArg0, 0 },   
        .{ linear_ij_j_ReverseArg1, 1 },   
        .{ linear_bias_ReverseArg3, 3 },   
    }, NoCleanup
);

pub fn linear_i_ij(
    stream: Stream, 
    x: anytype, 
    A: anytype,
    alpha: f16,
    b: anytype,
    beta: f16,
    y: anytype,
) void {
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(m == x.len());
    std.debug.assert(n == y.len());

    __linear_i_ij(
        stream, x.values(), A.values(), alpha, b.values(), beta, y.values(), m, n
    );
}

pub fn linear_i_ij_ReverseArg0(
    stream: Stream, 
    x: anytype, 
    A: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    y: anytype,
) void {
    // inner product ij,j to get i
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(m == x.len());
    std.debug.assert(n == y.len());

    __linear_ij_j(
        stream, A.values(), y.grads().?, alpha, x.grads().?, 1.0, x.grads().?, m, n,
    );
}

pub fn linear_i_ij_ReverseArg1(
    stream: Stream, 
    x: anytype, 
    A: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    y: anytype,
) void {
    // outer product i,j to get ij
    std.debug.assert(x.sizes().len == 1);
    std.debug.assert(A.sizes().len == 2);
    std.debug.assert(y.sizes().len == 1);

    const A_sizes = A.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];

    std.debug.assert(m == x.len());
    std.debug.assert(n == y.len());

    __outerProduct_i_j(
        stream, x.values(), y.grads().?, alpha, A.grads().?, 1.0, m, n
    );
}

const Linear_i_ij_Callback = CallbackBuilder(
    linear_i_ij, .{
        .{ linear_i_ij_ReverseArg0, 0 },   
        .{ linear_i_ij_ReverseArg1, 1 },   
        .{ linear_bias_ReverseArg3, 3 },   
    }, NoCleanup
);

pub fn linear_i_ji(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_ij_j(stream, A, x, alpha, b, beta, y);
}
pub fn linear_i_ji_ReverseArg0(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_ij_j_ReverseArg0(stream, A, x, alpha, b, beta, y);
}
pub fn linear_i_ji_ReverseArg1(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_ij_j_ReverseArg1(stream, A, x, alpha, b, beta, y);
}
const Linear_i_ji_Callback = CallbackBuilder(
    linear_i_ji, .{
        .{ linear_i_ji_ReverseArg0, 0 },
        .{ linear_i_ji_ReverseArg1, 1 },
        .{ linear_bias_ReverseArg3, 3 },
    }, NoCleanup
);

pub fn linear_ij_i(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_i_ij(stream, A, x, alpha, b, beta, y);
}
pub fn linear_ij_i_ReverseArg0(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_i_ij_ReverseArg0(stream, A, x, alpha, b, beta, y);
}
pub fn linear_ij_i_ReverseArg1(stream: Stream, x: anytype, A: anytype, alpha: f16, b: anytype, beta: f16, y: anytype) void {
    return linear_i_ij_ReverseArg1(stream, A, x, alpha, b, beta, y);
}

const Linear_ij_i_Callback = CallbackBuilder(
    linear_ij_i, .{
        .{ linear_ij_i_ReverseArg0, 0 },
        .{ linear_ij_i_ReverseArg1, 1 },
        .{ linear_bias_ReverseArg3, 3 },
    }, NoCleanup
);

//////////////////////////////
// ---- matrix-to-matrix -----

inline fn __linear_ij_jk(
    stream: Stream,
    A_values: anytype,
    B_values: anytype,
    alpha: f16,
    C_values: anytype,
    beta: f16,
    Y_values: anytype,
    m: TC.SizeType,
    n: TC.SizeType,
    k: TC.SizeType,
) void {
    const T = UT.Child(@TypeOf(Y_values));
    
    overloads.kernel_linear_ij_jk.call(.{
        stream.context, 
        A_values.ptr, 
        B_values.ptr, 
        SC.asScalar(T, alpha),
        C_values.ptr,
        SC.asScalar(T, beta), 
        Y_values.ptr,
        m, n, k
    });
}

pub inline fn linear_ij_jk(
    stream: Stream, 
    A: anytype, 
    B: anytype,
    alpha: f16,
    C: anytype,
    beta: f16,
    Y: anytype,
) void {
    const A_sizes = A.sizes();
    const B_sizes = B.sizes();
    const m = A_sizes[0];
    const n = A_sizes[1];
    const k = B_sizes[1];

    std.debug.assert(A_sizes.len == 2);
    std.debug.assert(B_sizes.len == 2);
    std.debug.assert(n == B_sizes[0]);
    std.debug.assert(Y.len() == m * k);
    std.debug.assert(C.len() == Y.len());

    __linear_ij_jk(
        stream,
        A.values(),
        B.values(),
        alpha,
        C.values(),
        beta, //beta
        Y.values(),
        m, n, k
    );
}

inline fn linear_ij_jk_ReverseArg0(
    stream: Stream, 
    A: anytype, 
    B: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    Y: anytype,
) void {
    const T = Child(@TypeOf(A));
    const A_grads = A.grads().?;
    const B_values = B.values();
    const Y_grads = Y.grads().?;
    const Y_sizes = Y.sizes();
    const B_sizes = B.sizes();

    const B_tran = stream.getScratch(T, B_values.len);

    overloads.kernel_permutate_ij_ji.call(.{
        stream.context, 
        B_values.ptr, 
        B_tran.ptr, 
        SC.asScalar(T, 0.0), 
        B_sizes[0], 
        B_sizes[1] 
    });

    __linear_ij_jk(
        stream, 
        Y_grads,
        B_tran,
        alpha, // alpha
        A_grads,
        1.0, // beta
        A_grads,
        Y_sizes[0],
        Y_sizes[1],
        B_sizes[0],
    );
}

inline fn linear_ij_jk_ReverseArg1(
    stream: Stream, 
    A: anytype, 
    B: anytype,
    alpha: f16,
    _: anytype,
    _: f16,
    Y: anytype,
) void {
    const T = Child(@TypeOf(A));

    const A_values = A.values();
    const B_grads = B.grads().?;
    const Y_grads = Y.grads().?;

    const Y_sizes = Y.sizes();
    const A_sizes = A.sizes();

    const A_tran = stream.getScratch(T, A_values.len);

    overloads.kernel_permutate_ij_ji.call(.{
        stream.context, 
        A_values.ptr, 
        A_tran.ptr, 
        SC.asScalar(T, 0.0), 
        A_sizes[0], 
        A_sizes[1] 
    });

    __linear_ij_jk(
        stream, 
        A_tran,
        Y_grads,
        alpha,
        B_grads,
        1.0, // beta
        B_grads,
        A_sizes[1],
        A_sizes[0],
        Y_sizes[1],
    );
}

const Linear_ij_jk_Callback = CallbackBuilder(
    linear_ij_jk, .{
        .{ linear_ij_jk_ReverseArg0, 0 },
        .{ linear_ij_jk_ReverseArg1, 1 },
        .{ linear_bias_ReverseArg3,  3 },
    }, NoCleanup,
);

const inner_product_expressions = std.ComptimeStringMap(
    type, .{
        // Rank-1-to-Rank-2
        .{ "i,ij->j", Linear_i_ij_Callback },
        .{ "i,ji->j", Linear_i_ji_Callback },
        .{ "j,ij->i", Linear_i_ji_Callback },
        .{ "j,ji->i", Linear_i_ij_Callback },

        // Rank-2-to-Rank-1
        .{ "ij,j->i", Linear_ij_j_Callback },
        .{ "ij,i->j", Linear_ij_i_Callback },
        .{ "ji,i->j", Linear_ij_j_Callback },
        .{ "ji,j->i", Linear_ij_i_Callback },

        // Rank-2-to-Rank-2
        .{ "ij,jk->ik", Linear_ij_jk_Callback }
    }
);

pub fn findLinear(comptime expression: []const u8, comptime bias: bool) type {
    const parsed = comptime Parser.innerProductExpression(expression);
    if (inner_product_expressions.get(parsed)) |ip| {
        return LinearType(ip, bias);
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
