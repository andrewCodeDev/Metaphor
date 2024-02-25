
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

pub fn additionReverseArg1(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream.context, x_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn additionReverseArg2(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream.context, y_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub const AddImpl = CallbackBuilder(
    additionForward, .{
        .{ additionReverseArg1, 1 },
        .{ additionReverseArg2, 2 }
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

pub fn subtractionReverseArg1(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    // scalar is positive for left-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), 1.0);
    
    overloads.kernel_subtraction_reverse.call(.{
        stream.context, x_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub fn subtractionReverseArg2(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
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
        .{ subtractionReverseArg1, 1 },
        .{ subtractionReverseArg2, 2 }
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

pub fn hadamardReverseArg1(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const y_value = Y.values();
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream.context, x_grads.ptr, y_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn hadamardReverseArg2(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_value = X.values();
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream.context, y_grads.ptr, x_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub const HadamardImpl = CallbackBuilder(
    hadamardForward, .{
        .{ hadamardReverseArg1, 1 },
        .{ hadamardReverseArg2, 2 },
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
        .{ leakyReluReverse, 1 },
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
        .{ tanhReverse, 1 },
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

pub inline fn transpose2D(
    stream: Stream, 
    x: anytype, 
    y: anytype,
) void {
    const T = UT.Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();
    const x_sizes = x.sizes();
    overloads.kernel_transpose_2D.call(.{
       stream.context, x_values.ptr, y_values.ptr, SC.asScalar(T, 0.0), x_sizes[0], x_sizes[1] 
    });
}

pub inline fn transpose2DReverse(
    stream: Stream, 
    x: anytype, 
    y: anytype,
) void {
    const T = UT.Child(@TypeOf(x));
    const x_grads = x.grads().?;
    const y_grads = y.grads().?;
    const y_sizes = y.sizes();
    overloads.kernel_transpose_2D.call(.{
       stream.context, y_grads.ptr, x_grads.ptr, SC.asScalar(T, 1.0), y_sizes[0], y_sizes[1] 
    });
}

pub fn permutate(
    stream: Stream, 
    x: anytype, 
    y: anytype,    
    comptime expression: []const u8
) void {
    const permutation = comptime Parser.permutation(expression);

    switch (permutation) {
        .@"ij->ji" => transpose2D(stream, x, y),
        // try to never reach this branch
        // this is the unoptimized kernel
        .unknown => {
            @compileError("TODO: Declare General Permutation Kernel.");            
        }
    }
}

pub fn permutateReverse(
    stream: Stream, 
    x: anytype, 
    y: anytype,    
    comptime expression: []const u8
) void {
    const permutation = comptime Parser.permutation(expression);

    switch (permutation) {
        .@"ij->ji" => transpose2DReverse(stream, x, y),
        // try to never reach this branch
        // this is the unoptimized kernel
        .unknown => {
            @compileError("TODO: Declare General Permutation Kernel.");            
        }
    }
}

// reversing a permutation means applying it again to itself
pub fn PermutateCallback(comptime expression: []const u8) type {
    return struct {
        reverse: struct {
            
            callback: fn (Stream, anytype, anytype) void = struct {
                // bind expression to reversal for undoing permutation
                // this function will be called because i
                pub fn call(stream: Stream, x: anytype, y: anytype) void {
                    permutateReverse(stream.context, x, y, expression);
                }
            }.call,

            edge_index: comptime_int = 1,

        } = .{ },
    };
}

// <>--------------------------------------------------------<>

inline fn __matmul2D(
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

pub inline fn matmul2D(
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

    __matmul2D(
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

inline fn matmul2DReverseArg1(
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

    overloads.kernel_transpose_2D.call(.{
        stream.context, 
        y_values.ptr, 
        y_tran.ptr, 
        SC.asScalar(T, 0.0), 
        y_sizes[0], 
        y_sizes[1] 
    });

    __matmul2D(
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

inline fn matmul2DReverseArg2(
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

    overloads.kernel_transpose_2D.call(.{
        stream.context, 
        x_values.ptr, 
        x_tran.ptr, 
        SC.asScalar(T, 0.0), 
        x_sizes[0], 
        x_sizes[1] 
    });

    __matmul2D(
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

const MatMul2DImpl = CallbackBuilder(
    matmul2D, .{
        .{ matmul2DReverseArg1, 1 },
        .{ matmul2DReverseArg2, 2 },
    }, NoCleanup,
);

pub fn innerProduct(
    stream: Stream, 
    x: anytype, 
    y: anytype,
    z: anytype,    
    comptime expression: []const u8
) void {
    const inner_product = comptime Parser.innerProduct(expression);

    switch (inner_product) {
        .@"ij,jk->ik" => matmul2D(stream, x, y, z),
        // try to never reach this branch
        // this is the unoptimized kernel
        .unknown => {
            @compileError("TODO: Declare General Inner Product Kernel.");            
        }
    }
}

pub fn innerProductReverse(
    // this is different than permutate
    // because we have separate reversals
    // for each argument in the call
    comptime expression: []const u8
) type {
    const inner_product = comptime Parser.innerProduct(expression);

    return switch (inner_product) {
        .@"ij,jk->ik" => MatMul2DImpl,
        // try to never reach this branch
        // this is the unoptimized kernel
        .unknown => {
            @compileError("TODO: Declare General Inner Product Kernel.");            
        }
    };
}
// <>--------------------------------------------------------<>
