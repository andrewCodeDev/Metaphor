
const std = @import("std");
const math = std.math;
const Child = std.meta.Child;
const SizeType = usize;
const SC = @import("scalar.zig");
const UT = @import("utility.zig");
const CB = @import("callback_builder.zig");
const CallbackBuilder = CB.CallbackBuilder;
const NoCleanup = CB.NoCleanup;
const overloads = @import("kernel_overloads.zig");
const Stream = @import("cimport.zig").C.Stream;

//pub const InnerProductPlan = @import("expression_parsing.zig").InnerProductPlan;
pub const contractionParse = @import("expression_parsing.zig").contractionParse;
pub const innerProductParse = @import("expression_parsing.zig").innerProductParse;
pub const outerProductParse = @import("expression_parsing.zig").outerProductParse;
//pub const compUT.TensorIndex = @import("Tensor.zig").compUT.TensorIndex;

// TODO: 
//  add extra parameter for device - how to do that? Stream? Maybe.
//  add same parameter to realArithmetic and complexArithmetic.
//  write 'add' cuda kernel for dispatch - needs multiple types.
//  write cuda allocator... caching_allocator with cuda backing?

// <>--------------------------------------------------------<>

pub fn additionForward(stream: Stream, x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_addition.call(.{
        stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn additionReverseArg1(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream, x_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn additionReverseArg2(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        stream, y_grads.ptr, z_grads.ptr, z_grads.len
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
        stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn subtractionReverseArg1(stream: Stream, X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    // scalar is positive for left-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), 1.0);
    
    overloads.kernel_subtraction_reverse.call(.{
        stream, x_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub fn subtractionReverseArg2(stream: Stream, _: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    // scalar is negative for right-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), -1.0);

    overloads.kernel_subtraction_reverse.call(.{
        stream, y_grads.ptr, z_grads.ptr, coef, z_grads.len
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
        stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn hadamardReverseArg1(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const y_value = Y.values();
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream, x_grads.ptr, y_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn hadamardReverseArg2(stream: Stream, X: anytype, Y: anytype, Z: anytype) void {
    const x_value = X.values();
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        stream, y_grads.ptr, x_value.ptr, z_grads.ptr, z_grads.len
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
        stream,
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
        stream,
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
        stream, x_values.ptr,  y_values.ptr,  y_values.len
    });
}

pub fn tanhReverse(stream: Stream, x: anytype, y: anytype) void {
    const x_grads = x.values();
    const y_values = y.values();
    const y_grads = UT.assertGrads(y);

    overloads.kernel_tanh_reverse.call(.{
        stream, x_grads.ptr, y_values.ptr, y_grads.ptr, y_values.len
    });
}

pub const TanhImpl = CallbackBuilder(
    tanhForward, .{
        .{ tanhReverse, 1 },
    }, NoCleanup
);

