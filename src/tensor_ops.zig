
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

pub fn additionForward(x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_addition.call(.{
        z.ptr.stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn additionReverseArg0(X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        Z.ptr.stream, x_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn additionReverseArg1(_: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_addition_reverse.call(.{
        Z.ptr.stream, y_grads.ptr, z_grads.ptr, z_grads.len
    });
}

pub const AddImpl = CallbackBuilder(
    additionForward, .{
        .{ additionReverseArg0, 0 },
        .{ additionReverseArg1, 1 }
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn subtractionForward(x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_subtraction.call(.{
        z.ptr.stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn subtractionReverseArg0(X: anytype, _: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const z_grads = UT.assertGrads(Z);

    // scalar is positive for left-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), 1.0);
    
    overloads.kernel_subtraction_reverse.call(.{
        Z.ptr.stream, x_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub fn subtractionReverseArg1(_: anytype, Y: anytype, Z: anytype) void {
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    // scalar is negative for right-hand operand
    const coef = SC.asScalar(SC.DemoteComplex(Child(z_grads)), -1.0);

    overloads.kernel_subtraction_reverse.call(.{
        Z.ptr.stream, y_grads.ptr, z_grads.ptr, coef, z_grads.len
    });
}

pub const SubImpl = CallbackBuilder(
    subtractionForward, .{
        .{ subtractionReverseArg0, 0 },
        .{ subtractionReverseArg1, 1 }
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn hadamardForward(x: anytype, y: anytype, z: anytype) void {
    const x_values = x.values();
    const y_values = y.values();
    const z_values = z.values();

    overloads.kernel_hadamard.call(.{
        z.ptr.stream, x_values.ptr, y_values.ptr, z_values.ptr, z_values.len
    });
}

pub fn hadamardReverseArg0(X: anytype, Y: anytype, Z: anytype) void {
    const x_grads = UT.assertGrads(X);
    const y_value = Y.values();
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        Z.ptr.stream, x_grads.ptr, y_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub fn hadamardReverseArg1(X: anytype, Y: anytype, Z: anytype) void {
    const x_value = X.values();
    const y_grads = UT.assertGrads(Y);
    const z_grads = UT.assertGrads(Z);

    overloads.kernel_hadamard_reverse.call(.{
        Z.ptr.stream, y_grads.ptr, x_value.ptr, z_grads.ptr, z_grads.len
    });
}

pub const HadamardImpl = CallbackBuilder(
    hadamardForward, .{
        .{ hadamardReverseArg0, 0 },
        .{ hadamardReverseArg1, 1 },
    }, NoCleanup
);

// <>--------------------------------------------------------<>

pub fn leakyReluForward(x: anytype, coef: anytype, y: anytype) void {
    const T = Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_leaky_relu.call(.{
        y.ptr.stream,
        x_values.ptr, 
        y_values.ptr, 
        SC.asScalar(T, coef), 
        y_values.len
    });
}

pub fn leakyReluReverse(x: anytype, coef: anytype, y: anytype) void {
    const T = Child(@TypeOf(x));
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_leaky_relu_reverse.call(.{
        y.ptr.stream,
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

pub fn tanhForward(x: anytype, y: anytype) void {
    const x_values = x.values();
    const y_values = y.values();

    overloads.kernel_tanh.call(.{
        y.ptr.stream, x_values.ptr,  y_values.ptr,  y_values.len
    });
}

pub fn tanhReverse(x: anytype, y: anytype) void {
    const x_grads = x.values();
    const y_values = y.values();
    const y_grads = UT.assertGrads(y);

    overloads.kernel_tanh_reverse.call(.{
        y.ptr.stream, x_grads.ptr, y_values.ptr, y_grads.ptr, y_values.len
    });
}

pub const TanhImpl = CallbackBuilder(
    tanhForward, .{
        .{ tanhReverse, 0 },
    }, NoCleanup
);

