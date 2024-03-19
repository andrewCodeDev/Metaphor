const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

pub fn main() !void {
    // Casting between types is important for creating default
    // and initial values. Metaphor has 7 scalar types it
    // recognizes:

    // real:      r16, r32, r64
    // complex:   c16, c32, c64
    // quantized: q8

    // The reason that rN was used instead of fN is that
    // translating through the cuda types requires support
    // for __half. Half is a 16-bit type used to represent
    // f16. Due to this, handling r16 is different than f16.

    // To make this more convenient, Metaphor has a scalar
    // casting function to allow smoother operations between
    // native types. This casting operation also respects
    // values like `inf` when going between integers to floats.
    {
        // can be integer or float for rN types
        const x = mp.scalar.as(mp.scalar.r16, 42);

        // getting a native scalar value back
        const y = mp.scalar.as(f16, x);

        std.log.info("Value from comptime_int -> r16 -> f16: {}", .{y});
    }

    // casting for complex types is a one-way operation. You cannot cast
    // back to a floating point number from a complex type. If this is
    // required, the members { r, i } can be individually casted:
    {
        // creates complex number { r = 2.0, i = 0.0 }
        const cx = mp.scalar.as(mp.scalar.c64, 2.0);

        // cast to a different complex type
        const cy = mp.scalar.as(mp.scalar.c32, cx);

        std.log.info("Value from comptime_float -> c32: {}", .{cx});
        std.log.info("Value from c64 -> c32: {}", .{cy});
    }

    // casting between integers and floats is automatically handled
    // for all native integer and floating point types using builtin
    // casting operations:
    {
        const i: i32 = 99;
        const f: f16 = 16.0;

        // casting float-to-int
        const i_to_f = mp.scalar.as(f16, i);
        const f_to_i = mp.scalar.as(i32, f);

        std.log.info("Value from i32 -> f16: {}", .{i_to_f});
        std.log.info("Value from f16 -> i32: {}", .{f_to_i});

        // casting int-to-int
        const i_1: usize = 100;
        const i_2 = mp.scalar.as(i8, i_1);

        std.log.info("Value from usize -> i8: {}", .{i_2});

        // casting float-to-float
        const f_1: f64 = -50;
        const f_2 = mp.scalar.as(f16, f_1);

        std.log.info("Value from f64 -> f16: {}", .{f_2});
    }
}
