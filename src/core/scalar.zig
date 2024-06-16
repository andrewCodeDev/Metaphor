const std = @import("std");
const UT = @import("utils.zig");
const SC = @This();

////////////////////////////////////////////////
///// Scalar Types /////////////////////////////
////////////////////////////////////////////////

// real types
pub const r16 = UT.dev.r16;
pub const r32 = UT.dev.r32;
pub const r64 = UT.dev.r64;

// quantized types
pub const q8 = i8;

// we use this as a field accessor for union types
// @typeName will call through to the name of the
// underlying type but we need the alias
pub fn name(comptime T: type) []const u8 {
    return switch (T) {
         q8 => "q8",
        r16 => "r16",
        r32 => "r32",
        r64 => "r64",
        else => @compileError("Invalid type for scalar_name: " ++ @typeName(T)),
    };
}

pub fn native(comptime T: type) []const u8 {
    return switch (T) {
         q8 => u8,
        r16 => f16,
        r32 => f32,
        r64 => f64,
        else => @compileError("Invalid type for native: " ++ @typeName(T)),
    };
}

pub const Tag = enum(u2) {

    // do not change ordering - q8 is not a generated type
     q8 = 3,

    // these replace scalars in declarations in the given order
    r16 = 0,
    r32 = 1,
    r64 = 2,

    pub fn as_type(comptime tag: Tag) type {
        return switch (tag) {
             .q8 => SC.q8,
            .r16 => SC.r16,
            .r32 => SC.r32,
            .r64 => SC.r64,
        };
    }
    pub fn as_tag(comptime T: type) Tag {
        return switch (T) {
             SC.q8 => .q8,
            SC.r16 => .r16,
            SC.r32 => .r32,
            SC.r64 => .r64,
            else => @compileError("Invalid type for asTag: " ++ @typeName(T)),
        };
    }
};

////////////////////////////////////////////////
///// Constraints //////////////////////////////
////////////////////////////////////////////////

pub const is_integer = UT.is_integer;
pub const is_float = UT.is_float;

// we have to support some weird types of floating
// point "equivalents". Because of this, we cannot
// rely on the builtin @typeInfo.

pub inline fn is_real(comptime T: type) bool {
    return switch (T) {
        r16, r32, r64 => true,
        else => false,
    };
}

////////////////////////////////////////////////
///// Scalar-Type Deduction ////////////////////
////////////////////////////////////////////////

fn MaxType(comptime T: type, comptime U: type) type {
    return if (@sizeOf(T) > @sizeOf(U)) T else U;
}

// this works from
inline fn r16_init(x: anytype) r16 {
    if (comptime @TypeOf(x) == r16) {
        return x;
    }

    // r16 internally uses unsigned short, so
    // to get our bit pattern correct, first
    // we go to an f16 first and cast to u16

    switch (@typeInfo(@TypeOf(x))) {
        .Int, .ComptimeInt => {
            const u: f16 = @floatFromInt(x);
            return r16{ .x = @bitCast(u) };
        },
        .Float, .ComptimeFloat => {
            const u: f16 = @floatCast(x);
            return r16{ .x = @bitCast(u) };
        },
        else => @compileError("Invalid Type for r16 Conversion: " ++ @typeName(@TypeOf(x))),
    }
}

inline fn r16_as(comptime T: type, u: r16) T {
    if (comptime T == r16) {
        return u;
    } else if (comptime is_float(T) or is_real(T)) {
        return @floatCast(@as(f16, @bitCast(u.x)));
    } else if (comptime is_integer(T)) {
        return @intFromFloat(@as(f16, @bitCast(u.x)));
    } else {
        @compileError("Cannot cast r16 to: " ++ @typeName(T));
    }
}

pub fn as(comptime T: type, x: anytype) T {
    // This is complicated because of the inclusion of "half"
    // types that use integral types to internally represent
    // floating point numbers. That's our main concern here.
    const U = @TypeOf(x);

    if (comptime !(is_integer(T) or is_float(T) or is_real(T))) {
        @compileError("Invalid result type for asScalar: " ++ @typeName(T));
    }

    if (comptime T == U) {
        return x;
    }

    // casting to float or integer from float/r16
    else if (comptime is_float(T) and (is_float(U) or U == r16)) {
        return if (comptime U == r16) r16_as(T, x) else @floatCast(x);
    } else if (comptime is_integer(T) and (is_float(U) or U == r16)) {
        return if (comptime U == r16) r16_as(T, x) else @intFromFloat(x);
    }

    // native casting operations between types
    else if (comptime is_float(T) and is_integer(U)) {
        return @floatFromInt(x);
    } else if (comptime is_integer(T) and is_integer(U)) {
        return @intCast(x);
    }

    /////////////////////////////////////////////
    ///// Real type casting /////////////////////

    // TODO: inspect if we ever reach this branch for r32, r64?

    else if (comptime is_real(T) and is_integer(U)) {
        return switch (T) {
            r16 => r16_init(x),
            r32, r64 => @floatFromInt(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };
    } else if (comptime is_real(T) and is_float(U)) {
        return switch (T) {
            r16 => r16_init(x),
            r32, r64 => @floatCast(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };
    } else if (comptime is_real(T) and is_real(U)) {
        // we have already handled the branch where U == T
        return switch (T) {
            // U: r32, r64
            r16 => r16_init(x),
            // U: r16, r64
            r32 => if (U == r16) r16_as(T, x) else @floatCast(x),
            // U: r16, r32
            r64 => if (U == r16) r16_as(T, x) else @floatCast(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };

    } else {
        @compileError("Invalid types for asScalar: " ++ @typeName(T) ++ ", " ++ @typeName(U));
    }
}
