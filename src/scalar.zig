const std = @import("std");
const C = @import("cimport.zig").C;
const SC = @This();
const UT = @import("utility.zig");

////////////////////////////////////////////////
///// Scalar Types /////////////////////////////
////////////////////////////////////////////////

// real types
pub const r16 = C.r16;
pub const r32 = C.r32;
pub const r64 = C.r64;

// complex types
pub const c16 = C.c16;
pub const c32 = C.c32;
pub const c64 = C.c64;

// quantize types
pub const q8 = i8;

// we use this as a field accessor for union types
// @typeName will call through to the name of the
// underlying type but we need the alias
pub fn scalarName(comptime T: type) []const u8 {
    return switch (T) {
        q8 => "q8",
        r16 => "r16",
        r32 => "r32",
        r64 => "r64",
        c16 => "c16",
        c32 => "c32",
        c64 => "c64",
        else => @compileError("Invalid type for scalarName: " ++ @typeName(T)),
    };
}

pub fn Native(comptime T: type) []const u8 {
    return switch (T) {
        q8 => u8,
        r16 => f16,
        r32 => f32,
        r64 => f64,
        c16 => std.math.Complex(f16),
        c32 => std.math.Complex(f32),
        c64 => std.math.Complex(f64),
        else => @compileError("Invalid type for scalarName: " ++ @typeName(T)),
    };
}

pub const ScalarTag = enum {
    q8,
    r16,
    r32,
    r64,
    c16,
    c32,
    c64,

    pub fn asType(comptime opt: ScalarTag) type {
        return switch (opt) {
            .q8 => SC.q8,
            .r16 => SC.r16,
            .r32 => SC.r32,
            .r64 => SC.r64,
            .c16 => SC.c16,
            .c32 => SC.c32,
            .c64 => SC.c64,
        };
    }

    pub fn asTag(comptime T: type) ScalarTag {
        return switch (T) {
            SC.q8 => .q8,
            SC.r16 => .r16,
            SC.r32 => .r32,
            SC.r64 => .r64,
            SC.c16 => .c16,
            SC.c32 => .c32,
            SC.c64 => .c64,
            else => @compileError("Invalid type for asTag: " ++ @typeName(T)),
        };
    }
};

////////////////////////////////////////////////
///// Constraints //////////////////////////////
////////////////////////////////////////////////

pub const isInteger = UT.isInteger;
pub const isFloat = UT.isFloat;

// we have to support some weird types of floating
// point "equivalents". Because of this, we cannot
// rely on the builtin @typeInfo.

pub inline fn isReal(comptime T: type) bool {
    return switch (T) {
        r16, r32, r64 => true,
        else => false,
    };
}

pub inline fn isComplex(comptime T: type) bool {
    return switch (T) {
        c16, c32, c64 => true,
        else => false,
    };
}

pub inline fn isScalar(comptime T: type) bool {
    return isComplex(T) or isReal(T);
}

////////////////////////////////////////////////
///// Scalar-Type Deduction ////////////////////
////////////////////////////////////////////////

fn MaxType(comptime T: type, comptime U: type) type {
    return if (@sizeOf(T) > @sizeOf(U)) T else U;
}

pub fn PromoteComplex(comptime T: type) type {
    if (comptime isComplex(T)) {
        return T;
    }
    return switch (T) {
        r16 => c16,
        r32 => c32,
        r64 => c64,
        else => {
            @compileError("Can only promote complex from { f16, f32, f64 }");
        },
    };
}

pub fn DemoteComplex(comptime T: type) type {
    if (comptime isReal(T)) {
        return T;
    }
    return switch (T) {
        c16 => r16,
        c32 => r32,
        c64 => r64,
        else => {
            @compileError("Can only demote complex from { c16, c32, c64 }");
        },
    };
}

pub fn DeduceComplex(comptime ctype: type, comptime rtype: type) type {
    // 2x means the individual members are sized appropriately
    return if (@sizeOf(rtype) <= (@sizeOf(ctype) / 2)) ctype else PromoteComplex(rtype);
}

pub fn ScalarResult(comptime T: type, comptime U: type) type {
    if (@typeInfo(T) == .Struct) {
        if (comptime @hasDecl(T, "DataType") and @hasDecl(U, "DataType")) {
            return ScalarResult(T.DataType, U.DataType);
        }
    }
    if (comptime !(isScalar(T) and isScalar(U))) {
        @compileError("Scalar arithmetic requires scalar types.");
    }
    if (comptime isInteger(T) and isInteger(U)) {
        return MaxType(T, U);
    }
    if (comptime isInteger(T) or isInteger(U)) {
        @compileError("Metaphor does not support integer to floating point arithmetic.");
    }
    if (comptime isReal(T) and isReal(U)) {
        return MaxType(T, U);
    } else if (comptime isComplex(T) and isReal(U)) {
        return DeduceComplex(T, U);
    } else if (comptime isReal(T) and isComplex(U)) {
        return DeduceComplex(U, T);
    } else {
        return MaxType(T, U);
    }
}

// this works from
inline fn __r16_init(x: anytype) r16 {
    if (comptime @TypeOf(x) == r16) {
        return x;
    }

    // r16 internally uses unsigned short, so
    // to get our bit pattern correct, first
    // we go to an f16 first and cast to u16

    switch (@typeInfo(@TypeOf(x))) {
        .Int, .ComptimeInt => {
            const u: f16 = @floatFromInt(x);
            return r16{ .__x = @bitCast(u) };
        },
        .Float, .ComptimeFloat => {
            const u: f16 = @floatCast(x);
            return r16{ .__x = @bitCast(u) };
        },
        else => @compileError("Invalid Type for r16 Conversion: " ++ @typeName(@TypeOf(x))),
    }
}

inline fn __r16_as(comptime T: type, u: r16) T {
    if (comptime T == r16) {
        return u;
    } else if (comptime isFloat(T) or isReal(T)) {
        return @floatCast(@as(f16, @bitCast(u.__x)));
    } else if (comptime isInteger(T)) {
        return @intFromFloat(@as(f16, @bitCast(u.__x)));
    } else if (comptime isComplex(T)) {
        if (comptime T == c16) {
            return c16{ .r = u, .i = r16{ .__x = 0.0 } };
        }
        return .{
            .r = @floatCast(@as(f16, @bitCast(u.__x))),
            .i = 0.0,
        };
    } else {
        @compileError("Cannot cast r16 to: " ++ @typeName(T));
    }
}

pub fn asScalar(comptime T: type, x: anytype) T {
    // This is complicated because of the inclusion of "half"
    // types that use integral types to internally represent
    // floating point numbers. That's our main concern here.
    const U = @TypeOf(x);

    if (comptime !(isInteger(T) or isFloat(T) or isReal(T) or isComplex(T))) {
        @compileError("Invalid result type for asScalar: " ++ @typeName(T));
    }

    if (comptime T == U) {
        return x;
    }

    // casting to float or integer from float/r16
    else if (comptime isFloat(T) and (isFloat(U) or U == r16)) {
        return if (comptime U == r16) __r16_as(T, x) else @floatCast(x);
    } else if (comptime isInteger(T) and (isFloat(U) or U == r16)) {
        return if (comptime U == r16) __r16_as(T, x) else @intFromFloat(x);
    }

    // native casting operations between types
    else if (comptime isFloat(T) and isInteger(U)) {
        return @floatFromInt(x);
    } else if (comptime isInteger(T) and isInteger(U)) {
        return @intCast(x);
    }

    /////////////////////////////////////////////
    ///// Real type casting /////////////////////

    // TODO: inspect if we ever reach this branch for r32, r64?

    else if (comptime isReal(T) and isInteger(U)) {
        return switch (T) {
            r16 => __r16_init(x),
            r32, r64 => @floatFromInt(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };
    } else if (comptime isReal(T) and isFloat(U)) {
        return switch (T) {
            r16 => __r16_init(x),
            r32, r64 => @floatCast(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };
    } else if (comptime isReal(T) and isReal(U)) {
        // we have already handled the branch where U == T
        return switch (T) {
            // U: r32, r64
            r16 => __r16_init(x),
            // U: r16, r64
            r32 => if (U == r16) __r16_as(T, x) else @floatCast(x),
            // U: r16, r32
            r64 => if (U == r16) __r16_as(T, x) else @floatCast(x),
            else => @compileError("Cannot cast to scalar from: " ++ @typeName(T)),
        };
    }

    /////////////////////////////////////////////
    ///// Complex type casting //////////////////

    else if (comptime isComplex(T) and (isReal(U) or isFloat(U) or isInteger(U))) {
        return .{
            .r = asScalar(DemoteComplex(T), x),
            .i = asScalar(DemoteComplex(T), 0),
        };
    } else if (comptime isComplex(T) and isComplex(U)) {
        return .{
            .r = asScalar(DemoteComplex(T), x.r),
            .i = asScalar(DemoteComplex(T), x.i),
        };
    } else {
        @compileError("Invalid types for asScalar: " ++ @typeName(T) ++ ", " ++ @typeName(U));
    }
}
