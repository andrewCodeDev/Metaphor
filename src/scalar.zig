const std = @import("std");
const C = @import("cimport.zig").C;

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

// we use this as an field accessor for union types
// @typeName will call through to the name of the
// underlying typle but we need the alias
pub fn scalarName(comptime T: type) []const u8 {
    return switch (T) {
         q8 =>  "q8",
        r16 => "r16",
        r32 => "r32",
        r64 => "r64",
        c16 => "c16",
        c32 => "c32",
        c64 => "c64",
        else => @compileError("Invalid type for scalarName: " ++ @typeName(T)),
    };
}

  ////////////////////////////////////////////////
 ///// Constraints //////////////////////////////
////////////////////////////////////////////////

pub inline fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Int, .ComptimeInt => true,
        else => false
    };
}

pub inline fn isFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Float, .ComptimeFloat => true, 
        else => false
    };
}

// we have to support some weird types of floating
// point "equivalents". Because of this, we cannot
// rely on the builtin @typeInfo.

pub inline fn isReal(comptime T: type) bool {
    return switch (T) {
        r16, r32, r64 => true, 
        else => false
    };
}

pub inline fn isComplex(comptime T: type) bool {
    return switch (T) {
        c16, c32, c64 => true, 
        else => false
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
        }
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
        }
    };
}

pub fn DeduceComplex(comptime ctype: type, comptime rtype: type) type {
    if (@sizeOf(rtype) <= (@sizeOf(ctype) / 2)) {
        return C; // 2x means the inidivdual members are sized appropriately
    }
    else return PromoteComplex(rtype);
}

pub fn ScalarResult(comptime T: type, comptime U: type) type {
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
    }
    else if (comptime isComplex(T) and isReal(U)) {
        return DeduceComplex(T, U);     
    }
    else if (comptime isReal(T) and isComplex(U)) {
        return DeduceComplex(U, T);     
    }
    else {
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

            return r16 { 
                .__x = @as(*const u16, @ptrCast(@alignCast(&u))).* 
            };
        },
        .Float, .ComptimeFloat => {
            const u: f16 = @floatCast(x);

            return r16 { 
                .__x = @as(*const u16, @ptrCast(@alignCast(&u))).* 
            };
        },
        else => @compileError(
          "Invalid Type for r16 Conversion: " ++ @typeName(@TypeOf(x))  
        ),
    }
}

inline fn __r16_as(comptime T: type, u: r16) T {

    if (comptime T == r16) {
        return u;
    }
    else if (comptime isFloat(T) or isReal(T)) {
        return @floatCast(@as(*const f16, @ptrCast(@alignCast(&u.__x))).*);
    }
    else if (comptime isComplex(T)) {
        if (comptime T == c16) {
            return c16{ .r = u, .i = r16{ .__x = 0 } };
        }
        return T {
            .r = @floatCast(@as(*const f16, @ptrCast(@alignCast(&u.__x))).*),
            .i = 0,
        };        
    }
    else {
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

    else if(comptime isFloat(T) and isFloat(U)) {
        return @floatCast(x);
    }

    else if(comptime isFloat(T) and isInteger(U)) {
        return @floatFromInt(x);
    }

    /////////////////////////////////////////////
    ///// Real type casting /////////////////////

    else if (comptime isReal(T) and isInteger(U)) {
        return switch (T) {
            r16 => __r16_init(x), 
            r32, r64 => @floatFromInt(x),
            else => @compileError(
                "Cannot cast to scalar from: " ++ @typeName(T)  
            ),
        };
    }

    else if (comptime isReal(T) and isFloat(U)) {
        return switch (T) {
            r16 => __r16_init(x), 
            r32, r64 => @floatCast(x),
            else => @compileError(
                "Cannot cast to scalar from: " ++ @typeName(T)  
            ),
        };
    }

    else if (comptime isReal(T) and isReal(U)) {
        // we have already handled the branch where U == T
        return switch (T) {
            // U: r32, r64
            r16 => __r16_init(x), 

            // U: r16, r64
            r32 => if (U == r16)
                __r16_as(T, x) else @floatCast(x),

            // U: r16, r32
            r64 => if (U == r16)
                __r16_as(T, x) else @floatCast(x),
            
            else => @compileError(
                "Cannot cast to scalar from: " ++ @typeName(T)  
            ),
        };
    }

    /////////////////////////////////////////////
    ///// Complex type casting //////////////////

    else if (comptime isComplex(T) and (isReal(U) or isFloat(U) or isInteger(U))) {
        return T {
            .r = asScalar(DemoteComplex(T), x),
            .i = asScalar(DemoteComplex(T), 0),
        };
    }

    else if (comptime isComplex(T) and isComplex(U)) {
        return T {
            .r = asScalar(DemoteComplex(T), x.r),
            .i = asScalar(DemoteComplex(T), x.i),
        };
    }

    else {
        @compileError("Invalid types for asScalar: " ++ @typeName(T) ++ ", " ++ @typeName(U));
    }
}

