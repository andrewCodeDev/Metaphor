const std = @import("std");
const builtin = @import("builtin");
const debug = (builtin.mode == std.builtin.OptimizeMode.Debug);

//////////////////////////////////////////////////////////////////////////////////
// Allocator clean-up functions... these basically exist to regularize the syntax.
pub inline fn alloc(comptime T: type, n: usize, allocator: std.mem.Allocator) []T {
    return allocator.alloc(T, n) catch @panic("Alloc: Out of memory.");
}
pub inline fn dupe(x: anytype, allocator: std.mem.Allocator) @TypeOf(x) {
    return allocator.dupe(Child(@TypeOf(x)), x) catch @panic("Dupe: Out of memory.");
}
pub inline fn append(array: anytype, value: anytype) void {
    return array.append(value) catch @panic("Append: Out of memory.");
}

// TODO: remove this function. It hides the source location of the error.
pub inline fn assertGrads(x: anytype) std.meta.Child(@TypeOf(x.grads())) {
    if (comptime debug) {
        if (x.grads()) |grd| {
            return grd;
        } else {
            @panic("Unassigned tensor gradient.");
        }
    } else {
        return x.grads().?;
    }
}

/////////////////////////////////////
// Contracts for function call sites.

// TODO: remove this function.
pub fn Contract(comptime constraint: bool, comptime result: type) type {
    if (!constraint) {
        @compileError("Failed type constraints.");
    }
    return result;
}

// TODO: remove this function.
pub fn Returns(comptime result: type) type {
    return result;
}

pub fn isPointer(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => true,
        else => false,
    };
}

pub fn isArray(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Array => true,
        else => false,
    };
}

pub fn isSlice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| ptr.size == .Slice,
        else => false,
    };
}

pub inline fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Int, .ComptimeInt => true,
        else => false,
    };
}

pub inline fn isFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Float, .ComptimeFloat => true,
        else => false,
    };
}

pub fn fieldsLen(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.fields.len,
        else => @compileError("fieldsLen: T must be a struct/tuple type."),
    };
}

pub fn isStruct(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => true,
        else => false,
    };
}

pub fn isFunction(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Fn => true,
        else => false,
    };
}

pub fn isTuple(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.is_tuple,
        else => false,
    };
}

pub fn tupleSize(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.fields.len,
        else => @compileError("Type must be a tuple."),
    };
}

pub fn Child(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .Struct => if (comptime @hasDecl(T, "DataType")) T.DataType else @compileError("Child function expects tensor or slice type."),
        .Pointer => return std.meta.Child(T),
        else => @compileError("Child function expects tensor or slice type."),
    };
}

pub fn NonConstPtr(comptime Parent: type) type {

    if (comptime !isPointer(Parent)) {
        @compileError("Non-pointer type passed to pointer deduction.");
    }

    const info = @typeInfo(Parent).Pointer;

    return @Type(.{
        .Pointer = .{
            .size = info.size,
            .is_const = false,
            .is_volatile = info.is_volatile,
            .alignment = info.alignment,
            .address_space = info.address_space,
            .child = info.child,
            .is_allowzero = info.is_allowzero,
            .sentinel = info.sentinel,
        }
    });
}

// same as ceiling division - named after kernel macro
pub inline fn dimpad(n: anytype, m: @TypeOf(n)) @TypeOf(n) {
    return (n + (m - 1)) / m;
}

pub inline fn swap(x: anytype, y: @TypeOf(x)) void {
    if (comptime !isPointer(@TypeOf(x))) {
        @compileError("Swap requires pointer types");
    }
    const tmp = x.*;
    x.* = y.*;
    y.* = tmp;
}

pub inline fn isEven(x: anytype) bool {
    return (x & 1) == 0;
}

pub inline fn isOdd(x: anytype) bool {
    return !isEven(x);
}

pub fn product(comptime T: type, slice: []const T) T {
    var tmp: T = 1;
    for (slice) |val| {
        tmp *= val;
    }
    return tmp;
}

// calculates the number of windows with a given stride that fit in n
pub fn windowCount(n: usize, window_size: usize, stride: usize) usize {
    return ((n - window_size) / stride) + 1;
}

