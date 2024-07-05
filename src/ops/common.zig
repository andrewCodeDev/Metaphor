// standard utilities for deriving operations

const std = @import("std");

pub inline fn eps_equal(x: f64, y: f64) bool {
    return @abs(x - y) < 1e-8; // or something
}
pub inline fn is_zero(x: f64) bool {
    return eps_equal(x, 0.0);
}
pub inline fn is_one(x: f64) bool {
    return eps_equal(x, 1.0);
}

pub fn commute(expr: []const u8, infix: []const u8, allocator: std.mem.Allocator) []const u8 {

    std.debug.assert(expr.len > 0);
    std.debug.assert(infix.len > 0);

    const dupe = allocator.alloc(u8, expr.len) catch @panic("Failed to allocate expression.");

    const i = std.mem.indexOf(u8, expr, infix) orelse unreachable;

    const lhs = expr[0..i];
    const rhs = expr[i + infix.len..];

    std.debug.assert(lhs.len > 0);
    std.debug.assert(rhs.len > 0);

    @memcpy(dupe[0..rhs.len], rhs);
    @memcpy(dupe[rhs.len..][0..infix.len], infix);
    @memcpy(dupe[rhs.len..][infix.len..], lhs);

    return dupe;
}

// assumes expressions as "...->..."
pub fn arrow_position(expr: []const u8) struct { head: usize, tail: usize }{

    const head: usize = std.mem.indexOfScalar(u8, expr, '-') orelse unreachable;

    std.debug.assert(expr[head + 1] == '>');

    return .{ .head = head, .tail = head + 1 };
}

pub inline fn invoke(comptime kernels: anytype, key: usize, args: anytype) void {
    switch (key) {
        0 => @call(.auto, kernels[0], args),
        1 => @call(.auto, kernels[1], args),
        2 => @call(.auto, kernels[2], args),
        else => unreachable,
    }
}
