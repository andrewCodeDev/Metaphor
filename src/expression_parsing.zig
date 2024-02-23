////////////////////////////////////////////////////////////////
// Expression Parsing for Einsum style string expressions.

// Currently, the expression parser does not tolerate
// whitespace in expressions. This will be reviewed
// at a later date, but currently is not required to
// create well-formed strings.

// parser utility functions. These functions are intended
// to be executed at comptime.

const std = @import("std");

pub fn between(comptime value: u8, comptime lower: u8, comptime upper: u8) bool {
    return lower <= value and value <= upper;
}

pub fn isAlphaLower(comptime value: u8) bool {
    return switch (value) {
        'a'...'z' => true, else => false,
    };
}

pub fn allAlphaLower(comptime str: []const u8) bool {
    comptime var i: usize = 0;
    inline while (i < str.len) : (i += 1) {
        if (!isAlphaLower(str[i])) {
            return false;
        }
    }
    return true;
}

pub fn contains(comptime char: u8, comptime string: []const u8) bool {
    comptime var i: usize = 0;
    inline while (i < string.len) : (i += 1) {
        if (char == string[i]) {
            return true;
        }
    }
    return false;
}

// check that a permutation is both full and accounted for
pub fn isPermutation(comptime source: []const u8, comptime target: []const u8) bool {
    if (source.len != target.len) {
        return false;
    }
    if (source.len == 0) { // the empty set is a permutation of itself
        return true;
    }
    // create mask for proper permutation
    const full: usize = (1 << source.len) - 1;
    comptime var i_mask: usize = 0;
    comptime var j_mask: usize = 0;

    comptime var i: usize = 0;
    comptime var j: usize = 0;
    inline while (i < source.len) : ({ i += 1; j = 0; }) {
        inline while (j < target.len) : (j += 1) {
            if (source[i] == target[j]) {
                i_mask |= (1 << i);
                j_mask |= (1 << j);
            }
        }
    }
    return i_mask == j_mask and i_mask == full;
}

pub fn countUniqueAlpha(comptime string: []const u8) usize {
    comptime var n: u64 = 0;
    comptime var i: usize = 0;
    inline while (i < string.len) : (i += 1) {
        if (isAlphaLower(string[i])) {
            n |= (1 << (string[i] - 65));
        }
    }
    return @popCount(n);
}

pub fn uniqueAlpha(comptime string: []const u8) [countUniqueAlpha(string)]u8 {
    const N = comptime countUniqueAlpha(string);
    comptime var i: usize = 0;
    comptime var j: usize = 0;
    comptime var chars: [N]u8 = .{0} ** N;
    inline while (i < string.len) : (i += 1) {
        if (comptime isAlphaLower(string[i]) and !contains(string[i], &chars)) {
            chars[j] = string[i];
            j += 1;
        }
    }
    return chars;
}

const ArrowOp = struct {
    tail: usize = 0,
    head: usize = 0,
};

pub fn findArrowOp(str: []const u8) ArrowOp {
    // reference for array operator
    const arrow: []const u8 = "->";

    comptime var head: usize = 0;
    comptime var tail: usize = 0;
    comptime var index: usize = 0;
    inline while (index < str.len) : (index += 1) {
        if (str[index] == arrow[0]) {
            tail = index;
        }
        if (str[index] == arrow[1]) {
            head = index;
        }
    }
    if ((tail + 1) != head) {
        @compileError("Malformed arrow operator: " ++ str);
    }
    if (tail == 0 or head > (str.len - 2)) {
        @compileError("Arrow must be used as infix operator: " ++ str);
    }
    return ArrowOp{ .tail = tail, .head = head };
}

pub fn findCommaOp(str: []const u8) usize {
    comptime var comma: usize = 0;
    comptime var index: usize = 0;
    inline while (index < str.len) : (index += 1) {
        if (str[index] == ',') {
            comma = index;
            break;
        }
    }
    if (comma == 0 or comma >= (str.len - 1)) {
        @compileError("Comma must be used as infix operator: " ++ str);
    }
    return comma;
}

// this function disambiguates index characters
pub fn translateIndices(comptime string: []const u8) []const u8 {
    comptime var min: u8 = std.math.maxInt(u8);
    for (string) |char| {
        if (comptime !isAlphaLower(char)) {
            @compileError("Only lower case alpha characters permitted for indices.");
        }
        min = @min(char, min);
    }
    comptime var buffer: [string.len]u8 = undefined;

    if (min < 'i') {
        const dif = 'i' - min;
        for (string, 0..) |char, i| { buffer[i] = char + dif;  }
    } else {
        const dif = min - 'i';
        for (string, 0..) |char, i| { buffer[i] = char - dif;  }
    }
    return buffer[0..];
}

////////////////////////////////////////////
///////// PERMUTATIONS /////////////////////

// optimized permutation patterns

const Permutation = enum {
    unknown, @"ij->ji",
};

const permutation_map = std.ComptimeStringMap(
    Permutation, .{
        .{ @tagName(Permutation.@"ij->ji"), Permutation.@"ij->ji" }
    }
);

// Contraction parsing is expects strings of the form:
//
//     example: ijk->jik
//
// The left and right operands must be alpha-characters.
// Both sides of the arrow operand must be permutations of eachother.
//

pub fn permutation(comptime str: []const u8) Permutation {

    const arrow = comptime findArrowOp(str);
    const lhs = comptime translateIndices(str[0..arrow.tail]);
    const rhs = comptime translateIndices(str[arrow.head + 1 ..]);

    if (!comptime isPermutation(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of eachother." ++ str);
    }

    return comptime permutation_map.get(lhs ++ "->" ++ rhs)
        orelse Permutation.unknown;
}

pub fn permutateSizes(comptime str: []const u8) struct { perm: []const usize, len: usize }{

    const arrow = comptime findArrowOp(str);
    const lhs = comptime translateIndices(str[0..arrow.tail]);
    const rhs = comptime translateIndices(str[arrow.head + 1 ..]);

    if (!comptime isPermutation(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of eachother." ++ str);
    }

    comptime var indices: [rhs.len]usize = undefined;

    for (lhs, 0..) |l, i| {
        for (rhs, 0..) |r, j| {
            if (l == r) indices[i] = j;
        }
    }
    return .{
        .perm = indices[0..],
        .len = lhs.len
    };
}
