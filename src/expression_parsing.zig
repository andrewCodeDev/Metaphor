////////////////////////////////////////////////////////////////
// Expression Parsing for Einsum style string expressions.

// Currently, the expression parser does not tolerate
// whitespace in expressions. This will be reviewed
// at a later date, but currently is not required to
// create well-formed strings.

// parser utility functions. These functions are intended
// to be executed at comptime.
pub fn symmetricDifference(
    comptime lhs: []const u8,
    comptime rhs: []const u8,
) []const u8 {
    const tot_len: usize = lhs.len + rhs.len;

    // inspired by LucasSantos91:
    //    https://ziggit.dev/t/algorithm-translation-challenge-1-symmetric-difference/2572/2
    
    // buffer to contain the results
    comptime var out_buf: [tot_len]u8 = .{ '0' } ** tot_len;
    // slices to resize and advance
    comptime var tmp_lhs = lhs;
    comptime var tmp_rhs = rhs;
    // final output length
    comptime var out_len: usize = tot_len;
    comptime var i: usize = 0;

    while (true) {
        if (tmp_lhs.len == 0) {
            @memcpy(out_buf[i..out_len], tmp_rhs);
            break;
        }
        if (tmp_rhs.len == 0) {
            @memcpy(out_buf[i..out_len], tmp_lhs);
            break;
        }
        switch (std.math.order(tmp_lhs[0], tmp_rhs[0])) {
            .lt => {
                out_buf[i] = tmp_lhs[0];
                tmp_lhs = tmp_lhs[1..];
                i += 1;
            },
            .gt => {
                out_buf[i] = tmp_rhs[0];
                tmp_rhs = tmp_rhs[1..];
                i += 1;
            },
            .eq => {
                tmp_lhs = tmp_lhs[1..];
                tmp_rhs = tmp_rhs[1..];
                out_len -= 2;
            },
        }
    }
    return out_buf[0..out_len];
}

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
    for (string) |c| {

        if (comptime !isAlphaLower(c))
            continue;

        min = @min(c, min);
    }
    comptime var buffer: [string.len]u8 = undefined;

    @memcpy(buffer[0..], string);

    const dif = if (min < 'i') 'i' - min else min - 'i';    

    for (string, 0..) |c, i| { 

        if (comptime !isAlphaLower(c))
            continue;
        
        buffer[i] = if (min < 'i') c + dif else c - dif;  
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

// Permutation parsing expects strings of the form:
//
//     example: ijk->jik
//
// The left and right operands must be alpha-characters.
// Both sides of the arrow operand must be permutations of eachother.
//

pub fn permutation(comptime str: []const u8) Permutation {

    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime translateIndices(trn[0..arrow.tail]);
    const rhs = comptime translateIndices(trn[arrow.head + 1 ..]);

    if (!comptime isPermutation(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of eachother." ++ str);
    }

    return comptime permutation_map.get(lhs ++ "->" ++ rhs)
        orelse Permutation.unknown;
}

pub fn permutateSizes(comptime str: []const u8) struct { perm: []const usize, len: usize }{

    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime translateIndices(trn[0..arrow.tail]);
    const rhs = comptime translateIndices(trn[arrow.head + 1 ..]);

    if (!allAlphaLower(lhs) or !allAlphaLower(rhs)) {
        @compileError("Tensor indices must be lower case characters: " ++ str);
    }
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

////////////////////////////////////////////
///////// INNER PRODUCT ////////////////////

// optimized permutation patterns

const InnerProduct = enum {
    unknown, @"ij,jk->ik",
};

const inner_product_map = std.ComptimeStringMap(
    InnerProduct, .{
        .{ @tagName(InnerProduct.@"ij,jk->ik"), InnerProduct.@"ij,jk->ik" }
    }
);

// Contraction parsing expects strings of the form:
//
//     example: ijk->jik
//
// The left and right operands must be alpha-characters.
// Both sides of the arrow operand must be permutations of eachother.
//

pub fn innerProduct(comptime str: []const u8) InnerProduct {

    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const comma = comptime findCommaOp(trn);
    const lhs = trn[0..comma];
    const rhs = trn[comma + 1..arrow.tail];
    const out = trn[arrow.head + 1..];

    if (!allAlphaLower(lhs) or !allAlphaLower(rhs) or !allAlphaLower(out)) {
        @compileError("Tensor indices must be lower case characters: " ++ str);
    }

    const sym_dif = comptime symmetricDifference(lhs, rhs);

    if (!comptime isPermutation(sym_dif, out)) {
        @compileError("Invalid inner product expression: " ++ str);
    }

    return comptime inner_product_map.get(lhs ++ "," ++ rhs ++ "->" ++ out)
        orelse inner_product_map.unknown;
}


pub fn innerProductSizes(comptime str: []const u8) struct { 
    x_map: []const ?usize, 
    y_map: []const ?usize, 
    len: usize 
}{
    const trn = translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const comma = comptime findCommaOp(trn);
    const lhs = comptime trn[0..comma];
    const rhs = comptime trn[comma + 1..arrow.tail];
    const out = comptime trn[arrow.head + 1..];

    // TODO: add checks for inner product

    comptime var x_map: [lhs.len]?usize = .{ null } ** lhs.len;
    comptime var y_map: [rhs.len]?usize = .{ null } ** rhs.len;

    // left hand side indices
    for (lhs, 0..) |l, i| {
        for (out, 0..) |o, j| {
            if (l == o) x_map[i] = j;
        }
    }

    // right hand side indices
    for (rhs, 0..) |r, i| {
        for (out, 0..) |o, j| {
            if (r == o) y_map[i] = j;
        }
    }

    return .{
        .x_map = x_map[0..],
        .y_map = y_map[0..],
        .len = out.len
    };
}
