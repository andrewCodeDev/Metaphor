////////////////////////////////////////////////////////////////
// Expression Parsing for Einsum style string expressions.

// Currently, the expression parser does not tolerate
// whitespace in expressions. This will be reviewed
// at a later date, but currently is not required to
// create well-formed strings.

// parser utility functions. These functions are intended
// to be executed at comptime.
pub fn symmetricDifference(comptime lhs: []const u8, comptime rhs: []const u8) []const u8 {

    const N = 26;

    comptime var lhs_bits = std.StaticBitSet(N).initEmpty();
    comptime var rhs_bits = std.StaticBitSet(N).initEmpty();

    comptime var diff: [N]u8 = undefined;

    for (lhs) |c| lhs_bits.setValue(c - 'a', true);
    for (rhs) |c| rhs_bits.setValue(c - 'a', true);

    const union_bits = lhs_bits.unionWith(rhs_bits);
    const inter_bits = lhs_bits.intersectWith(rhs_bits);
    const symdif_bits = union_bits.differenceWith(inter_bits);

    comptime var cur: u8 = 'a';
    comptime var len: usize = 0;

    while (cur <= 'z') : (cur += 1){
        if (symdif_bits.isSet(cur - 'a')) {
            diff[len] = cur;
            len += 1;
        }
    }
    const _diff = diff;
    return _diff[0..len];
}

const std = @import("std");

pub fn between(comptime value: u8, comptime lower: u8, comptime upper: u8) bool {
    return lower <= value and value <= upper;
}

pub fn isAlphaLower(comptime value: u8) bool {
    return switch (value) {
        'a'...'z' => true,
        else => false,
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

// check that a permutation is both complete and has unique elements
pub fn isPermutationUnique(comptime lhs: []const u8, comptime rhs: []const u8) bool {
    if (lhs.len != rhs.len) {
        return false;
    }
    if (lhs.len == 0) { // the empty set is a permutation of itself
        return true;
    }

    const N = 26;

    comptime var lhs_bits = std.StaticBitSet(N).initEmpty();
    comptime var rhs_bits = std.StaticBitSet(N).initEmpty();

    for (lhs) |c| lhs_bits.setValue(c - 'a', true);
    for (rhs) |c| rhs_bits.setValue(c - 'a', true);

    // check for repeats because or'ing will compress repeats
    const all_unique: bool = (lhs.len == lhs_bits.count()) and (rhs.len == rhs_bits.count());

    // order of bits doesn't matter, but they have to be the same
    return all_unique and (lhs_bits.eql(rhs_bits));
}

// check that a permutation is both complete and has unique elements
pub fn isSubset(comptime sub: []const u8, comptime set: []const u8) bool {
    const N = 26;

    comptime var lhs_bits = std.StaticBitSet(N).initEmpty();
    comptime var rhs_bits = std.StaticBitSet(N).initEmpty();

    for (sub) |c| lhs_bits.setValue(c - 'a', true);
    for (set) |c| rhs_bits.setValue(c - 'a', true);

    return lhs_bits.subsetOf(rhs_bits);

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
    return .{ .tail = tail, .head = head };
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
    const _buffer = buffer;
    return _buffer[0..];
}

////////////////////////////////////////////
///////// PERMUTATIONS /////////////////////

// optimized permutation patterns

// Permutation parsing expects strings of the form:
//
//     example: ijk->jik
//
// The left and right operands must be alpha-characters.
// Both sides of the arrow operand must be permutations of each other.
//

pub fn permutationExpression(comptime str: []const u8) []const u8 {
    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime translateIndices(trn[0..arrow.tail]);
    const rhs = comptime translateIndices(trn[arrow.head + 1 ..]);

    if (!comptime isPermutationUnique(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of each other." ++ str);
    }

    return comptime lhs ++ "->" ++ rhs;
}

pub fn permutateSizes(comptime str: []const u8) struct { perm: []const usize, len: usize } {
    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime translateIndices(trn[0..arrow.tail]);
    const rhs = comptime translateIndices(trn[arrow.head + 1 ..]);

    if (!allAlphaLower(lhs) or !allAlphaLower(rhs)) {
        @compileError("Tensor indices must be lower case characters: " ++ str);
    }
    if (!comptime isPermutationUnique(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of each other." ++ str);
    }

    comptime var indices: [rhs.len]usize = undefined;

    for (lhs, 0..) |l, i| {
        for (rhs, 0..) |r, j| {
            if (l == r) indices[i] = j;
        }
    }
    const _indices = indices;
    
    return .{ .perm = _indices[0..], .len = lhs.len };
}

////////////////////////////////////////////
///////// INNER PRODUCT ////////////////////

// Inner product parsing expects strings of the form:
//
//     example: ij,jk->ik
//
// The left, right, and output operands must be alpha-characters.
// Output string must be the symetric difference of left and right operands.
// There must be common indicies in the left and right operands.
//

pub fn innerProductExpression(comptime str: []const u8) []const u8 {
    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const comma = comptime findCommaOp(trn);
    const lhs = trn[0..comma];
    const rhs = trn[comma + 1 .. arrow.tail];
    const out = trn[arrow.head + 1 ..];

    if (!allAlphaLower(lhs) or !allAlphaLower(rhs) or !allAlphaLower(out)) {
        @compileError("Tensor indices must be lower case characters: " ++ str);
    }

    const sym_dif = comptime symmetricDifference(lhs, rhs);

    if (!comptime isPermutationUnique(sym_dif, out)) {
        @compileError("Invalid inner product expression: " ++ str);
    }

    return lhs ++ "," ++ rhs ++ "->" ++ out;
}

pub fn innerProductSizes(comptime str: []const u8) struct { x_map: []const ?usize, y_map: []const ?usize, len: usize } {
    const trn = translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const comma = comptime findCommaOp(trn);
    const lhs = comptime trn[0..comma];
    const rhs = comptime trn[comma + 1 .. arrow.tail];
    const out = comptime trn[arrow.head + 1 ..];

    // TODO: add checks for inner product

    comptime var x_map: [lhs.len]?usize = .{null} ** lhs.len;
    comptime var y_map: [rhs.len]?usize = .{null} ** rhs.len;

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
    const _x_map = x_map;
    const _y_map = x_map;
    return .{ .x_map = _x_map[0..], .y_map = _y_map[0..], .len = out.len };
}

////////////////////////////////////////////////
// Softmax expression parser

pub fn softmaxExpression(comptime str: []const u8) []const u8 {
    const pipe = comptime std.mem.indexOfScalar(u8, str, '|') orelse @compileError("No pipe operator found on softmax.");

    if (comptime pipe != str.len - 2) {
        @compileError("Softmax requires that right hand index specifies either a row or column:" ++ str);
    }
    const trn = comptime translateIndices(str);
    const lhs = comptime trn[0..pipe];
    const rhs = comptime trn[pipe + 1 ..];

    if (comptime lhs.len <= rhs.len) {
        @compileError("Softmax found extra indices in expression:" ++ str);
    }
    if (comptime !isSubset(rhs, lhs)) {
        @compileError("Softmax requires that right hand index is a subset of left-hand indices:" ++ str);
    }

    // TODO: More checks plz...

    return comptime lhs ++ "|" ++ rhs;
}

////////////////////////////////////////////////
// Reduce expression parser

pub fn reduceSizes(comptime str: []const u8) struct { x_map: []const ?usize, len: usize } {
    const trn = translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime trn[0..arrow.tail];
    const rhs = comptime trn[arrow.head + 1..];

    // TODO: add checks for inner product

    comptime var x_map: [lhs.len]?usize = .{null} ** lhs.len;

    // left hand side indices
    for (lhs, 0..) |l, i| {
        for (rhs, 0..) |r, j| {
            if (l == r) x_map[i] = j;
        }
    }
    const _x_map = x_map;
    return .{ .x_map = _x_map[0..], .len = rhs.len };
}

pub fn reduceExpression(comptime str: []const u8) []const u8 {
    const trn = comptime translateIndices(str);
    const arrow = comptime findArrowOp(trn);
    const lhs = comptime trn[0..arrow.tail];
    const rhs = comptime trn[arrow.head + 1 ..];

    if (comptime lhs.len < rhs.len) {
        @compileError("Reduce found extra indices in expression:" ++ str);
    }
    if (comptime !isSubset(rhs, lhs)) {
        @compileError("Reduce requires that right hand index is a subset of left-hand indices:" ++ str);
    }

    // TODO: More checks plz...

    return comptime lhs ++ "->" ++ rhs;
}
