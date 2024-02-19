////////////////////////////////////////////////////////////////
// Expression Parsing for Einsum style string expressions.

// Currently, the expression parser does not tolerate
// whitespace in expressions. This will be reviewed
// at a later date, but currently is not required to
// create well-formed strings.

// parser utility functions. These functions are intended
// to be executed at comptime.

const SizeType = usize;

const std = @import("std");

pub fn between(comptime value: u8, comptime lower: u8, comptime upper: u8) bool {
    return lower <= value and value <= upper;
}

pub fn isAlpha(comptime value: u8) bool {
    return between(value, 65, 90) or between(value, 97, 122); // [91, 96] are: [\]^_`
}

pub fn allAlpha(comptime str: [] const u8) bool {
    comptime var i: usize = 0;
    inline while(i < str.len) : (i += 1) {
        if(!isAlpha(str[i])) { return false; }
    }
    return true;
}

pub fn contains(comptime char: u8, comptime string: [] const u8) bool {
    comptime var i: usize = 0;
    inline while(i < string.len) : (i += 1) {
        if(char == string[i]) { return true; }
    }
    return false;
}

// check that a permutation is both full and accounted for
pub fn isPermutation(comptime source: [] const u8, comptime target: [] const u8) bool {

    if(source.len != target.len) {
        return false;
    }
    if(source.len == 0) { // the empty set is a permutation of itself
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

pub fn countUniqueAlpha(comptime string: [] const u8) usize {
    comptime var n: u64 = 0;    
    comptime var i: usize = 0;
    inline while(i < string.len) : (i += 1) {
        if(isAlpha(string[i])) { n |= (1 << (string[i] - 65)); }
    }
    return @popCount(n);
}

pub fn uniqueAlpha(comptime string: [] const u8) [countUniqueAlpha(string)]u8 {
    const N = comptime countUniqueAlpha(string);
    comptime var i: usize = 0;
    comptime var j: usize = 0;
    const chars: [N]u8 = .{0} ** N;
    inline while(i < string.len) : (i += 1) {
        if(comptime isAlpha(string[i]) and !contains(string[i], &chars)) { 
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

pub fn findArrowOp(str: [] const u8) ArrowOp { 
    // reference for array operator
    const arrow: [] const u8 = "->";

    comptime var head: usize = 0;
    comptime var tail: usize = 0;    
    comptime var index: usize = 0;
    inline while(index < str.len) : (index += 1) {
        if(str[index] == arrow[0]) { tail = index; }
        if(str[index] == arrow[1]) { head = index; }
    }
    if((tail + 1) != head) {
        @compileError("Malformed arrow operator: " ++ str);
    }
    if(tail == 0 or head > (str.len - 2)) {
        @compileError("Arrow must be used as infix operator: " ++ str);
    }
    return ArrowOp{ .tail = tail, .head = head };
}

pub fn findCommaOp(str: [] const u8) usize { 
    comptime var comma: usize = 0;
    comptime var index: usize = 0;
    inline while(index < str.len) : (index += 1) {
        if(str[index] == ","[0]) { comma = index; break; }
    }
    if(comma == 0 or comma >= (str.len - 1)) {
        @compileError("Comma must be used as infix operator: " ++ str);
    }
    return comma;
}

const PermutationRank = enum {
    rank_2,
    rank_3,
    rank_4,
    rank_n
};

pub fn isWhitespace(comptime c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C  => true, else => false,
    };
}

pub fn trim(comptime str: [] const u8) [] const u8 {
    if (str.len == 0) {
        return;
    } else if (str.len == 1) {
        if (comptime isWhitespace(str[0])) {
            return "";
        }
    }
    
    comptime var i: usize = 0;
    inline while (i < str.len) : (i += 1) {
        if (!isWhitespace(str[i])) { break; }
    }

    if (i == str.len) { return ""; }
    
    comptime var j: usize = str.len;
    inline while (j > 0) {
        j -= 1; 
        if (!isWhitespace(str[j])) { break; }
        else if (j == 0) { return ""; }
    }    
    return str[i..j + 1];    
}

fn TransposePlan(comptime Rank: usize) type {
    return struct {
        rank: usize = Rank,
        perm: [Rank]usize,
    };
}

pub fn transposeParse(comptime str: [] const u8) 
    TransposePlan(countUniqueAlpha(str))  {

    const arrow = comptime findArrowOp(str);
    const lhs = trim(str[0..arrow.tail]);
    const rhs = trim(str[arrow.head + 1..]);

    if(!comptime allAlpha(lhs)) {
        @compileError("Non-alphabetical character found in: " ++ lhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }
    if(comptime countUniqueAlpha(lhs) < lhs.len) {
        @compileError("Duplicate characters found in: " ++ lhs);
    }
    if(comptime countUniqueAlpha(rhs) < rhs.len) {
        @compileError("Duplicate characters found in: " ++ rhs);
    }
    if(!comptime isPermutation(lhs, rhs)) {
        @compileError("Permutate requires left and right operands to be permutations of eachother." ++ str);
    }

    ////////////////////////////////////////
    // deduce permutation type for dispatch

    if (lhs.len == 2) {
        return TransposePlan(lhs.len){ 
            .rank = lhs.len, .perm = [lhs.len]usize{ 1, 0 } 
        };
    }
    
    comptime var perm: [lhs.len]usize = undefined;

    comptime var i: usize = 0;
    comptime var j: usize = 0;
    inline while(i < lhs.len) : ({ i += 1; j = 0; }) {
        inline while(j < rhs.len) : (j += 1) {
            if (rhs[i] == lhs[j]) {
                perm[i] = j;
                break;
            }
        }
    }        
    return TransposePlan(lhs.len){ 
        .rank = lhs.len, .perm = perm 
    };
}

// Contraction parsing is expects strings of the form:
//
//     example: ijk->jk
//
// The expressions must be larger on the left-operand than
// the right operand (denoting contracted indices).
//
// The left and right operands must be alpha-characters.

pub fn contractedRank(comptime str: [] const u8) usize {
    return (str.len - (comptime findArrowOp(str)).head) - 1;
}

//pub fn ContractionPlan(comptime lRank: usize, comptime rRank: usize) type {
//    return struct {
//        lhs : [lRank]SizeType = usize;usize;comptime lRank: usize,
//    comptime rRank: usize,
//    comptime str: [] const u8
//    ) ContractionPlan(lRank, rRank) {
//
//    comptime var index: usize = 0;
//
//    const arrow = comptime findArrowOp(str);
//    const lhs = str[0..arrow.tail];
//    const rhs = str[arrow.head + 1..];
//
//    if (lhs.len == 0) {
//        @compileError("Empty left-side operand: " ++ str);
//    }
//    if (rhs.len == 0) {
//        @compileError("Empty right-side operand: " ++ str);
//    }
//    if(lhs.len != lRank) {
//        @compileError("Provided indices do not match left-side operand rank: " ++ lhs);
//    }
//    if(rhs.len != rRank) {
//        @compileError("Provided indices do not match right-side operand rank: " ++ rhs);
//    }
//    if(!comptime allAlpha(lhs)) {
//        @compileError("Non-alphabetical character found in: " ++ lhs);
//    }
//    if(!comptime allAlpha(rhs)) {
//        @compileError("Non-alphabetical character found in: " ++ rhs);
//    }
//
//    ////////////////////////////////////////
//    // build permutation contraction indices
//
//    comptime var x_indices: [lhs.len]u32 = undefined;
//    comptime var y_indices: [rhs.len]u32 = undefined;
//    comptime var remainder: [lhs.len + rhs.len]u32 = undefined;
//    comptime var char: u8 = undefined;
//    comptime var match: u32 = 0;
//    comptime var rhs_i: u32 = 0;
//    comptime var rem_i: u32 = 0;
//    comptime var found: bool = false;
//
//    index = 0;
//    inline while(index < lhs.len) : (index += 1) {
//
//        // matched + unmatched = total
//        if(match == rhs.len and rem_i == remainder.len) {
//             break; 
//        }
//
//        char = lhs[index];
//
//        found = false;
//
//        // try to match the current char
//        // in both rhs and lhs operands
//        
//        rhs_i = 0;
//        inline while(rhs_i < rhs.len) : (rhs_i += 1) {
//            if (rhs[rhs_i] == char) {
//                x_indices[match] = index;
//                y_indices[match] = rhs_i;
//                found = true;
//                match += 1;
//                break;
//            }
//        }
//
//        // if no match, add to remainder
//        
//        if(!found) {
//            remainder[rem_i] = index;
//            rem_i += 1;
//        }
//    }
//
//    if(match != rhs.len) {
//        @compileError("Unmatched dimensions between operands:" ++ str);
//    }
//
//    rem_i = 0;
//    index = rhs.len;
//    inline while(index < lhs.len) : ({ index += 1; rem_i += 1; }){
//        x_indices[index] = remainder[rem_i];
//    }   
//    return ContractionPlan(lRank, rRank){ .lhs = x_indices, .rhs = y_indices };
//}

///////////////////////
//// Inner-Product ////

pub fn InnerProductPlan(comptime N: usize) type {

    const pass_flag: usize = 9999;
    
    return struct {
        pass: usize = pass_flag,
        x_perm: [N]usize = .{ pass_flag } ** N,
        y_perm: [N]usize = .{ pass_flag } ** N,
        z_perm: [N]usize = .{ pass_flag } ** N,
        s_ctrl: [N]usize = .{ pass_flag } ** N,
        total: usize = N,
    };
}

pub fn innerProductParse(
    comptime XRank: usize,
    comptime YRank: usize,
    comptime ZRank: usize,
    comptime expression: [] const u8
    ) InnerProductPlan(countUniqueAlpha(expression)) {

    const arrow = comptime findArrowOp(expression);
    const comma = comptime findCommaOp(expression);

    if(comma >= (arrow.tail - 1)) {
        @compileError("Comma operator must come before left operand: " ++ expression);
    }

    const lhs = expression[0..comma];
    const rhs = expression[comma+1..arrow.tail];
    const out = expression[arrow.head+1..];

    if(lhs.len == 0) {
        @compileError("Empty left-side operand: " ++ expression);
    }
    if(rhs.len == 0) {
        @compileError("Empty right-side operand: " ++ expression);
    }
    if(out.len == 0) {
        @compileError("Empty expression result: " ++ expression);
    }
    if(lhs.len != XRank) {
        @compileError("Provided indices do not match left-side operand rank: " ++ lhs);
    }
    if(rhs.len != YRank) {
        @compileError("Provided indices do not match right-side operand rank: " ++ rhs);
    }
    if(out.len != ZRank) {
        @compileError("Provided indices do not match result rank: " ++ out);
    }
    if(!comptime allAlpha(lhs)) {
        @compileError("Non-alphabetical character found in: " ++ lhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }
    if(!comptime allAlpha(out)) {
        @compileError("Non-alphabetical character found in: " ++ out);
    }

    ////////////////////////////////////////
    // build inner product control indices

    const N = countUniqueAlpha(expression);

    comptime var plan = InnerProductPlan(N){ };

    // loop index variables
    comptime var i = 0;
    comptime var j = 0;    
    const chars = comptime uniqueAlpha(expression);

    i = 0;
    inline while(i < N) : (i += 1) {
        j = 0;
        inline while(j < lhs.len) : (j += 1) {
            if(lhs[j] == chars[i]) { 
                plan.x_perm[i] = j;
                plan.s_ctrl[i] = 0;
            }
        }
        j = 0;
        inline while(j < rhs.len) : (j += 1) {
            if(rhs[j] == chars[i]) { 
                plan.y_perm[i] = j;
                plan.s_ctrl[i] = 1;
            }
        }
        j = 0;
        inline while(j < out.len) : (j += 1) {
            if(out[j] == chars[i]) { 
                plan.z_perm[i] = j;
            }
        }
    }
    return plan;
}


pub fn OuterProductPlan(comptime N: usize) type {

    const pass_flag: usize = 9999;
    
    return struct {
        pass: usize = pass_flag,
        x_perm: [N]usize = .{ pass_flag } ** N,
        y_perm: [N]usize = .{ pass_flag } ** N,
        z_perm: [N]usize = .{ pass_flag } ** N,
        total: usize = N,
    };
}

pub fn outerProductParse(
    comptime XRank: usize,
    comptime YRank: usize,
    comptime ZRank: usize,
    comptime expression: [] const u8
    ) OuterProductPlan(countUniqueAlpha(expression)) {

    const arrow = comptime findArrowOp(expression);
    const comma = comptime findCommaOp(expression);

    if(comma >= (arrow.tail - 1)) {
        @compileError("Comma operator must come before left operand: " ++ expression);
    }

    const lhs = expression[0..comma];
    const rhs = expression[comma+1..arrow.tail];
    const out = expression[arrow.head+1..];

    if(lhs.len == 0) {
        @compileError("Empty left-side operand: " ++ expression);
    }
    if(rhs.len == 0) {
        @compileError("Empty right-side operand: " ++ expression);
    }
    if(out.len == 0) {
        @compileError("Empty expression result: " ++ expression);
    }
    if(lhs.len != XRank) {
        @compileError("Provided indices do not match left-side operand rank: " ++ lhs);
    }
    if(rhs.len != YRank) {
        @compileError("Provided indices do not match right-side operand rank: " ++ rhs);
    }
    if(out.len != ZRank) {
        @compileError("Provided indices do not match result rank: " ++ out);
    }
    if(!comptime allAlpha(lhs)) {
        @compileError("Non-alphabetical character found in: " ++ lhs);
    }
    if(!comptime allAlpha(rhs)) {
        @compileError("Non-alphabetical character found in: " ++ rhs);
    }
    if(!comptime allAlpha(out)) {
        @compileError("Non-alphabetical character found in: " ++ out);
    }

    ////////////////////////////////////////
    // build inner product control indices

    const N = countUniqueAlpha(expression);

    comptime var plan = OuterProductPlan(N){ };

    // loop index variables
    comptime var i = 0;
    comptime var j = 0;    
    const chars = comptime uniqueAlpha(expression);

    i = 0;
    inline while(i < N) : (i += 1) {
        j = 0;
        inline while(j < lhs.len) : (j += 1) {
            if(lhs[j] == chars[i]) { 
                plan.x_perm[i] = j;
            }
        }
        j = 0;
        inline while(j < rhs.len) : (j += 1) {
            if(rhs[j] == chars[i]) { 
                plan.y_perm[i] = j;
            }
        }
        j = 0;
        inline while(j < out.len) : (j += 1) {
            if(out[j] == chars[i]) { 
                plan.z_perm[i] = j;
            }
        }
    }
    return plan;
}
