// Special thanks is given to the contributors at the Ziggit forum for their help and feedback.
// To see the discussion, check out: https://ziggit.dev/t/making-overloaded-function-sets-using-comptime/2475

// Special thanks to Sze for much of the implementation below. To see the original file,
// check out: https://gist.github.com/SimonLSchlee/32d6a9a66de9e74797c5a935350ac996

const std = @import("std");
const SC = @import("scalar.zig");
const UT = @import("utility.zig");

pub fn OverloadSet(comptime def: anytype) type {
    if (comptime detectArgsError(def)) |error_message| {
        @compileError(error_message);
    }
    return struct {
        fn candidatesMessage() []const u8 {
            var msg: []const u8 = "";
            for (def) |f| {
                const T = @TypeOf(f);
                msg = msg ++ "    " ++ @typeName(T) ++ "\n";
            }
            return msg;
        }

        fn formatArguments(comptime args_type: type) []const u8 {
            const params = @typeInfo(args_type).Struct.fields;
            var msg: []const u8 = "{ ";
            for (params, 0..) |arg, i| {
                msg = msg ++ @typeName(arg.type) ++ if (i < params.len - 1) ", " else "";
            }
            return msg ++ " }";
        }

        pub fn call(args: anytype) GetResultType(findMatchingFunctionType(def, @TypeOf(args)).result) {
            const args_type = @TypeOf(args);
            if (comptime !UT.isTuple(args_type)) {
                @compileError("OverloadSet's call argument must be a tuple.");
            }
            if (comptime findMatchingFunctionType(def, args_type).result == noreturn) {
                @compileError("No overload for " ++ formatArguments(args_type) ++ "\n" ++ "Candidates are:\n" ++ candidatesMessage());
            }
            return @call(.auto, findMatchingFunction(def, args_type), args);
        }
    };
}

// check that the param's constness is the max of the two arguments
fn constCompatible(arg_const: bool, param_const: bool) bool {
    return (arg_const or param_const) == param_const;
}

fn sizeCompatible(
    n: std.builtin.Type.Pointer.Size, 
    m: std.builtin.Type.Pointer.Size, 
) bool {
    return (n == m)
        or ((n == .One or n == .Many) and (m == .C))
        or ((m == .One or m == .Many) and (n == .C));
}

const Match = struct {
    result: type = noreturn,
    indices: []const usize = &.{},
};

fn findMatchingFunctionType(comptime def: anytype, comptime args_type: type) Match {
    const args_fields = @typeInfo(args_type).Struct.fields;
    const func_fields = @typeInfo(@TypeOf(def)).Struct.fields;

    // track each function with a score. the score is based on how
    // each argument matches the parameter at the same position.
    // a positive score for that parameter means zig will accept it.
    // any failed match sets the score to null and drops that overload
    comptime var scores: [func_fields.len]?usize = .{0} ** func_fields.len;

    // scan for the best fit match
    comptime var max_score: usize = 0;
    comptime var match: Match = .{};

    for (def, 0..func_fields.len) |function, i| {
        const params = @typeInfo(@TypeOf(function)).Fn.params;

        if (params.len != args_fields.len) {
            scores[i] = null;
            continue;
        }

        // allow functions with no parameters
        if (params.len == 0 and args_fields.len == 0) {
            return @TypeOf(def[i]);
        }

        // track our previous score - the only way the score can go up
        // is one of the valid branches of the switch increments it.
        // if no branches are valid, the score stays the same.
        const prev_score = scores[i].?;

        for (params, args_fields) |param, field| {
            if (scores[i] == null)
                break;

            switch (@typeInfo(field.type)) {
                .Pointer => |f_ptr| {
                    switch (@typeInfo(param.type.?)) {
                        .Pointer => |p_ptr| {
                            if (p_ptr.child == f_ptr.child and
                                sizeCompatible(p_ptr.size, f_ptr.size) and
                                p_ptr.alignment == f_ptr.alignment)
                            {

                                // const compatible is a valid match, but
                                // exact const scores higher (2 instead of 1)
                                if (constCompatible(f_ptr.is_const, p_ptr.is_const)) {
                                    scores[i].? += if (f_ptr.is_const == p_ptr.is_const) 2 else 1;
                                }
                            }
                        },
                        else => {},
                    }
                },
                else => {
                    if (param.type.? == field.type) scores[i].? += 1;
                },
            }

            // if we did not increase the score, then
            // we didn't match on our last parameter.
            if (scores[i].? == prev_score) {
                scores[i] = null;
                break;
            }
        }

        // track our max best score - the only way it could not
        // be null is if each parameter was matched and scored
        if (scores[i] != null and max_score <= scores[i].?) {
            if (max_score < scores[i].?) {
                match.indices = &.{};
            }
            max_score = scores[i].?;
            match.indices = match.indices ++ @as([]const usize, &.{i});
        }
    }

    if (match.indices.len == 1 and match.indices[0] < func_fields.len) {
        match.result = @TypeOf(def[match.indices[0]]);
    }

    return match;
}

fn findMatchingFunction(comptime def: anytype, comptime args_type: type) findMatchingFunctionType(def, args_type).result {
    const match = findMatchingFunctionType(def, args_type);
    return for (def) |function| {
        if (@TypeOf(function) == match.result) break function;
    } else noreturn;
}

fn GetResultType(comptime function: type) type {
    if (function == noreturn) return noreturn;
    return @typeInfo(function).Fn.return_type.?;
}

fn isFunctionsTuple(args: anytype) bool {
    return UT.isTuple(@TypeOf(args)) and for (args) |arg| {
        const T = @TypeOf(arg);
        if (!UT.isFunction(T)) break false;
    } else true;
}

fn detectArgsError(comptime args: anytype) ?[]const u8 {
    if (!isFunctionsTuple(args)) {
        return "Non-function argument in overload set.";
    }
    for (args) |f| {
        const T = @TypeOf(f);
        for (@typeInfo(T).Fn.params) |param| {
            if (param.type == null) {
                return "Generic parameter types in overload set.";
            }
        }
    }
    for (0..args.len) |i| {
        const T0 = @TypeOf(args[i]);
        const params0 = @typeInfo(T0).Fn.params;
        for (i + 1..args.len) |j| {
            const T1 = @TypeOf(args[j]);
            const params1 = @typeInfo(T1).Fn.params;
            const signatures_are_identical = params0.len == params1.len and
                for (params0, params1) |param0, param1| {
                    if (param0.type != param1.type) break false;
                } else true;
            if (signatures_are_identical) {
                return "Identical function signatures in overload set.";
            }
        }
    }
    return null;
}

test "best-type-matching-1-param" {

    const foo = struct {
        pub fn call1(_: []const u8) bool { return false; }  
        pub fn call2(_: []u8) bool { return true; }  
    };

    const os = OverloadSet(.{ foo.call1, foo.call2 });

    const x: []u8 = undefined;

    const result = os.call(.{ x });

    try std.testing.expect(result);    
}

test "best-type-matching-N-param" {

    const foo = struct {
        pub fn call1( // worse match
            _: []const u8, 
            _: []const u8, 
            _: []const u8, 
            _: usize
        ) bool {
             return false; 
        }  
        pub fn call2( // better match
            _: []const u8, 
            _: []u8, 
            _: []const u8, 
            _: usize
        ) bool {
             return true; 
        }  
        pub fn call3( // no match
            _: []u8, 
            _: []u8, 
            _: []const u8, 
            _: usize
        ) bool {
             return false; 
        }  
    };

    const os = OverloadSet(.{ foo.call1, foo.call2, foo.call3 });

    const x: []const u8 = undefined;
    const y: []u8 = undefined;
    const z: []const u8 = undefined;
    const w: usize = undefined;

    const result = os.call(.{ x, y, z, w });

    try std.testing.expect(result);    
}


test "non-const-promotion" {

    const foo = struct {
        pub fn call(_: []const u8) bool { return true; }  
    };

    const os = OverloadSet(.{ foo.call });

    const x: []u8 = undefined;

    const result = os.call(.{ x });

    try std.testing.expect(result);    
}

test "empty-parameter-call" {

    const foo = struct {
        pub fn call1(_: u8) bool { return false; }
        pub fn call2() bool { return true; }  
    };

    const os = OverloadSet(.{ foo.call1, foo.call2 });

    const result = os.call(.{ });

    try std.testing.expect(result);    
}

test "cannot-demote-const" {

    const foo = struct {
        pub fn call1(_: []u8) bool { return false; }  
        pub fn call2(_: []const u8) bool { return true; }  
    };

    const os = OverloadSet(.{ foo.call1, foo.call2 });

    const x: []const u8 = undefined;

    const result = os.call(.{ x });

    try std.testing.expect(result);    
}

test "multi-parameter-matching" {

    const foo = struct {
        pub fn call(_: u8, _: []u8) bool { return true; }  
    };

    const os = OverloadSet(.{ foo.call });

    const x: u8 = undefined;
    const y: []u8 = undefined;

    const result = os.call(.{ x, y });

    try std.testing.expect(result);    
}

