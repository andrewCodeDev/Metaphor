
//////////////////////////////////////////////////////////
////////////////////// EXAMPLE ///////////////////////////
//////////////////////////////////////////////////////////
//
// Forward Function:
//     fn fooImpl(x: usize, y: usize) void { std.debug.print("\nFoo: {}, {}\n", .{ x, y }); }
//
// Reverse Functions:
//     fn barImpl(x: usize, _: usize) void { std.debug.print("\nBar: {}\n", .{ x }); }
//     fn bazImpl(_: usize, y: usize) void { std.debug.print("\nBaz: {}\n", .{ y }); }
//
// Build Callback:
//    const Decls = CallbackBuilder(
//      fooImpl, .{
//        .{ barImpl, 0 }, // calls bar and continues to reverse on edge 0
//        .{ bazImpl, 1 }, // calls baz and continues to reverse on edge 1
//      }, NoCleanup // indicates no resources to cleanup
//    );
//
// All functions have same number of arguments (can ignore args like _: some_type)
// All functions take the same type of argument
// Each function returns void
//
// If a function allocates resources, it must provide a cleanup function
//

const std = @import("std");
const UT = @import("utility.zig");

pub const NoCleanup = opaque { };

pub fn CallbackBuilder(
    comptime forward: anytype,
    comptime reverse_tuple: anytype,
    comptime cleanup: anytype,
) type {

    if (comptime !UT.isFunction(@TypeOf(forward))){
        @compileError("Reversible field 'func' argument must be a function.");
    }

    // we need an extra field if there is a cleanup provided
    const M: usize = if (comptime @TypeOf(cleanup) == @TypeOf(NoCleanup)) 1 else 2;

    if (comptime M == 2 and !UT.isFunction(@TypeOf(cleanup))) {
        @compileError("Cleanup must be a function, otherwise use NoCleaup.");
    }

    const N: usize = comptime UT.fieldsLen(@TypeOf(reverse_tuple)) + M;

    if (comptime N == M) {
        @compileError("FunctionDeclarations: Empty Reverse Tuple.");
    }
    
    comptime var fields: [N]std.builtin.Type.StructField = undefined;

    fields[0] = .{
        .name = "forward",
        .type = @TypeOf(forward),
        .default_value = forward,
        .is_comptime = true,
        .alignment = 0,
    };

    if (comptime M == 2) {
        fields[1] = .{
            .name = "cleanup",
            .type = @TypeOf(cleanup),
            .default_value = cleanup,
            .is_comptime = true,
            .alignment = 0,
        };
    }

    comptime var suffix: u8 = 'a';
        
    inline for (M..N) |i| {
        const elem = reverse_tuple[i - M];
        
        const FieldType = ReversibleField(elem[0], elem[1]);

        fields[i] = .{ 
            .name = "reverse_" ++ &[_]u8{ suffix },
            .type = FieldType,
            .default_value = &FieldType{ },
            .is_comptime = true,
            .alignment = 0,
        };

        suffix += 1;
    }

    return @Type(.{
        .Struct = .{
            .layout = .Auto,
            .fields = fields[0..],
            .decls = &.{},
            .is_tuple = false,
            .backing_integer = null
        },
    });
}

fn ReversibleField(
    comptime func: anytype,
    comptime edge_index: usize
) type {

    if (comptime !UT.isFunction(@TypeOf(func))){
        @compileError("Reversible field 'func' argument must be a function.");
    }

    const fields: [2]std.builtin.Type.StructField = .{
        std.builtin.Type.StructField {
            .name = "callback",
            .type = @TypeOf(func),
            .default_value = func,
            .is_comptime = true,
            .alignment = 0,
        },
        std.builtin.Type.StructField {
            .name = "edge_index",
            .type = usize,
            .default_value = &edge_index,
            .is_comptime = true,
            .alignment = 0,
        },
    };

    return @Type(.{
        .Struct = .{
            .layout = .Auto,
            .fields = fields[0..],
            .decls = &.{},
            .is_tuple = false,
            .backing_integer = null
        },
    });
}
