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

pub const NoCleanup = opaque {};
pub const NoArg = opaque {};

pub fn CallbackBuilder(
    comptime forward: anytype,
    comptime reverse_tuple: anytype,
) type {
    if (comptime !UT.isFunction(@TypeOf(forward))) {
        @compileError("Reversible field 'func' argument must be a function.");
    }

    const N: usize = comptime UT.fieldsLen(@TypeOf(reverse_tuple)) + 1;

    comptime var fields: [N]std.builtin.Type.StructField = undefined;

    fields[0] = .{
        .name = "forward",
        .type = @TypeOf(forward),
        .default_value = forward,
        .is_comptime = true,
        .alignment = 0,
    };

    comptime var suffix: u8 = 'a';

    inline for (1..N) |i| {
        const elem = reverse_tuple[i - 1];

        const FieldType = ReversibleField(elem[0], elem[1]);

        fields[i] = .{
            .name = "reverse_" ++ &[_]u8{suffix},
            .type = FieldType,
            .default_value = &FieldType{},
            .is_comptime = true,
            .alignment = 0,
        };

        suffix += 1;
    }

    return @Type(.{ .Struct = .{
        .layout = .auto,
        .fields = fields[0..],
        .decls = &.{},
        .is_tuple = false,
        .backing_integer = null,
    } });
}

pub fn CallbackDropReverse(comptime callback: type, comptime drop_index: usize) type {
    if (comptime drop_index == 0)
        @compileError("Dropping field zero removes forward function.");

    const fields = @typeInfo(callback).Struct.fields;

    if (comptime fields.len <= 2)
        @compileError("Dropping fields will result in empty callback or no reversals.");

    // all - dropped
    const N = fields.len - 1;

    comptime var mod_fields: [N]std.builtin.Type.StructField = undefined;

    mod_fields[0] = fields[0];

    comptime var org_idx: usize = 1;
    comptime var mod_idx: usize = 1;

    while (mod_idx < N) {
        // we offset by -1 to only consider reversals
        if (org_idx == drop_index) {
            org_idx += 1;
            continue;
        }

        mod_fields[mod_idx] = fields[org_idx];

        mod_idx += 1;
        org_idx += 1;
    }

    return @Type(.{ .Struct = .{
        .layout = .auto,
        .fields = mod_fields[0..],
        .decls = &.{},
        .is_tuple = false,
        .backing_integer = null,
    } });
}

fn ReversibleField(comptime func: anytype, comptime edge_index: usize) type {
    if (comptime !UT.isFunction(@TypeOf(func))) {
        @compileError("Reversible field 'func' argument must be a function.");
    }

    const fields: [2]std.builtin.Type.StructField = .{
        std.builtin.Type.StructField{
            .name = "callback",
            .type = @TypeOf(func),
            .default_value = func,
            .is_comptime = true,
            .alignment = 0,
        },
        std.builtin.Type.StructField{
            .name = "edge_index",
            .type = usize,
            .default_value = &edge_index,
            .is_comptime = true,
            .alignment = 0,
        },
    };

    return @Type(.{ .Struct = .{
        .layout = .auto,
        .fields = fields[0..],
        .decls = &.{},
        .is_tuple = false,
        .backing_integer = null,
    } });
}
