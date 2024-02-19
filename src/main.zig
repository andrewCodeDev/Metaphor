
const SCL = @import("scalar.zig");
const mp = @import("metaphor.zig");
const UT = @import("utility.zig");
const dev = @import("device_utils.zig");
const std = @import("std");

pub fn print_c(comptime T: type, x: [*c]const T) void {
    std.debug.print("\nc {*}\n", .{ x });
}

pub fn print_zo(comptime T: type, x: *const T) void {
    std.debug.print("\nzo {*}\n", .{ x });
}

pub fn print_zm(comptime T: type, x: [*]const T) void {
    std.debug.print("\nzm {*}\n", .{ x });
}

pub fn main() !void {

    const G: *mp.Graph = mp.Graph.init(.{
        .optimizer = mp.null_optimizer        
    });

    defer G.deinit();

    const X = G.tensor("X", .inp, .r16, mp.Dims(2){4, 4});
        defer X.free();

    const Y = G.tensor("X", .inp, .r16, mp.Dims(2){4, 4});
        defer Y.free();

    mp.ops.fill(X, 20);
    mp.ops.fill(Y, 19);

}

fn codeDump() void {
    {
        const x = SCL.asScalar(SCL.r16, 42);
        const y = SCL.asScalar(SCL.r32, x);
        const z = SCL.asScalar(SCL.r64, x);
        std.debug.print("\n{}, {}\n", .{ y, z });
    }
    {
        const x = SCL.asScalar(SCL.r16, 42);
        const y = SCL.asScalar(SCL.c32, x);
        const z = SCL.asScalar(SCL.c64, x);
        std.debug.print("\n{}, {}\n", .{ y, z });
    }
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.q8),  @alignOf(SCL.q8)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.r16), @alignOf(SCL.r16)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.r32), @alignOf(SCL.r32)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.r64), @alignOf(SCL.r64)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.c16), @alignOf(SCL.c16)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.c32), @alignOf(SCL.c32)} );
    std.debug.print("\n{s}: {}\n", .{ SCL.scalarName(SCL.c64), @alignOf(SCL.c64)} );

    const a: f32 = 42;

    const cptr: [*c]const f32 = @ptrCast(@alignCast(&a));
    const optr:   *const f32 =  @ptrCast(@alignCast(&a));
    const mptr: [*]const f32 =  @ptrCast(@alignCast(&a));

    print_zo(f32, cptr);
    print_zm(f32, cptr);

    print_c(f32, optr);
    print_c(f32, mptr);
}
