const mp = @import("metaphor");
const std = @import("std");
const eu = @import("example_utils.zig");

pub fn main() !void {

    mp.device.init(0);

    const total = mp.device.total_memory(0);

    std.log.info("total: {}", .{ total });

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
        defer G.deinit();

    const x = G.tensor(.{ 
        .class = .wgt,
        .dtype = .r32,
        .sizes = &.{ 2, 3 }
    });

    const y = G.tensor(.{ 
        .class = .wgt,
        .dtype = .r32,
        .sizes = &.{ 3 }
    });

    mp.algo.sequence(x, 1.0, 1.0);
    mp.algo.sequence(y, 1.0, 1.0);

    const z = mp.ops.inner_product(x, y, "ij,j->i");

    z.reverse(.keep);

    const xn = x.native(f32, std.heap.c_allocator);
        defer xn.free(std.heap.c_allocator);

    const yn = y.native(f32, std.heap.c_allocator);
        defer yn.free(std.heap.c_allocator);

    const zn = z.native(f32, std.heap.c_allocator);
        defer zn.free(std.heap.c_allocator);

    eu.print_vector("z", zn.data, zn.sizes);
    eu.print_matrix("x", xn.grad.?, xn.sizes);
    eu.print_vector("y", yn.grad.?, yn.sizes);

    mp.device.check();

    //var xc: [4]f32 = .{ 1, 2, 3, 4 };
    //var yc: [6]f32 = .{ 1, 2, 3, 4, 5, 6 };
    //var zc: [6]f32 = undefined;

    //eu.cpuMatmul(xc[0..], yc[0..], zc[0..], 2, 2, 3);

    //eu.cpuPrintMatrix("cpu", zc[0..], 2, 3);

    //const ddx = dx.derive(x) orelse {
    //    return std.debug.print("\no derivative\n", .{});
    //};
    //mp.util.from_device(ddx.data().r32, z_cpu[0..], stream);
    //mp.stream.sync(stream);
    //std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });

    //mp.util.from_device(x.grad().?.r32, z_cpu[0..], stream);

    //mp.stream.sync(stream);
    //std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });

    //mp.util.from_device(y.grad().?.r32, z_cpu[0..], stream);

    //mp.stream.sync(stream);
    //std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });

    //{
    //    const z = mp.ops.add(x, y);
    //    mp.util.transfer(z, z_cpu[0..], stream);
    //    mp.stream.sync(stream);
    //    std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });
    //}
    //{
    //    const z = mp.ops.sub(x, y);
    //    mp.util.transfer(z, z_cpu[0..], stream);
    //    mp.stream.sync(stream);
    //    std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });
    //}
    //{
    //    const z = mp.ops.hadamard(x, y);
    //    mp.util.transfer(z, z_cpu[0..], stream);
    //    mp.stream.sync(stream);
    //    std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });
    //}
}
