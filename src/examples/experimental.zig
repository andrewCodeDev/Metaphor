const mp = @import("metaphor");
const std = @import("std");

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
        .sizes = &.{ 10 }
    });

    //const y = G.tensor(.{ 
    //    .class = .wgt,
    //    .dtype = .r32,
    //    .sizes = &.{ 10, 10 }
    //});

    mp.algo.fill(x, 1.0);
    //mp.algo.fill(y, 1.0);

    const w = mp.ops.add(x, x);
    const z = mp.ops.broadcast(w, &.{ 10, 10 }, "i->ij");

    const dx = z.derive(x) orelse unreachable;

    var x_cpu: [10]f32 = .{ 0.0 } ** 10;
    mp.util.from_device(dx.data().r32, x_cpu[0..], stream);
    std.debug.print("\nIt worked.\n{any}\n", .{ x_cpu[0..] });

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
