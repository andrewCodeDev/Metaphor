const mp = @import("metaphor");
const std = @import("std");

pub fn main() !void {
    mp.device.init(0);

    const stream = mp.stream.init();
        defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
        defer G.deinit();

    const x = G.tensor(.{ 
        .class = .inp,
        .dtype = .r32,
        .sizes = &.{ 10, 10 }
    });
    const y = G.tensor(.{ 
        .class = .inp,
        .dtype = .r32,
        .sizes = &.{ 10, 10 }
    });

    mp.algo.fill(x, 4.0);
    mp.algo.fill(y, 5.0);

    var z_cpu: [100]f32 = .{ 0.0 } ** 100;

    const z = mp.ops.add(x, x);

    const dx = z.derive(x) orelse unreachable;
    mp.util.from_device(dx.data().r32, z_cpu[0..], stream);
    mp.stream.sync(stream);
    std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });

    const ddx = dx.derive(x) orelse {
        return std.debug.print("\no derivative\n", .{});
    };
    mp.util.from_device(ddx.data().r32, z_cpu[0..], stream);
    mp.stream.sync(stream);
    std.debug.print("\nIt worked.\n{any}\n", .{ z_cpu[0..] });

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
