const std = @import("std");
const mp = @import("metaphor");

pub fn copyAndPrintFlat(
    name: []const u8,
    src: anytype,
    stream: anytype
) !void {
    
    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), src.len);
        defer std.heap.c_allocator.free(dst);

    mp.mem.copyFromDevice(dst, src, stream);

    mp.mem.synchronize(stream);

    std.debug.print("\n{s}: {any} " , .{ name, dst });
}

pub fn cpu_print_matrix(
    name: []const u8, 
    src: anytype, 
    row: usize,
    col: usize,
) !void {    
    std.debug.print("\nName: {s}:\n", .{ name });

    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), row * col);
        defer std.heap.c_allocator.free(dst);

    for (0..row) |i| {
        for (0..col) |j| {
            std.debug.print("{d:.1} ", .{ dst[i * col + j]});
        }
        std.debug.print("\n", .{});
    }
}

pub fn copyAndPrintMatrix(
    name: []const u8, 
    src: anytype, 
    row: usize,
    col: usize,
    stream: anytype
) !void {    
    std.debug.assert(src.len == row * col);
    
    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), src.len);
        defer std.heap.c_allocator.free(dst);

    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\nName: {s}:\n", .{ name });

    for (0..row) |i| {
        for (0..col) |j| {
            std.debug.print("{d:.1} ", .{ dst[i * col + j] });
        }
        std.debug.print("\n", .{});
    }
}

pub fn cpu_matmul(
    x: anytype,
    y: @TypeOf(x),
    z: @TypeOf(x),
    M: usize,
    N: usize,
    K: usize
) void {
    std.debug.assert(x.len == M * N);
    std.debug.assert(y.len == N * K);
    std.debug.assert(z.len == M * K);

    for (0..M) |m| {

        for (0..K) |k| {

            var tmp: f32 = 0;

            for (0..N) |n| {
                tmp += x[m * N + n] * y[n * K + k];
            }
            z[m * K + k] = tmp;
        }
    }
}

pub fn verify_results(
    name: []const u8,
    x: anytype,
    y: @TypeOf(x),
) void {
    std.debug.assert(x.len == y.len);

    var good: bool = true;

    for (x, y) |a, b| {
        if (@abs(a - b) > 0.0001) {
            std.log.err("Verification FAILURE: {s}", .{name});
            good = false;
            break;
        }
    }
    if (good) {    
        std.log.info("Verification SUCCESS: {s}", .{name});
    }
}

