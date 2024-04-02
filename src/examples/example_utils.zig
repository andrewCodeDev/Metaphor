const std = @import("std");
const mp = @import("metaphor");

pub fn copyAndPrintFlat(name: []const u8, src: anytype, stream: anytype) !void {
    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), src.len);
    defer std.heap.c_allocator.free(dst);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    std.debug.print("\n{s}: {any} ", .{ name, dst });
}

pub fn copyAndPrintKeyValues(src: anytype, keys: []mp.types.Key, stream: anytype) !void {
    std.debug.assert(src.len == keys.len);
    
    const cpu_src = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), keys.len);
    defer std.heap.c_allocator.free(cpu_src);

    const cpu_keys = try std.heap.c_allocator.alloc(mp.types.Key, keys.len);
    defer std.heap.c_allocator.free(cpu_keys);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(keys, cpu_keys, stream);
    mp.mem.copyFromDevice(src, cpu_src, stream);

    mp.stream.synchronize(stream);

    for (cpu_keys) |k| {
        std.debug.print("{}: {d:.4}\n", .{ k, mp.scalar.as(f32, cpu_src[k]) });
    }
}

pub fn copyAndPrintKey(keys: []mp.types.Key, stream: anytype) !void {
    
    const cpu_keys = try std.heap.c_allocator.alloc(mp.types.Key, keys.len);
    defer std.heap.c_allocator.free(cpu_keys);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(keys, cpu_keys, stream);

    mp.stream.synchronize(stream);

    for (cpu_keys) |k| {
        std.debug.print("{}\n", .{ k });
    }
}

pub fn copyAndValidateKeys(src: anytype, keys: []mp.types.Key, stream: anytype) !void {
    std.debug.assert(src.len == keys.len);
    
    const cpu_src = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), keys.len);
    defer std.heap.c_allocator.free(cpu_src);

    const cpu_keys = try std.heap.c_allocator.alloc(mp.types.Key, keys.len);
    defer std.heap.c_allocator.free(cpu_keys);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(keys, cpu_keys, stream);
    mp.mem.copyFromDevice(src, cpu_src, stream);

    mp.stream.synchronize(stream);

    var last: f32 = -std.math.inf(f32);
    for (cpu_keys) |k| {
        std.debug.assert(cpu_src[k] >= last);
        last = cpu_src[k];
    }
}

pub fn copyToCPU(src: anytype, stream: anytype) ![]std.meta.Child(@TypeOf(src)) {
    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), src.len);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    return dst;
}

pub fn freeCPU(src: anytype) void {
    std.heap.c_allocator.free(src);
}

pub fn allocCPU(comptime T: type, N: usize) ![]T {
    return try std.heap.c_allocator.alloc(T, N);
}

pub fn cpuPrintMatrix(name: []const u8, src: anytype, row: usize, col: usize) void {
    std.debug.print("\nName: {s}:\n", .{name});

    for (0..row) |i| {
        for (0..col) |j| {
            std.debug.print("{d:.4} ", .{src[i * col + j]});
        }
        std.debug.print("\n", .{});
    }
}

pub fn copyAndPrintMatrix(name: []const u8, src: anytype, row: usize, col: usize, stream: anytype) !void {
    std.debug.assert(src.len == row * col);

    const dst = try std.heap.c_allocator.alloc(std.meta.Child(@TypeOf(src)), src.len);
    defer std.heap.c_allocator.free(dst);

    mp.stream.synchronize(stream);

    mp.mem.copyFromDevice(src, dst, stream);

    mp.stream.synchronize(stream);

    cpuPrintMatrix(name, dst, row, col);
}

pub fn cpuMatmul(x: anytype, y: @TypeOf(x), z: @TypeOf(x), M: usize, N: usize, K: usize) void {
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

pub fn cpuTranspose(x: anytype, y: @TypeOf(x), M: usize, N: usize) void {
    std.debug.assert(x.len == M * N);
    std.debug.assert(y.len == N * N);

    for (0..M) |i| {
        for (0..N) |j| {
            y[j * M + i] = x[i * N + j];
        }
    }
}

pub fn cpuAdd(x: anytype, y: @TypeOf(x), z: @TypeOf(x)) void {
    std.debug.assert(x.len == y.len);
    std.debug.assert(y.len == z.len);

    for (0..x.len) |i| {
        z[i] = x[i] + y[i];
    }
}

pub fn verifyResults(name: []const u8, x: anytype, y: @TypeOf(x)) void {
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
