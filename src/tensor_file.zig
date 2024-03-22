const DU = @import("device_utils.zig");
const Child = @import("utility.zig").Child;
const std = @import("std");

// These are basic tensor utilities for saving data to the disk.
// Tensor values are read/written as direct binary files.

// The file naming scheme just based on the prefix and tensor index.
// Prefixes allow multiple graphs to be saved to the same place without
// collisions on the index. Otherwise, two tensors with the same index
// could overwrite eachother.

// example: prefix="p1", idx=0 -> "p1_0"
// example: prefix="p2", idx=0 -> "p2_0"

// Directory is relative to where the project is being run from. It also
// accepts absolute paths, too.

// it would suck if you finished training and couldn't save your weights
// but the rest of the API doesn't currently return errors, so this
// probably needs to be considered more carefully.
const SaveError = error { WrongByteCount };

pub fn loadTensor(
    dir: []const u8,
    prefix: []const u8,
    x: anytype,
    stream: DU.Stream,
) !void {
    const T = Child(@TypeOf(x));

    ////////////////////////////////////////////////
    // Create our file name and path ///////////////
    var name = try std.BoundedArray(u8, 128).init(0);

    try std.fmt.format(name.writer(), "{s}_{}", .{ prefix, x.idx });

    const path = try std.fs.path.join(std.heap.c_allocator, &.{ dir, name.slice() });
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Read our tensor file as raw bytes ///////////

    var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });

    const N = x.len() * @sizeOf(T);

    const values_cpu = try std.heap.c_allocator.alloc(u8, N);
    defer std.heap.c_allocator.free(values_cpu);

    if (N != try file.readAll(values_cpu)) 
        return SaveError.WrongByteCount;

    DU.copyToDevice(values_cpu, std.mem.sliceAsBytes(x.values()), stream);

    DU.synchronizeStream(stream);
}


pub fn saveTensor(
    dir: []const u8,
    prefix: []const u8,
    x: anytype,
    stream: DU.Stream,
) !void {
    const T = Child(@TypeOf(x));

    ////////////////////////////////////////////////
    // Create our file name and path ///////////////
    var name = try std.BoundedArray(u8, 128).init(0);

    try std.fmt.format(name.writer(), "{s}_{}", .{ prefix, x.idx });

    const path = try std.fs.path.join(std.heap.c_allocator, &.{ dir, name.slice() });
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Write our tensor file as raw bytes //////////

    const values_cpu = try std.heap.c_allocator.alloc(T, x.len());
    defer std.heap.c_allocator.free(values_cpu);
    
    DU.copyFromDevice(x.values(), values_cpu, stream);

    DU.synchronizeStream(stream);

    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    try file.writeAll(std.mem.sliceAsBytes(values_cpu));
}

