const DU = @import("device_utils.zig");
const Child = @import("utility.zig").Child;
const std = @import("std");
const TC = @import("tensor_components.zig");

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

pub fn loadTensorToGraph(
    dir: []const u8,
    prefix: []const u8,
    x_idx: TC.IndexType,
    x_gpu: []u8,
    x_cpu: []u8,
    stream: DU.Stream,
) !void {

    ////////////////////////////////////////////////
    // Create our file name and path ///////////////
    var name = try std.BoundedArray(u8, 128).init(0);

    try std.fmt.format(name.writer(), "{s}_{}", .{ prefix, x_idx });

    const path = try std.fs.path.join(std.heap.c_allocator, &.{ dir, name.slice() });
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Read our tensor file as raw bytes ///////////

    var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });

    if (x_gpu.len != try file.readAll(x_cpu)) 
        return SaveError.WrongByteCount;

    DU.copyToDevice(x_cpu, x_gpu, stream);

    DU.synchronizeStream(stream);
}

pub fn saveTensorFromGraph(
    dir: []const u8,
    prefix: []const u8,
    x_idx: TC.IndexType,
    x_gpu: []u8,
    x_cpu: []u8,
    stream: DU.Stream,
) !void {

    ////////////////////////////////////////////////
    // Create our file name and path ///////////////

    var name = try std.BoundedArray(u8, 128).init(0);

    try std.fmt.format(name.writer(), "{s}_{}", .{ prefix, x_idx });

    const path = try std.fs.path.join(std.heap.c_allocator, &.{ dir, name.slice() });
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Write our tensor file as raw bytes //////////
    
    DU.copyFromDevice(x_gpu, x_cpu, stream);

    DU.synchronizeStream(stream);

    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    try file.writeAll(x_cpu);
}

//////////////////////////////////////////////////////////////////////
// These functions requires a full filename. They're for special cases 
// where the user wants to load in something like an input tensor
// independent of the network. Word embeddings, for example.

pub fn saveTensor(
    dir: []const u8,
    filename: []const u8,
    x: anytype,
    stream: DU.Stream,
) void {
    const path = std.fs.path.join(std.heap.c_allocator, &.{ dir, filename }) catch @panic("Failed to create save path");
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Write our tensor file as raw bytes //////////

    const x_bytes = x.raw_values().bytes();

    std.debug.assert(x_bytes.len > 0);

    const x_cpu = std.heap.c_allocator.alloc(u8, x_bytes.len) catch @panic("Failed to allocate save buffer");
    defer std.heap.c_allocator.free(x_cpu);
    
    DU.copyFromDevice(x_bytes, x_cpu, stream);

    DU.synchronizeStream(stream);

    var file = std.fs.cwd().createFile(path, .{ .truncate = true }) catch @panic("Failed to create save file");
    defer file.close();

    file.writeAll(x_cpu) catch @panic("Failed to write to save file");
}

pub fn loadTensor(
    dir: []const u8,
    filename: []const u8,
    x: anytype,
    stream: DU.Stream,
) void {
    const path = std.fs.path.join(std.heap.c_allocator, &.{ dir, filename }) catch @panic("Failed to create load path");
    defer std.heap.c_allocator.free(path);

    ////////////////////////////////////////////////
    // Read our tensor file as raw bytes ///////////

    const x_bytes = x.raw_values().bytes();

    std.debug.assert(x_bytes.len > 0);

    const x_cpu = std.heap.c_allocator.alloc(u8, x_bytes.len) catch @panic("Failed to allocator load buffer.");
    defer std.heap.c_allocator.free(x_cpu);
    
    var file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch @panic("Failed to open load file.");

    if (x_bytes.len != file.readAll(x_cpu) catch @panic("Failed to read into load buffer."))
        @panic("Byte count loaded does not match provided tensor size.");

    DU.copyToDevice(x_cpu, x_bytes, stream);

    DU.synchronizeStream(stream);
}
