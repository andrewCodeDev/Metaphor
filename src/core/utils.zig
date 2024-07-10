const std = @import("std");
const SC = @import("scalar.zig");
const Tensor = @import("graph.zig").Tensor;
pub const dev = @import("cimport.zig").C;

pub const StreamContext = dev.StreamContext;

const StreamEntry = struct {
    ID: usize,
    context: StreamContext,
    scratch: struct {
        head: usize = 0,
        tail: usize = 0,
    } = .{},

    // Each stream has it's own scratch memory because streams work
    // like queues. It's safe if the same queue tries to access its
    // own memory, but dangerous if streams can use other scratch.

    pub fn get_scratch(self: *StreamEntry, dtype: SC.Tag, n: usize) *anyopaque {

        const byte_size = dtype.size();
        
        const offset: usize = byte_size * n;

        // check if we have enough scratch to provide a payload
        if (self.scratch.tail < (self.scratch.head + offset)) {
            if (self.scratch.head != 0)
                free(@as(*anyopaque, @ptrFromInt(self.scratch.head)), self);

            const ptr = alloc_raw(byte_size * n, self);
            self.scratch.head = @intFromPtr(ptr);
            self.scratch.tail = self.scratch.head + offset;
        }
        return @ptrFromInt(self.scratch.head);
    }
};

pub const MAX_STREAMS: usize = 16;

pub const Stream = *StreamEntry;

var stream_mutex: std.Thread.Mutex = .{};

pub var stream_array: [MAX_STREAMS]?StreamEntry = .{null} ** 16;

pub fn StreamGroup(comptime N: usize) type {
    if (MAX_STREAMS < N) {
        @compileError("Stream group is larger than max streams.");
    }

    return struct {
        const Self = @This();

        items: [N]Stream,

        pub fn init() Self {
            var items: [N]Stream = undefined;
            for (0..N) |i| items[i] = init_stream();
            return .{ .items = items };
        }

        pub fn synchronize(self: *const Self) void {
            for (0..N) |i| synchronize_stream(self.items[i]);
        }

        pub fn deinit(self: *const Self) void {
            for (0..N) |i| deinit_stream(self.items[i]);
        }
    };
}

pub fn init_stream() Stream {
    stream_mutex.lock();
    defer stream_mutex.unlock();

    for (0..16) |i| {
        if (stream_array[i] == null) {
            stream_array[i] = .{ 
                .ID = i,
                .context = dev.mpInitStream(),
                .scratch = .{ .head = 0, .tail = 0 },
            };
            return &(stream_array[i].?);
        }
    }

    @panic("stream limit reached - increase streams");
}

pub fn init_device(device_number: u32) void {
    dev.mpInitDevice(device_number);
}

pub fn total_memory(device_number: u32) usize {
    return dev.mpDeviceTotalMemory(device_number);
}

pub fn deinit_stream(stream: Stream) void {
    // std.debug.assert(stream != null);
    if (stream.scratch.head > 0) {
        free(@as(*anyopaque, @ptrFromInt(stream.scratch.head)), stream);
        synchronize_stream(stream);
    }

    dev.mpDeinitStream(stream.context);

    synchronize_device();

    stream_array[stream.ID] = null;
}

pub fn synchronize_stream(stream: Stream) void {
    dev.mpStreamSynchronize(stream.context);
}

pub fn synchronize_device() void {
    dev.mpDeviceSynchronize();
}

pub fn check_last_error() void {
    dev.mpCheckLastError(); // calls device sync
}

pub fn alloc_raw(bytes: usize, stream: Stream) *anyopaque {
    return dev.mpMemAlloc(bytes, stream.context) orelse {
        check_last_error();
        @panic("Failed to allocate memory.");
    };
}

pub fn alloc(comptime T: type, N: usize, stream: Stream) []T {
    const ptr: *anyopaque = alloc_raw(@sizeOf(T) * N, stream);
    const out: [*]T = @ptrCast(@alignCast(ptr));
    return out[0..N];
}

pub fn create(comptime T: type, stream: Stream) [*]T {
    //std.debug.assert(stream != null);
    const ptr: *anyopaque = dev.mpMemAlloc(@sizeOf(T), stream.context) orelse {
        check_last_error();
        @panic("Failed to create memory.");
    };
    return @ptrCast(@alignCast(ptr));
}

pub fn copy_to_device_raw(
    dtype: SC.Tag,
    src: anytype, 
    dst: *anyopaque,
    len: usize,
    stream: Stream
) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;

            if (dtype != SC.Tag.as_tag(T)) {
                @panic("Mismatched types for copying to the device.");
            }
            if (p.size == .Slice) {
                std.debug.assert(len == src.len);
                dev.mpMemcpyHtoD(dst, src.ptr, len * @sizeOf(T), stream.context);
            } else {
                std.debug.assert(len == 1);
                dev.mpMemcpyHtoD(dst, src, @sizeOf(T), stream.context);
            }
        },
        else => @compileError("Invalid type: " ++ @typeName(@TypeOf(src))),
    }
}

pub fn copy_to_device(
    src: anytype,
    dst: Tensor,
    stream: Stream,
) void {
    return copy_to_device_raw(dst.dtype(), src, dst.data_ptr(), dst.len(), stream);        
}

pub fn copy_from_device_raw(
    dtype: SC.Tag,
    src: *anyopaque,
    dst: anytype, // slice 
    len: usize,
    stream: Stream
) void {

    std.debug.assert(len > 0);
        
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(dst))) {
        .Pointer => |p| {
            const T = p.child;

            if (dtype != SC.Tag.as_tag(T)) {
                @panic("Mismatched types for copying from the device.");
            }
            if (p.size == .Slice) {
                std.debug.assert(len == dst.len);
                dev.mpMemcpyDtoH(dst.ptr, src, len * @sizeOf(T), stream.context);
            } else {
                std.debug.assert(len == 1);
                dev.mpMemcpyDtoH(dst, src, @sizeOf(T), stream.context);
            }
        },
        else => @compileError("Invalid type: " ++ @typeName(@TypeOf(src))),
    }
}

pub fn copy_from_device(
    src: Tensor,
    dst: anytype,
    stream: Stream,
) void {
    return copy_from_device_raw(src.dtype(), src.data_ptr(), dst, src.len(), stream);        
}

pub fn free(dev_mem: anytype, stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(dev_mem))) {
        .Pointer => |p| {
            if (p.size == .Slice) {
                dev.mpMemFree(dev_mem.ptr, stream.context);
            } else {
                dev.mpMemFree(dev_mem, stream.context);
            }
        },
        else => @compileError("Invalid type for Free: " ++ @typeName(@TypeOf(dev_mem))),
    }
}

pub fn free_raw(dev_mem: *anyopaque, stream: Stream) void {
    dev.mpMemFree(dev_mem, stream.context);
}

/////////////////////////////////////
// Contracts for function call sites.

const builtin = @import("builtin");
const debug = (builtin.mode == std.builtin.OptimizeMode.Debug);

pub fn is_pointer(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => true,
        else => false,
    };
}

pub fn is_array(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Array => true,
        else => false,
    };
}

pub fn is_slice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| ptr.size == .Slice,
        else => false,
    };
}

pub inline fn is_integer(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Int, .ComptimeInt => true,
        else => false,
    };
}

pub inline fn is_float(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Float, .ComptimeFloat => true,
        else => false,
    };
}

pub fn fields_len(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.fields.len,
        else => @compileError("fields_len: T must be a struct/tuple type."),
    };
}

pub fn is_struct(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => true,
        else => false,
    };
}

pub fn is_function(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Fn => true,
        else => false,
    };
}

pub fn is_tuple(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.is_tuple,
        else => false,
    };
}

pub fn tuple_size(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .Struct => |s| s.fields.len,
        else => @compileError("Type must be a tuple."),
    };
}

pub fn DeepChild(comptime T: type) type {
    // TODO: consider comptime support, should be Immutable only..

    const C = std.meta.Child(T);

    return switch (@typeInfo(C)) {
        .Array => |a| a.child,
        else => C,
    };
}

pub inline fn ceil(n: anytype, m: @TypeOf(n)) @TypeOf(n) {
    std.debug.assert(m > 0);
    return (n + (m - 1)) / m;
}

pub inline fn swap(x: anytype, y: @TypeOf(x)) void {
    if (comptime !is_pointer(@TypeOf(x))) {
        @compileError("Swap requires pointer types");
    }
    const tmp = x.*;
    x.* = y.*;
    y.* = tmp;
}

pub fn product(comptime T: type, slice: []const T) T {
    var tmp: T = 1;
    for (slice) |val| {
        tmp *= val;
    }
    return tmp;
}

// calculates the number of windows with a given stride that fit in n
pub fn window_count(n: usize, window_size: usize, stride: usize) usize {
    return ((n - window_size) / stride) + 1;
}


