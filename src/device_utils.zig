const std = @import("std");
const cuda = @import("cimport.zig").C;
const NonConstPtr = @import("utility.zig").NonConstPtr;

const StreamCtx = cuda.StreamCtx;

const StreamEntry = struct {
    ID: usize,
    context: StreamCtx,
    scratch: struct {
        head: usize = 0,
        tail: usize = 0,
    } = .{},

    // Each stream has it's own scratch memory because streams work
    // like queues. It's safe if the same queue tries to access its
    // own memory, but dangerous if streams can use other scratch.

    pub fn getScratch(self: *StreamEntry, comptime T: type, n: usize) []T {
        const offset: usize = @sizeOf(T) * n;

        // check if we have enough scratch to provide a payload
        if (self.scratch.tail < (self.scratch.head + offset)) {
            if (self.scratch.head != 0)
                free(@as(*anyopaque, @ptrFromInt(self.scratch.head)), self);

            const slice = alloc(T, n, self);
            self.scratch.head = @intFromPtr(slice.ptr);
            self.scratch.tail = self.scratch.head + offset;
        }
        const ptr: [*]T = @ptrFromInt(self.scratch.head);
        return ptr[0..n];
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
            for (0..N) |i| items[i] = initStream();
            return .{ .items = items };
        }

        pub fn synchronize(self: *const Self) void {
            for (0..N) |i| synchronizeStream(self.items[i]);
        }

        pub fn deinit(self: *const Self) void {
            for (0..N) |i| deinitStream(self.items[i]);
        }
    };
}

pub fn initStream() Stream {
    stream_mutex.lock();
    defer stream_mutex.unlock();

    for (0..16) |i| {
        if (stream_array[i] == null) {
            stream_array[i] = .{ .ID = i, .context = cuda.mpInitStream(), .scratch = .{ .head = 0, .tail = 0 } };
            return &(stream_array[i].?);
        }
    }

    @panic("stream limit reached - increase streams");
}

pub fn initDevice(device_number: u32) void {
    cuda.initDevice(device_number);
}

pub fn deinitStream(stream: Stream) void {
    // std.debug.assert(stream != null);
    if (stream.scratch.head > 0) {
        free(@as(*anyopaque, @ptrFromInt(stream.scratch.head)), stream);
        synchronizeStream(stream);
    }

    cuda.mpDeinitStream(stream.context);

    synchronizeDevice();

    stream_array[stream.ID] = null;
}

pub fn synchronizeStream(stream: Stream) void {
    cuda.mpStreamSynchronize(stream.context);
}

pub fn synchronizeDevice() void {
    cuda.mpDeviceSynchronize();
}

pub fn checkLastError() void {
    cuda.mpCheckLastError(); // calls device sync
}

pub fn alloc(comptime T: type, N: usize, stream: Stream) []T {
    // std.debug.assert(stream != null);
    const ptr: *anyopaque = cuda.mpMemAlloc(@sizeOf(T) * N, stream.context) orelse unreachable;
    const out: [*]T = @ptrCast(@alignCast(ptr));
    return out[0..N];
}

pub fn create(comptime T: type, stream: Stream) [*]T {
    //std.debug.assert(stream != null);
    const ptr: *anyopaque = cuda.mpMemAlloc(@sizeOf(T), stream.context) orelse unreachable;
    return @ptrCast(@alignCast(ptr));
}

pub fn copyToDevice(src: anytype, dst: NonConstPtr(@TypeOf(src)), stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyHtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), stream.context);
            } else {
                cuda.mpMemcpyHtoD(dst, src, @sizeOf(T), stream.context);
            }
        },
        else => @compileError("Invalid type for CopyTo: " ++ @typeName(@TypeOf(src))),
    }
}

pub fn copyFromDevice(src: anytype, dst: NonConstPtr(@TypeOf(src)), stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyDtoH(dst.ptr, src.ptr, src.len * @sizeOf(T), stream.context);
            } else {
                cuda.mpMemcpyDtoH(dst, src, @sizeOf(T), stream.context);
            }
        },
        else => @compileError("Invalid type for CopyFrom: " ++ @typeName(@TypeOf(src))),
    }
}

pub fn free(dev_mem: anytype, stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(dev_mem))) {
        .Pointer => |p| {
            if (p.size == .Slice) {
                cuda.mpMemFree(dev_mem.ptr, stream.context);
            } else {
                cuda.mpMemFree(dev_mem, stream.context);
            }
        },
        else => @compileError("Invalid type for Free: " ++ @typeName(@TypeOf(dev_mem))),
    }
}
